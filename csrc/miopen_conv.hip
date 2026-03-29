#include <torch/extension.h>
#include <miopen/miopen.h>

// Forward-declare getMiopenHandle to avoid pulling in ATen/hip/HIPContext.h
// which transitively includes thrust headers that may not be on the include path.
namespace at { namespace native {
    miopenHandle_t getMiopenHandle();
}}

#include <unordered_map>
#include <mutex>
#include <vector>
#include <tuple>

// ---- Helpers ----

static miopenHandle_t get_handle() {
    return at::native::getMiopenHandle();
}

static miopenDataType_t get_miopen_dtype(at::ScalarType type) {
    switch (type) {
        case at::kFloat: return miopenFloat;
        case at::kHalf: return miopenHalf;
        case at::kBFloat16: return miopenBFloat16;
        default:
            TORCH_CHECK(false, "Unsupported dtype for MIOpen conv: ", type);
    }
}

// ---- Algorithm cache ----

struct ConvKey {
    std::vector<int64_t> input_shape;
    std::vector<int64_t> weight_shape;
    std::vector<int> stride;
    std::vector<int> padding;
    std::vector<int> dilation;
    int groups;
    miopenDataType_t dtype;
    bool transposed;

    bool operator==(const ConvKey& other) const {
        return input_shape == other.input_shape &&
               weight_shape == other.weight_shape &&
               stride == other.stride &&
               padding == other.padding &&
               dilation == other.dilation &&
               groups == other.groups &&
               dtype == other.dtype &&
               transposed == other.transposed;
    }
};

struct ConvKeyHash {
    size_t operator()(const ConvKey& k) const {
        size_t h = std::hash<int>()(k.groups) ^ std::hash<int>()(k.dtype) ^
                   std::hash<bool>()(k.transposed);
        for (auto v : k.input_shape) h ^= std::hash<int64_t>()(v) * 31;
        for (auto v : k.weight_shape) h ^= std::hash<int64_t>()(v) * 37;
        for (auto v : k.stride) h ^= std::hash<int>()(v) * 41;
        for (auto v : k.padding) h ^= std::hash<int>()(v) * 43;
        for (auto v : k.dilation) h ^= std::hash<int>()(v) * 47;
        return h;
    }
};

struct CachedAlgo {
    miopenConvFwdAlgorithm_t fwd_algo;
    miopenConvBwdDataAlgorithm_t bwd_data_algo;
    size_t workspace_size;
    bool valid = false;
};

static std::unordered_map<ConvKey, CachedAlgo, ConvKeyHash> algo_cache;
static std::mutex cache_mutex;

// ---- RAII descriptor wrappers ----

struct TensorDesc {
    miopenTensorDescriptor_t desc;
    TensorDesc() { miopenCreateTensorDescriptor(&desc); }
    ~TensorDesc() { miopenDestroyTensorDescriptor(desc); }
    void set4d(miopenDataType_t dt, int n, int c, int h, int w) {
        miopenSet4dTensorDescriptor(desc, dt, n, c, h, w);
    }
};

struct ConvDesc {
    miopenConvolutionDescriptor_t desc;
    ConvDesc() { miopenCreateConvolutionDescriptor(&desc); }
    ~ConvDesc() { miopenDestroyConvolutionDescriptor(desc); }
    void set2d(int pad_h, int pad_w, int stride_h, int stride_w,
               int dilation_h, int dilation_w, int groups) {
        miopenInitConvolutionDescriptor(desc, miopenConvolution,
                                        pad_h, pad_w, stride_h, stride_w,
                                        dilation_h, dilation_w);
        miopenSetConvolutionGroupCount(desc, groups);
    }
};

// ---- Conv1d forward with workspace ----

torch::Tensor miopen_conv1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups
) {
    TORCH_CHECK(input.dim() == 3, "Expected 3D input (N, C, L)");
    TORCH_CHECK(weight.dim() == 3, "Expected 3D weight (C_out, C_in/g, K)");

    auto handle = get_handle();
    auto dtype = get_miopen_dtype(input.scalar_type());

    int N = input.size(0);
    int C_in = input.size(1);
    int L_in = input.size(2);
    int C_out = weight.size(0);
    int K = weight.size(2);

    int s = stride[0];
    int p = padding[0];
    int d = dilation[0];
    int g = groups;

    // Reshape 1D → 2D (H=1)
    auto input_2d = input.unsqueeze(2);   // (N, C_in, 1, L_in)
    auto weight_2d = weight.unsqueeze(2); // (C_out, C_in/g, 1, K)

    // Compute output length
    int L_out = (L_in + 2 * p - d * (K - 1) - 1) / s + 1;

    // Create output tensor
    auto output_2d = torch::empty({N, C_out, 1, L_out}, input.options());

    // Setup descriptors
    TensorDesc x_desc, w_desc, y_desc;
    ConvDesc conv_desc;

    x_desc.set4d(dtype, N, C_in, 1, L_in);
    w_desc.set4d(dtype, C_out, C_in / g, 1, K);
    y_desc.set4d(dtype, N, C_out, 1, L_out);
    conv_desc.set2d(0, p, 1, s, 1, d, g);

    // Build cache key
    ConvKey key;
    key.input_shape = {N, C_in, L_in};
    key.weight_shape = {C_out, C_in / g, K};
    key.stride = {s};
    key.padding = {p};
    key.dilation = {d};
    key.groups = g;
    key.dtype = dtype;
    key.transposed = false;

    // Lookup or find algorithm
    CachedAlgo cached;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = algo_cache.find(key);
        if (it != algo_cache.end()) {
            cached = it->second;
        }
    }

    if (!cached.valid) {
        // Query workspace size
        size_t ws_size = 0;
        miopenConvolutionForwardGetWorkSpaceSize(
            handle, w_desc.desc, x_desc.desc, conv_desc.desc, y_desc.desc, &ws_size);

        // Allocate workspace
        auto workspace = torch::empty({static_cast<int64_t>(ws_size)},
                                       input.options().dtype(torch::kByte));

        // Find best algorithm
        int algo_count = 0;
        miopenConvAlgoPerf_t perf[4];
        miopenFindConvolutionForwardAlgorithm(
            handle, x_desc.desc, input_2d.data_ptr(),
            w_desc.desc, weight_2d.data_ptr(),
            conv_desc.desc,
            y_desc.desc, output_2d.data_ptr(),
            4, &algo_count,
            perf,
            workspace.data_ptr(), ws_size, false);

        TORCH_CHECK(algo_count > 0, "MIOpen found no conv forward algorithms");

        cached.fwd_algo = perf[0].fwd_algo;
        cached.workspace_size = perf[0].memory;
        cached.valid = true;

        std::lock_guard<std::mutex> lock(cache_mutex);
        algo_cache[key] = cached;
    }

    // Allocate workspace for execution
    auto workspace = torch::empty({static_cast<int64_t>(cached.workspace_size)},
                                   input.options().dtype(torch::kByte));

    // Execute convolution
    float alpha = 1.0f, beta = 0.0f;
    miopenConvolutionForward(
        handle, &alpha,
        x_desc.desc, input_2d.data_ptr(),
        w_desc.desc, weight_2d.data_ptr(),
        conv_desc.desc, cached.fwd_algo,
        &beta,
        y_desc.desc, output_2d.data_ptr(),
        workspace.data_ptr(), cached.workspace_size);

    // Reshape back to 1D
    auto output = output_2d.squeeze(2);

    // Add bias if present
    if (bias.has_value() && bias->defined()) {
        output = output + bias->unsqueeze(0).unsqueeze(2);
    }

    return output;
}

// ---- ConvTranspose1d forward with workspace ----
// ConvTranspose1d forward = miopenConvolutionBackwardData

torch::Tensor miopen_conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups,
    std::vector<int64_t> dilation
) {
    TORCH_CHECK(input.dim() == 3, "Expected 3D input (N, C, L)");
    TORCH_CHECK(weight.dim() == 3, "Expected 3D weight (C_in, C_out/g, K)");

    auto handle = get_handle();
    auto dtype = get_miopen_dtype(input.scalar_type());

    int N = input.size(0);
    int C_in = input.size(1);
    int L_in = input.size(2);
    int C_out = weight.size(1) * groups;
    int K = weight.size(2);

    int s = stride[0];
    int p = padding[0];
    int op = output_padding[0];
    int d = dilation[0];
    int g = groups;

    // Compute output length
    int L_out = (L_in - 1) * s - 2 * p + d * (K - 1) + op + 1;

    // Reshape 1D → 2D (H=1)
    auto input_2d = input.unsqueeze(2);
    auto weight_2d = weight.unsqueeze(2);
    auto output_2d = torch::empty({N, C_out, 1, L_out}, input.options());

    // For ConvTranspose, the descriptors are reversed:
    // "input" to backward-data is the transposed conv's input
    // "output" of backward-data (dy) is the transposed conv's output
    TensorDesc dy_desc, w_desc, dx_desc;
    ConvDesc conv_desc;

    dy_desc.set4d(dtype, N, C_in, 1, L_in);
    w_desc.set4d(dtype, C_in / g, C_out / g, 1, K);
    dx_desc.set4d(dtype, N, C_out, 1, L_out);
    conv_desc.set2d(0, p, 1, s, 1, d, g);

    // Build cache key
    ConvKey key;
    key.input_shape = {N, C_in, L_in};
    key.weight_shape = {C_in / g, C_out / g, K};
    key.stride = {s};
    key.padding = {p};
    key.dilation = {d};
    key.groups = g;
    key.dtype = dtype;
    key.transposed = true;

    CachedAlgo cached;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = algo_cache.find(key);
        if (it != algo_cache.end()) {
            cached = it->second;
        }
    }

    if (!cached.valid) {
        size_t ws_size = 0;
        miopenConvolutionBackwardDataGetWorkSpaceSize(
            handle, dy_desc.desc, w_desc.desc, conv_desc.desc, dx_desc.desc, &ws_size);

        auto workspace = torch::empty({static_cast<int64_t>(ws_size)},
                                       input.options().dtype(torch::kByte));

        int algo_count = 0;
        miopenConvAlgoPerf_t perf[4];
        miopenFindConvolutionBackwardDataAlgorithm(
            handle, dy_desc.desc, input_2d.data_ptr(),
            w_desc.desc, weight_2d.data_ptr(),
            conv_desc.desc,
            dx_desc.desc, output_2d.data_ptr(),
            4, &algo_count,
            perf,
            workspace.data_ptr(), ws_size, false);

        TORCH_CHECK(algo_count > 0, "MIOpen found no conv backward-data algorithms");

        cached.bwd_data_algo = perf[0].bwd_data_algo;
        cached.workspace_size = perf[0].memory;
        cached.valid = true;

        std::lock_guard<std::mutex> lock(cache_mutex);
        algo_cache[key] = cached;
    }

    auto workspace = torch::empty({static_cast<int64_t>(cached.workspace_size)},
                                   input.options().dtype(torch::kByte));

    float alpha = 1.0f, beta = 0.0f;
    miopenConvolutionBackwardData(
        handle, &alpha,
        dy_desc.desc, input_2d.data_ptr(),
        w_desc.desc, weight_2d.data_ptr(),
        conv_desc.desc, cached.bwd_data_algo,
        &beta,
        dx_desc.desc, output_2d.data_ptr(),
        workspace.data_ptr(), cached.workspace_size);

    auto output = output_2d.squeeze(2);

    if (bias.has_value() && bias->defined()) {
        output = output + bias->unsqueeze(0).unsqueeze(2);
    }

    return output;
}
