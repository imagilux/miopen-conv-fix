#include <torch/extension.h>
#include <miopen/miopen.h>
#include <hip/hip_runtime.h>

// Forward-declare getMiopenHandle to avoid pulling in ATen/hip/HIPContext.h
// which transitively includes thrust headers that may not be on the include path.
namespace at { namespace native {
    miopenHandle_t getMiopenHandle();
}}

#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <vector>
#include <tuple>
#include <functional>

// ---- MIOpen error checking ----

#define MIOPEN_CHECK(expr)                                                    \
    do {                                                                      \
        miopenStatus_t _status = (expr);                                      \
        if (_status != miopenStatusSuccess) {                                 \
            throw std::runtime_error(                                         \
                std::string("MIOpen error: ") + miopenGetErrorString(_status) \
                + " in " #expr);                                              \
        }                                                                     \
    } while (0)

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

// ---- GPU architecture detection ----

static std::string get_gcn_arch_name() {
    static std::mutex arch_mutex;
    static std::unordered_map<int, std::string> arch_cache;

    int device = 0;
    (void)hipGetDevice(&device);

    std::lock_guard<std::mutex> lock(arch_mutex);
    auto it = arch_cache.find(device);
    if (it != arch_cache.end()) return it->second;

    hipDeviceProp_t props;
    (void)hipGetDeviceProperties(&props, device);
    std::string arch(props.gcnArchName);
    // Normalize: "gfx1201:sramecc+:xnack-" -> "gfx1201"
    auto colon = arch.find(':');
    if (colon != std::string::npos) arch = arch.substr(0, colon);
    arch_cache[device] = arch;
    return arch;
}

// ---- Tensor materialization ----
// Parametrized tensors (weight_norm), TorchScript outputs, torch.compile
// functionalized tensors, and certain views may not have accessible storage.
//
// The only guaranteed fix: allocate a fresh tensor and copy_ into it.
// This forces a real HIP allocation + a memcpy, no subclass wrappers.

static torch::Tensor force_materialize(const torch::Tensor& t) {
    auto fresh = torch::empty(t.sizes(), t.options().memory_format(torch::MemoryFormat::Contiguous));
    fresh.copy_(t);
    return fresh;
}

static torch::Tensor ensure_data_ptr(const torch::Tensor& t) {
    try {
        if (t.defined() && t.is_contiguous() && t.has_storage() &&
            t.storage().data_ptr().get() != nullptr) {
            (void)t.data_ptr();
            return t;
        }
    } catch (...) {}
    return force_materialize(t);
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
    uint64_t solution_id;
    size_t workspace_size;
    bool valid = false;
};

static std::unordered_map<ConvKey, CachedAlgo, ConvKeyHash> algo_cache;
static std::mutex cache_mutex;

// ---- Per-architecture solver blacklist ----

static std::unordered_map<std::string, std::unordered_set<uint64_t>> solver_blacklist;
static std::mutex blacklist_mutex;

static bool is_solution_blacklisted(const std::string& arch, uint64_t solution_id) {
    std::lock_guard<std::mutex> lock(blacklist_mutex);
    auto it = solver_blacklist.find(arch);
    if (it != solver_blacklist.end() && it->second.count(solution_id)) return true;
    it = solver_blacklist.find("*");
    if (it != solver_blacklist.end() && it->second.count(solution_id)) return true;
    return false;
}

void add_solver_blacklist(const std::string& arch, uint64_t solution_id) {
    std::lock_guard<std::mutex> lock(blacklist_mutex);
    solver_blacklist[arch].insert(solution_id);
}

void clear_solver_blacklist() {
    std::lock_guard<std::mutex> lock(blacklist_mutex);
    solver_blacklist.clear();
}

void clear_algo_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    algo_cache.clear();
}

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

// ---- Immediate Mode solution selection ----
//
// Instead of miopenFind*Algorithm (which runs benchmark kernels that can
// segfault on immature architectures like gfx1201), we use the Immediate
// Mode API: enumerate solutions by heuristic ranking, pick the best
// non-blacklisted one, compile it, and cache the selection.

struct SolutionCallbacks {
    std::function<miopenStatus_t(size_t*)> get_count;
    std::function<miopenStatus_t(size_t, size_t*, miopenConvSolution_t*)> get_solutions;
    std::function<miopenStatus_t(uint64_t)> compile;
};

static CachedAlgo select_solution(
    const ConvKey& key,
    const SolutionCallbacks& cb
) {
    std::string arch = get_gcn_arch_name();

    size_t solution_count = 0;
    MIOPEN_CHECK(cb.get_count(&solution_count));
    if (solution_count == 0) {
        throw std::runtime_error(
            "MIOpen found no solutions for this conv configuration on " + arch);
    }

    std::vector<miopenConvSolution_t> solutions(solution_count);
    size_t returned = 0;
    MIOPEN_CHECK(cb.get_solutions(solution_count, &returned, solutions.data()));
    solutions.resize(returned);

    // Pick the best non-blacklisted solution
    for (size_t i = 0; i < solutions.size(); ++i) {
        uint64_t sid = solutions[i].solution_id;

        if (is_solution_blacklisted(arch, sid)) continue;

        miopenStatus_t compile_status = cb.compile(sid);
        if (compile_status != miopenStatusSuccess) continue;

        CachedAlgo cached;
        cached.solution_id = sid;
        cached.workspace_size = solutions[i].workspace_size;
        cached.valid = true;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            algo_cache[key] = cached;
        }
        return cached;
    }

    throw std::runtime_error(
        "MIOpen: all " + std::to_string(solutions.size()) +
        " solutions blacklisted or failed to compile on " + arch +
        ". Falling back to PyTorch native conv.");
}

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

    auto input_c = ensure_data_ptr(input);
    auto weight_c = ensure_data_ptr(weight);

    auto input_2d = input_c.view({N, C_in, 1, L_in});
    auto weight_2d = weight_c.view({C_out, C_in / g, 1, K});

    int L_out = (L_in + 2 * p - d * (K - 1) - 1) / s + 1;
    auto output_2d = torch::empty({N, C_out, 1, L_out}, input_c.options());

    TensorDesc x_desc, w_desc, y_desc;
    ConvDesc conv_desc;

    x_desc.set4d(dtype, N, C_in, 1, L_in);
    w_desc.set4d(dtype, C_out, C_in / g, 1, K);
    y_desc.set4d(dtype, N, C_out, 1, L_out);
    conv_desc.set2d(0, p, 1, s, 1, d, g);

    ConvKey key;
    key.input_shape = {N, C_in, L_in};
    key.weight_shape = {C_out, C_in / g, K};
    key.stride = {s};
    key.padding = {p};
    key.dilation = {d};
    key.groups = g;
    key.dtype = dtype;
    key.transposed = false;

    // Lookup cache
    CachedAlgo cached;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = algo_cache.find(key);
        if (it != algo_cache.end()) cached = it->second;
    }

    if (!cached.valid) {
        SolutionCallbacks cb;
        cb.get_count = [&](size_t* count) {
            return miopenConvolutionForwardGetSolutionCount(
                handle, w_desc.desc, x_desc.desc, conv_desc.desc, y_desc.desc, count);
        };
        cb.get_solutions = [&](size_t max, size_t* ret, miopenConvSolution_t* sols) {
            return miopenConvolutionForwardGetSolution(
                handle, w_desc.desc, x_desc.desc, conv_desc.desc, y_desc.desc, max, ret, sols);
        };
        cb.compile = [&](uint64_t sid) {
            return miopenConvolutionForwardCompileSolution(
                handle, w_desc.desc, x_desc.desc, conv_desc.desc, y_desc.desc, sid);
        };

        cached = select_solution(key, cb);
    }

    // Execute with workspace
    auto workspace = torch::empty({static_cast<int64_t>(cached.workspace_size)},
                                   input_c.options().dtype(torch::kByte));
    MIOPEN_CHECK(miopenConvolutionForwardImmediate(
        handle, w_desc.desc, weight_2d.data_ptr(),
        x_desc.desc, input_2d.data_ptr(),
        conv_desc.desc,
        y_desc.desc, output_2d.data_ptr(),
        workspace.data_ptr(), cached.workspace_size,
        cached.solution_id));

    auto output = output_2d.squeeze(2);

    if (bias.has_value() && bias->defined()) {
        auto bias_c = ensure_data_ptr(*bias);
        output = output + bias_c.unsqueeze(0).unsqueeze(2);
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

    int L_out = (L_in - 1) * s - 2 * p + d * (K - 1) + op + 1;

    auto input_c = ensure_data_ptr(input);
    auto weight_c = ensure_data_ptr(weight);

    auto input_2d = input_c.view({N, C_in, 1, L_in});
    auto weight_2d = weight_c.view({C_in / g, C_out / g, 1, K});
    auto output_2d = torch::empty({N, C_out, 1, L_out}, input_c.options());

    // For ConvTranspose, descriptors are reversed:
    // "dy" = transposed conv's input, "dx" = transposed conv's output
    TensorDesc dy_desc, w_desc, dx_desc;
    ConvDesc conv_desc;

    dy_desc.set4d(dtype, N, C_in, 1, L_in);
    w_desc.set4d(dtype, C_in / g, C_out / g, 1, K);
    dx_desc.set4d(dtype, N, C_out, 1, L_out);
    conv_desc.set2d(0, p, 1, s, 1, d, g);

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
        if (it != algo_cache.end()) cached = it->second;
    }

    if (!cached.valid) {
        SolutionCallbacks cb;
        cb.get_count = [&](size_t* count) {
            return miopenConvolutionBackwardDataGetSolutionCount(
                handle, dy_desc.desc, w_desc.desc, conv_desc.desc, dx_desc.desc, count);
        };
        cb.get_solutions = [&](size_t max, size_t* ret, miopenConvSolution_t* sols) {
            return miopenConvolutionBackwardDataGetSolution(
                handle, dy_desc.desc, w_desc.desc, conv_desc.desc, dx_desc.desc, max, ret, sols);
        };
        cb.compile = [&](uint64_t sid) {
            return miopenConvolutionBackwardDataCompileSolution(
                handle, dy_desc.desc, w_desc.desc, conv_desc.desc, dx_desc.desc, sid);
        };

        cached = select_solution(key, cb);
    }

    // Execute with workspace
    auto workspace = torch::empty({static_cast<int64_t>(cached.workspace_size)},
                                   input_c.options().dtype(torch::kByte));
    MIOPEN_CHECK(miopenConvolutionBackwardDataImmediate(
        handle, dy_desc.desc, input_2d.data_ptr(),
        w_desc.desc, weight_2d.data_ptr(),
        conv_desc.desc,
        dx_desc.desc, output_2d.data_ptr(),
        workspace.data_ptr(), cached.workspace_size,
        cached.solution_id));

    auto output = output_2d.squeeze(2);

    if (bias.has_value() && bias->defined()) {
        auto bias_c = ensure_data_ptr(*bias);
        output = output + bias_c.unsqueeze(0).unsqueeze(2);
    }

    return output;
}
