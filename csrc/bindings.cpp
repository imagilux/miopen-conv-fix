#include <torch/extension.h>

// Declarations from miopen_conv.hip
torch::Tensor miopen_conv1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups);

torch::Tensor miopen_conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups,
    std::vector<int64_t> dilation);

PYBIND11_MODULE(_C, m) {
    m.def("conv1d_forward", &miopen_conv1d_forward,
          "MIOpen Conv1d forward with proper workspace allocation",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"),
          py::arg("groups"));
    m.def("conv_transpose1d_forward", &miopen_conv_transpose1d_forward,
          "MIOpen ConvTranspose1d forward with proper workspace allocation",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups"), py::arg("dilation"));
}
