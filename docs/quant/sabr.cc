#include <torch/extension.h>

inline void check_tensor(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_contiguous(), "contiguous tensor is required");
    TORCH_CHECK(tensor.dtype() == torch::kFloat64, "double tensor is required");
}

inline torch::Tensor test(torch::Tensor tensor) {
    check_tensor(tensor);
    return tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("test", &test, py::call_guard<py::gil_scoped_release>(),
          "test utilites");
}