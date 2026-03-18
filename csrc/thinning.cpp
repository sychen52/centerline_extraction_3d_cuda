#include <torch/extension.h>

void binary_thinning_cuda(torch::Tensor image, bool deterministic);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("binary_thinning", &binary_thinning_cuda, "3D Binary Thinning (CUDA)",
          pybind11::arg("image"), pybind11::arg("deterministic") = false);
}
