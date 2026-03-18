#include <torch/extension.h>

void binary_thinning_cuda(torch::Tensor image);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("binary_thinning", &binary_thinning_cuda, "3D Binary Thinning (CUDA)");
}
