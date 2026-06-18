#include <optional>
#include <torch/extension.h>

void binary_thinning_cuda(torch::Tensor image, int mode);
void extract_centerline_cuda(torch::Tensor mask, torch::Tensor probs, int mode);

void region_grow_backward_cuda(torch::Tensor mask, torch::Tensor cl,
                               torch::Tensor grad_S, torch::Tensor grad_prob);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("binary_thinning", &binary_thinning_cuda, "3D Binary Thinning (CUDA)",
        pybind11::arg("image"), pybind11::arg("mode") = 0);
  m.def("extract_centerline", &extract_centerline_cuda,
        "3D Centerline Extraction (CUDA)", pybind11::arg("mask"),
        pybind11::arg("probs"), pybind11::arg("mode") = 0);
  m.def("region_grow_backward", &region_grow_backward_cuda,
        "Region grow backward pass for continuous skeletonization",
        pybind11::arg("mask"), pybind11::arg("cl"), pybind11::arg("grad_S"),
        pybind11::arg("grad_prob"));
}
