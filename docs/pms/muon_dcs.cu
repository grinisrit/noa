#include <noa/kernels.cuh>

#include <torch/extension.h>

using namespace noa::pms;
using namespace noa::utils;

inline torch::Tensor bremsstrahlung(torch::Tensor kinetic_energies, torch::Tensor recoil_energies) {
    return dcs::cuda::map_bremsstrahlung(kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bremsstrahlung", &bremsstrahlung, py::call_guard<py::gil_scoped_release>(),
        "Standard Rock Bremsstrahlung DCS for Muons on CUDA");
}