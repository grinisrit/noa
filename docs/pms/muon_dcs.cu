#include <noa/pms/leptons/kernels.cuh>

#include <torch/extension.h>

using namespace noa::pms::leptons;
using namespace noa::utils;

inline torch::Tensor bremsstrahlung(torch::Tensor kinetic_energies, torch::Tensor recoil_energies) {
    const auto result = torch::zeros_like(kinetic_energies);
    dcs::cuda::vmap_bremsstrahlung(result, kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
    return result;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bremsstrahlung", &bremsstrahlung, py::call_guard<py::gil_scoped_release>(),
        "Standard Rock Bremsstrahlung DCS for Muons on CUDA");
}