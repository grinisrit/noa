#include <noa/pms/dcs.hh>
#include <noa/pms/constants.hh>
#include <noa/utils/common.hh>

#include <torch/extension.h>

using namespace noa::pms;
using namespace noa::utils;

inline torch::Tensor bremsstrahlung(torch::Tensor kinetic_energies, torch::Tensor recoil_energies) {
    const auto result = torch::zeros_like(kinetic_energies);
    dcs::vmap<Scalar>(dcs::pumas::bremsstrahlung)(
            result, kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
    return result;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bremsstrahlung", &bremsstrahlung, py::call_guard<py::gil_scoped_release>(),
          "Standard Rock Bremsstrahlung DCS for Muons");
}