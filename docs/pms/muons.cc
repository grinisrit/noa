#include <noa/utils/common.hh>
#include <noa/pms/constants.hh>
#include <noa/pms/dcs.hh>

#include <torch/extension.h>

using namespace noa::pms;
using namespace noa::utils;

inline Tensor bremsstrahlung(Tensor kinetic_energies, Tensor recoil_energies) {
    const auto result = torch::zeros_like(kinetic_energies);
    dcs::vmap<Scalar>(dcs::pumas::bremsstrahlung)(
            result, kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
    return result;
}

inline Tensor pair_production(Tensor kinetic_energies, Tensor recoil_energies) {
    const auto result = torch::zeros_like(kinetic_energies);
    dcs::vmap<Scalar>(dcs::pumas::pair_production)(
            result, kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
    return result;
}

inline Tensor photonuclear(Tensor kinetic_energies, Tensor recoil_energies) {
    const auto result = torch::zeros_like(kinetic_energies);
    dcs::vmap<Scalar>(dcs::pumas::photonuclear)(
            result, kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
    return result;
}

inline Tensor ionisation(Tensor kinetic_energies, Tensor recoil_energies) {
    const auto result = torch::zeros_like(kinetic_energies);
    dcs::vmap<Scalar>(dcs::pumas::ionisation)(
            result, kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
    return result;
}

inline void serialise(Tensor tensor, std::string path) {
    torch::save(tensor, path);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bremsstrahlung", &bremsstrahlung, py::call_guard<py::gil_scoped_release>(),
          "Standard Rock Bremsstrahlung DCS for Muons");
    m.def("pair_production", &pair_production, py::call_guard<py::gil_scoped_release>(),
          "Standard Rock Pair production DCS for Muons");
    m.def("photonuclear", &photonuclear, py::call_guard<py::gil_scoped_release>(),
          "Standard Rock Photonuclear DCS for Muons");
    m.def("ionisation", &ionisation, py::call_guard<py::gil_scoped_release>(),
          "Standard Rock Ionisation DCS for Muons");
    m.def("serialise", &serialise, py::call_guard<py::gil_scoped_release>(),
          "Save tensor to disk");
}