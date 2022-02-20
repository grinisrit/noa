#include <noa/pms/dcs.hh>

#include <torch/extension.h>

using namespace noa::pms;
using namespace noa::utils;

inline Tensor bremsstrahlung(Tensor kinetic_energies, Tensor recoil_energies) {
    return dcs::map(dcs::bremsstrahlung)(
        kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
  
}

inline Tensor pair_production(Tensor kinetic_energies, Tensor recoil_energies) {
    return dcs::map(dcs::pair_production)(
        kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
}

inline Tensor photonuclear(Tensor kinetic_energies, Tensor recoil_energies) {
    return dcs::map(dcs::photonuclear)(
        kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
}

inline Tensor ionisation(Tensor kinetic_energies, Tensor recoil_energies) {
    return dcs::map(dcs::ionisation)(
        kinetic_energies, recoil_energies, STANDARD_ROCK, MUON_MASS);
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