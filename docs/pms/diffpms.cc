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

inline void serialise(torch::Tensor tensor, std::string path) {
    torch::save(tensor, path);
}

inline const auto PI = 2.f * torch::acos(torch::tensor(0.f));

inline torch::Tensor rot(const torch::Tensor &theta){
    const auto n = theta.numel();
    const auto c = torch::cos(theta);
    const auto s = torch::sin(theta);
    return torch::stack({c,-s, s, c}).t().view({n,2,2});
}

inline torch::Tensor mix_density(const torch::Tensor &states, const torch::Tensor &node){
    return torch::exp(-(states - node.slice(0,0,2)).pow(2).sum(-1) / node[2].abs());
}


inline std::tuple<torch::Tensor, torch::Tensor> backward_mc_grad(
        torch::Tensor theta,
        torch::Tensor node) {
    const auto npar = 1;
    const auto detector = torch::zeros(2);
    auto bmc_grad = torch::zeros_like(node);

    const auto material_A = 0.01f;
    const auto material_B = 0.9f;

    const auto length1 =  1.f - 0.2f * torch::rand(npar);
    const auto rot1 = rot(theta);
    auto step1 = torch::stack({torch::zeros(npar),  length1}).t();
    step1 = rot1.matmul(step1.view({npar, 2,1})).view({npar, 2});
    const auto state1 = detector + step1;

    auto biasing = torch::randint(0,2, {npar});

    auto nodeg = node.detach().requires_grad_();
    auto densityg = mix_density(state1, nodeg);
    auto weightsg =
            torch::where(biasing > 0,
                         (densityg/0.5) * material_A,
                         ((1 - densityg)/0.5) * material_B) * torch::exp(-0.01f*length1);
    bmc_grad += torch::autograd::grad({weightsg}, {nodeg})[0];
    auto weights = weightsg.detach();

    const auto length2 =  1.f - 0.2f * torch::rand(npar);
    const auto rot2 = rot(0.05f * PI * (torch::rand(npar) - 0.5f));
    auto step2 = length2.view({npar,1}) * step1 / length1.view({npar,1});
    step2 = rot2.matmul(step2.view({npar,2,1})).view({npar, 2});
    const auto state2 = state1 + step2;

    biasing = torch::randint(0,2, {npar});

    nodeg = node.detach().requires_grad_();
    densityg = mix_density(state2, nodeg);
    weightsg =
            torch::where(biasing > 0,
                         (densityg/0.5) * material_A,
                         ((1 - densityg)/0.5) * material_B) * torch::exp(-0.01f*length2);
    const auto weight2 = weightsg.detach();
    bmc_grad = weights * torch::autograd::grad({weightsg}, {nodeg})[0] + weight2 * bmc_grad;
    weights *= weight2;

    return std::make_tuple(weights, bmc_grad);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bremsstrahlung", &bremsstrahlung, py::call_guard<py::gil_scoped_release>(),
          "Standard Rock Bremsstrahlung DCS for Muons");
    m.def("serialise", &serialise, py::call_guard<py::gil_scoped_release>(),
          "Save tensor to disk");
    m.def("backward_mc_grad", &backward_mc_grad, py::call_guard<py::gil_scoped_release>(),
          "Backward MC with adjoint sensitivity");
}