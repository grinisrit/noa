#pragma once

#include <noa/ghmc.hh>
#include <noa/utils/common.hh>

using namespace noa;
using namespace noa::utils;

inline const auto alog_funnel = [](const auto &ntheta) {
    auto dim = ntheta.numel() - 1;
    return -0.5 * ((torch::exp(ntheta[0]) * ntheta.slice(0, 1, dim + 1).pow(2).sum()) +
                   (ntheta[0].pow(2) / 9) - dim * ntheta[0]);
};

inline ghmc::FisherInfo get_fisher_info(const torch::Tensor &theta, torch::DeviceType device)
{
    torch::manual_seed(utils::SEED);
    auto ntheta = theta.clone().to(device).requires_grad_();
    auto log_prob = alog_funnel(ntheta);
    return ghmc::fisher_info(log_prob, ntheta);
}

inline ghmc::SymplecticFlow get_symplectic_flow(
    const torch::Tensor &theta, 
    const torch::Tensor &momentum,
    torch::DeviceType device)
{
    torch::manual_seed(utils::SEED);
    auto ntheta = theta.clone().to(device);
    auto nmomentum = momentum.clone().to(device);

    return ghmc::symplectic_flow(/*log_probability_density=*/alog_funnel,
                                 /*params=*/ntheta, /*nmomentum=*/nmomentum,
                                 /*leap_steps=*/1, /*epsilon=*/0.14, /*binding_const=*/10.,
                                 /*jitter=*/0.00001, /*jitter_max=*/0);
}

inline ghmc::SoftAbsMap get_metric(const torch::Tensor &theta)
{
    torch::manual_seed(utils::SEED);
    auto ntheta = theta.clone().requires_grad_();
    auto log_prob = alog_funnel(ntheta);
    return ghmc::softabs_map(log_prob, ntheta, 0.);
}

inline ghmc::Hamiltonian get_hamiltonian(
    const torch::Tensor &theta,
    const torch::Tensor &momentum)
{
    torch::manual_seed(utils::SEED);
    auto ntheta = theta.clone().requires_grad_();
    auto nmomentum = momentum.clone().requires_grad_();
    return ghmc::hamiltonian(alog_funnel, ntheta, nmomentum, 0.);
}
