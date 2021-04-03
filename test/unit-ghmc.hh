#pragma once

#include <noa/ghmc.hh>
#include <noa/utils/common.hh>

using namespace noa;

inline const auto alog_funnel = [](const auto &theta) {
    auto dim = theta.numel() - 1;
    return -0.5 * ((torch::exp(theta[0]) * theta.slice(0, 1, dim + 1).pow(2).sum()) +
                   (theta[0].pow(2) / 9) - dim * theta[0]);
};

inline const auto _theta = torch::tensor({0.8745f, -0.4383f, 0.5938f, 0.1020f});
inline const auto _momentum = torch::tensor({-0.4262f, -0.5880f, 0.0718f, 1.0845f});

inline const auto expected_val = 0.6038f;
inline const auto expected_grad = torch::tensor({0.7373f, 1.0509f, -1.4237f, -0.2446f});

inline const auto expected_fisher = torch::tensor(
    {{0.7766f, -1.0509f, 1.4237f, 0.2446f},
     {-1.0509f, 2.3977f, 0.0000f, 0.0000f},
     {1.4237f, 0.0000f, 2.3977f, 0.0000f},
     {0.2446f, 0.0000f, 0.0000f, 2.3977f}});

inline const auto expected_spectrum = torch::tensor({0.3745f, 2.3977f, 2.3977f, 3.5488f});

inline const auto expected_energy = torch::tensor({1.2519f});

inline const auto expected_flow_theta = torch::tensor({0.6839f, -0.5335f, 0.6796f, 0.1806f});
inline const auto expected_flow_moment = torch::tensor({-0.5150f, -0.3777f, -0.1253f, 0.9208f});

inline ghmc::FisherInfo get_fisher_info(torch::DeviceType device)
{
    torch::manual_seed(utils::SEED);
    auto theta = _theta.clone().to(device).requires_grad_();
    auto log_prob = alog_funnel(theta);
    return ghmc::fisher_info(log_prob, theta);
}

inline ghmc::SymplecticFlow get_symplectic_flow(torch::DeviceType device)
{
    torch::manual_seed(utils::SEED);
    auto theta = _theta.clone().to(device);
    auto momentum = _momentum.clone().to(device);

    return ghmc::symplectic_flow(/*log_probability_density=*/alog_funnel,
                                 /*params=*/theta, /*momentum=*/momentum,
                                 /*leap_steps=*/1, /*epsilon=*/0.14, /*binding_const=*/10.,
                                 /*jitter=*/0.00001, /*jitter_max=*/0);
}

inline ghmc::SoftAbsMap get_metric()
{
    torch::manual_seed(utils::SEED);
    auto theta = _theta.clone().requires_grad_();
    auto log_prob = alog_funnel(theta);
    return ghmc::softabs_map(log_prob, theta, 0.);
}

inline ghmc::Hamiltonian get_hamiltonian()
{
    torch::manual_seed(utils::SEED);
    auto theta = _theta.clone().requires_grad_();
    auto momentum = _momentum.clone().requires_grad_();
    return ghmc::hamiltonian(alog_funnel, theta, momentum, 0.);
}
