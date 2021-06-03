#pragma once

#include "test-data.hh"

#include <noa/ghmc.hh>
#include <noa/utils/common.hh>

#include <gtest/gtest.h>

using namespace noa;
using namespace noa::ghmc;
using namespace noa::utils;


inline TensorsOpt get_funnel_hessian(const torch::Tensor &theta_, torch::DeviceType device) {
    torch::manual_seed(utils::SEED);
    const auto log_prob_graph = log_funnel({theta_.to(device, false, true)});
    return numerics::hessian(log_prob_graph);
}

inline void test_funnel_hessian(torch::DeviceType device = torch::kCPU) {
    const auto hess_ = get_funnel_hessian(GHMCData::get_theta(), device);

    ASSERT_TRUE(hess_.has_value());
    const auto &hess = hess_.value().at(0);
    ASSERT_TRUE(hess.device().type() == device);

    const auto res = hess.detach().to(torch::kCPU);

    const auto err = (res + GHMCData::get_neg_hessian_funnel()).abs().sum().item<float>();
    ASSERT_NEAR(err, 0.f, 1e-3f);
}

inline MetricDecompositionOpt get_softabs_metric(const torch::Tensor &theta_, torch::DeviceType device) {
    torch::manual_seed(utils::SEED);
    const auto log_prob_graph = log_funnel(Parameters{theta_.to(device, false, true)});
    return softabs_metric(conf_funnel)(log_prob_graph);
}

inline void test_softabs_metric(torch::DeviceType device = torch::kCPU) {

    const auto metric_ = get_softabs_metric(GHMCData::get_theta(), device);
    ASSERT_TRUE(metric_.has_value());

    const auto &[spec_, Q_] = metric_.value();
    ASSERT_TRUE(spec_.at(0).device().type() == device);
    ASSERT_TRUE(Q_.at(0).device().type() == device);
    const auto spec = spec_.at(0).detach().to(torch::kCPU, false, true);
    const auto Q = Q_.at(0).detach().to(torch::kCPU, false, true);

    const auto err = (spec - GHMCData::get_expected_spectrum()).abs().sum().item<float>();
    const auto orthogonality = (Q.mm(Q.t()) - torch::eye(GHMCData::get_theta().numel())).abs().sum().item<float>();

    ASSERT_NEAR(err, 0.f, 1e-3f);
    ASSERT_NEAR(orthogonality, 0.f, 1e-5f);
}

inline PhaseSpaceFoliationOpt get_hamiltonian(
        const torch::Tensor &theta_,
        const torch::Tensor &momentum_,
        torch::DeviceType device) {
    torch::manual_seed(utils::SEED);
    return hamiltonian(log_funnel, conf_funnel)(Parameters{theta_.to(device, false, true)},
                                         Momentum{momentum_.to(device, false, true)});
}

inline void test_hamiltonian(torch::DeviceType device = torch::kCPU) {
    const auto ham_ = get_hamiltonian(GHMCData::get_theta(), GHMCData::get_momentum(), device);
    ASSERT_TRUE(ham_.has_value());
    const auto &energy_ = std::get<Energy>(ham_.value());
    ASSERT_TRUE(energy_.device().type() == device);
    const auto energy = energy_.detach().to(torch::kCPU, false, true);
    const auto err = (energy - GHMCData::get_expected_energy()).abs().sum().item<float>();
    ASSERT_NEAR(err, 0., 1e-3);
}

inline HamiltonianFlow get_hamiltonian_flow(
        const torch::Tensor &theta_,
        const torch::Tensor &momentum_,
        torch::DeviceType device) {
    torch::manual_seed(utils::SEED);
    return hamiltonian_flow(log_funnel, conf_funnel)(Parameters{theta_.to(device, false, true)},
                                              Momentum{momentum_.to(device, false, true)});
}

inline void test_hamiltonian_flow(torch::DeviceType device = torch::kCPU) {

    const auto[theta_flow, momentum_flow, energy] =
    get_hamiltonian_flow(GHMCData::get_theta(), GHMCData::get_momentum(), device);

    ASSERT_TRUE(theta_flow.size() == 2);
    ASSERT_TRUE(momentum_flow.size() == 2);
    ASSERT_TRUE(energy.size() == 2);

    ASSERT_TRUE(theta_flow.at(1).at(0).device().type() == device);
    ASSERT_TRUE(momentum_flow.at(1).at(0).device().type() == device);
    ASSERT_TRUE(energy.at(1).device().type() == device);

    const auto theta_proposal = theta_flow.at(1).at(0).to(torch::kCPU, false, true);
    auto err = (theta_proposal - GHMCData::get_expected_flow_theta()).abs().sum().item<float>();
    ASSERT_NEAR(err, 0., 1e-3);

    const auto momentum_proposal = momentum_flow.at(1).at(0).to(torch::kCPU, false, true);
    err = (momentum_proposal - GHMCData::get_expected_flow_moment()).abs().sum().item<float>();
    ASSERT_NEAR(err, 0., 1e-2);
}
