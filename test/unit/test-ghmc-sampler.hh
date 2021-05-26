#pragma once

#include "test-data.hh"

#include <noa/ghmc.hh>
#include <noa/utils/common.hh>

#include <gtest/gtest.h>

using namespace noa;
using namespace noa::ghmc;
using namespace noa::utils;

inline const auto log_funnel = [](const auto &theta_) {
    auto dim = theta_.numel() - 1;
    return -0.5 * ((torch::exp(theta_[0]) * theta_.slice(0, 1, dim + 1).pow(2).sum()) +
                   (theta_[0].pow(2) / 9) - dim * theta_[0]);
};

inline const auto conf = Configuration<float>{}
        .set_max_flow_steps(1)
        .set_cutoff(1e-6f)
        .set_verbosity(true);

inline TensorOpt get_funnel_hessian(const torch::Tensor &theta_, torch::DeviceType device) {
    torch::manual_seed(utils::SEED);
    const auto theta = theta_.detach().to(device, false, true).requires_grad_();
    const auto log_prob = log_funnel(theta);
    return numerics::hessian(log_prob, theta);
}

inline void test_funnel_hessian(torch::DeviceType device = torch::kCPU) {
    const auto hess = get_funnel_hessian(GHMCData::get_theta(), device);

    ASSERT_TRUE(hess.has_value());
    ASSERT_TRUE(hess.value().device().type() == device);
    const auto res = hess.value().detach().to(torch::kCPU);
    const auto err = (res + GHMCData::get_neg_hessian_funnel()).abs().sum().item<float>();
    ASSERT_NEAR(err, 0.f, 1e-3f);
}

inline MetricDecompositionOpt get_softabs_metric(const torch::Tensor &theta_, torch::DeviceType device) {
    torch::manual_seed(utils::SEED);
    const auto theta = theta_.detach().to(device, false, true).requires_grad_();
    const auto log_prob = log_funnel(theta);
    return softabs_metric(conf)(log_prob, theta);
}

inline void test_softabs_metric(torch::DeviceType device = torch::kCPU) {

    const auto metric_ = get_softabs_metric(GHMCData::get_theta(), device);
    ASSERT_TRUE(metric_.has_value());

    const auto &[spec_, Q_] = metric_.value();
    ASSERT_TRUE(spec_.device().type() == device);
    ASSERT_TRUE(Q_.device().type() == device);
    const auto spec = spec_.detach().to(torch::kCPU);
    const auto Q = Q_.detach().to(torch::kCPU);

    const auto err = (spec - GHMCData::get_expected_spectrum()).abs().sum().item<float>();
    const auto orthogonality = (Q.mm(Q.t()) - torch::eye(GHMCData::get_theta().numel())).abs().sum().item<float>();

    ASSERT_NEAR(err, 0.f, 1e-3f);
    ASSERT_NEAR(orthogonality, 0.f, 1e-5f);
}

inline std::optional<Hamiltonian> get_hamiltonian(
        const torch::Tensor &theta_,
        const torch::Tensor &momentum_,
        torch::DeviceType device) {
    torch::manual_seed(utils::SEED);
    const auto theta = theta_.detach().to(device, false, true).requires_grad_();
    const auto log_prob = log_funnel(theta);
    const auto metric = softabs_metric(conf)(log_prob, theta);
    if (!metric.has_value()) return std::nullopt;

    const auto momentum = momentum_.detach().to(device, false, true);
    return geometric_hamiltonian(log_prob, theta, metric.value(), momentum);

}

inline void test_hamiltonian(torch::DeviceType device = torch::kCPU) {
    const auto ham_ = get_hamiltonian(GHMCData::get_theta(), GHMCData::get_momentum(), device);
    ASSERT_TRUE(ham_.has_value());
    const auto energy_ = std::get<0>(ham_.value());
    ASSERT_TRUE(energy_.device().type() == device);
    const auto energy = energy_.detach().to(torch::kCPU);
    const auto err = (energy - GHMCData::get_expected_energy()).abs().sum().item<float>();
    ASSERT_NEAR(err, 0., 1e-3);
}

inline void get_symplectic_map(
        const torch::Tensor &theta_,
        const torch::Tensor &momentum_,
        torch::DeviceType device) {
    torch::manual_seed(utils::SEED);
    const auto theta = theta_.detach().to(device, false, true).requires_grad_();
    const auto momentum = momentum_.detach().to(device, false, true);
}


/////////////////////////////////////////////////////////////////
inline FisherInfo get_fisher_info(const torch::Tensor &theta_, torch::DeviceType device) {
    torch::manual_seed(utils::SEED);
    auto theta = theta_.clone().to(device).requires_grad_();
    auto log_prob = log_funnel(theta);
    return fisher_info(log_prob, theta);
}

inline SymplecticFlow get_symplectic_flow_ref(
        const torch::Tensor &theta,
        const torch::Tensor &momentum,
        torch::DeviceType device) {
    torch::manual_seed(utils::SEED);
    auto ntheta = theta.clone().to(device);
    auto nmomentum = momentum.clone().to(device);

    return symplectic_flow(/*log_probability_density=*/log_funnel,
            /*params=*/ntheta, /*nmomentum=*/nmomentum,
            /*leap_steps=*/1, /*epsilon=*/0.14, /*binding_const=*/10.,
            /*jitter=*/0.00001, /*jitter_max=*/0);
}

inline SoftAbsMap get_metric(const torch::Tensor &theta) {
    torch::manual_seed(utils::SEED);
    auto ntheta = theta.clone().requires_grad_();
    auto log_prob = log_funnel(ntheta);
    return softabs_map(log_prob, ntheta, 0.);
}

inline HamiltonianRef get_hamiltonian_ref(
        const torch::Tensor &theta,
        const torch::Tensor &momentum) {
    torch::manual_seed(utils::SEED);
    auto ntheta = theta.clone().requires_grad_();
    auto nmomentum = momentum.clone().requires_grad_();
    return hamiltonian(log_funnel, ntheta, nmomentum, 0.);
}

inline void test_fisher_info(torch::DeviceType device = torch::kCPU) {
    auto fisher = get_fisher_info(GHMCData::get_theta(), device);

    ASSERT_TRUE(fisher.has_value());
    ASSERT_TRUE(fisher.value().device().type() == device);
    auto res = fisher.value().to(torch::kCPU);
    auto err = (res - GHMCData::get_neg_hessian_funnel()).abs().sum().item<float>();
    ASSERT_NEAR(err, 0., 1e-3);
}

inline void test_symplectic_flow_ref(torch::DeviceType device = torch::kCPU) {
    auto flow = get_symplectic_flow_ref(GHMCData::get_theta(), GHMCData::get_momentum(), device);

    ASSERT_TRUE(flow.has_value());
    auto[p_flow_, m_flow_] = flow.value();
    ASSERT_TRUE(p_flow_.device().type() == device);
    auto p_flow = p_flow_.to(torch::kCPU);
    ASSERT_TRUE(m_flow_.device().type() == device);
    auto m_flow = m_flow_.to(torch::kCPU);
    auto err = (p_flow[-1] - GHMCData::get_expected_flow_theta()).abs().sum().item<float>();
    ASSERT_NEAR(err, 0., 1e-2);
    err = (m_flow[-1] - GHMCData::get_expected_flow_moment()).abs().sum().item<float>();
    ASSERT_NEAR(err, 0., 1e-2);
}
