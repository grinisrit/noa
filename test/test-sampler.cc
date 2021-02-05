#include "sampler.hh"

#include <gtest/gtest.h>

inline void test_fisher_info(torch::DeviceType device = torch::kCPU)
{
    auto fisher = get_fisher_info(device);
    
    ASSERT_TRUE(fisher.has_value());
    ASSERT_TRUE(fisher.value().device().type() == device);
    auto res = fisher.value().to(torch::kCPU);
    auto err = (res - expected_fisher).abs().sum().item<float>();
    ASSERT_NEAR(err, 0., 1e-3);
}

inline void test_symplectic_flow(torch::DeviceType device = torch::kCPU)
{
    auto flow = get_symplectic_flow(device);

    ASSERT_TRUE(flow.has_value());
    auto [p_flow_, m_flow_] = flow.value();
    ASSERT_TRUE(p_flow_.device().type() == device);
    auto p_flow = p_flow_.to(torch::kCPU);
    ASSERT_TRUE(m_flow_.device().type() == device);
    auto m_flow = m_flow_.to(torch::kCPU);
    auto err = (p_flow[-1] - expected_flow_theta).abs().sum().item<float>();
    ASSERT_NEAR(err, 0., 1e-2);
    err = (m_flow[-1] - expected_flow_moment).abs().sum().item<float>();
    ASSERT_NEAR(err, 0., 1e-2);
}

TEST(Sampler, FisherInfo)
{
    test_fisher_info();
}

TEST(Sampler, FisherInfoCUDA)
{
    ASSERT_TRUE(torch::cuda::is_available());
    test_fisher_info(torch::kCUDA);
}

TEST(Sampler, SoftAbsMap)
{
    auto metric = get_metric();
    ASSERT_TRUE(metric.has_value());
    auto [spec, Q] = metric.value();
    auto err = (spec - expected_spectrum).abs().sum().item<float>();
    auto orth = (Q.mm(Q.t()) - torch::eye(_theta.numel())).abs().sum().item<float>();
    ASSERT_NEAR(err, 0., 1e-3);
    ASSERT_NEAR(orth, 0., 1e-3);
}

TEST(Sampler, Hamiltonian)
{
    auto ham_ = get_hamiltonian();
    ASSERT_TRUE(ham_.has_value());
    torch::Tensor energy = std::get<0>(ham_.value());
    auto err = (energy - expected_energy).abs().sum().item<double>();
    ASSERT_NEAR(err, 0., 1e-3);
}

TEST(Sampler, SymplecticFlow)
{
    test_symplectic_flow();
}

TEST(Sampler, SymplecticFlowCUDA)
{
    ASSERT_TRUE(torch::cuda::is_available());
    test_symplectic_flow(torch::kCUDA);
}
