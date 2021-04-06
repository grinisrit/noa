#include "unit-ghmc.hh"
#include "test-data.hh"

#include <noa/utils/common.hh>
#include <gtest/gtest.h>


inline void test_fisher_info(torch::DeviceType device = torch::kCPU)
{
    auto fisher = get_fisher_info(GHMCData::get_theta(), device);

    ASSERT_TRUE(fisher.has_value());
    ASSERT_TRUE(fisher.value().device().type() == device);
    auto res = fisher.value().to(torch::kCPU);
    auto err = (res - GHMCData::get_expected_fisher()).abs().sum().item<float>();
    ASSERT_NEAR(err, 0., 1e-3);
}

inline void test_symplectic_flow(torch::DeviceType device = torch::kCPU)
{
    auto flow = get_symplectic_flow(GHMCData::get_theta(), GHMCData::get_momentum(), device);

    ASSERT_TRUE(flow.has_value());
    auto [p_flow_, m_flow_] = flow.value();
    ASSERT_TRUE(p_flow_.device().type() == device);
    auto p_flow = p_flow_.to(torch::kCPU);
    ASSERT_TRUE(m_flow_.device().type() == device);
    auto m_flow = m_flow_.to(torch::kCPU);
    auto err = (p_flow[-1] - GHMCData::get_expected_flow_theta()).abs().sum().item<float>();
    ASSERT_NEAR(err, 0., 1e-2);
    err = (m_flow[-1] - GHMCData::get_expected_flow_moment()).abs().sum().item<float>();
    ASSERT_NEAR(err, 0., 1e-2);
}

TEST(GHMC, FisherInfo)
{
    test_fisher_info();
}

TEST(GHMC, FisherInfoCUDA)
{
    ASSERT_TRUE(torch::cuda::is_available());
    test_fisher_info(torch::kCUDA);
}

TEST(GHMC, SoftAbsMap)
{
    auto metric = get_metric(GHMCData::get_theta());
    ASSERT_TRUE(metric.has_value());
    auto [spec, Q] = metric.value();
    auto err = (spec - GHMCData::get_expected_spectrum()).abs().sum().item<float>();
    auto orth = (Q.mm(Q.t()) - torch::eye(GHMCData::get_theta().numel())).abs().sum().item<float>();
    ASSERT_NEAR(err, 0., 1e-3);
    ASSERT_NEAR(orth, 0., 1e-3);
}

TEST(GHMC, Hamiltonian)
{
    auto ham_ = get_hamiltonian(GHMCData::get_theta(), GHMCData::get_momentum());
    ASSERT_TRUE(ham_.has_value());
    torch::Tensor energy = std::get<0>(ham_.value());
    auto err = (energy - GHMCData::get_expected_energy()).abs().sum().item<float>();
    ASSERT_NEAR(err, 0., 1e-3);
}

TEST(GHMC, SymplecticFlow)
{
    test_symplectic_flow();
}

TEST(GHMC, SymplecticFlowCUDA)
{
    ASSERT_TRUE(torch::cuda::is_available());
    test_symplectic_flow(torch::kCUDA);
}
