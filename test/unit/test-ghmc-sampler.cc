#include "test-ghmc-sampler.hh"


TEST(GHMC, FisherInfo)
{
    test_fisher_info();
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
