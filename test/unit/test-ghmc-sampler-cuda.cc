#include "test-ghmc-sampler.hh"


TEST(GHMC, FunnelHessianCUDA)
{
    ASSERT_TRUE(torch::cuda::is_available());
    test_funnel_hessian(torch::kCUDA);
}

TEST(GHMC, SoftAbsMetricCUDA)
{
    ASSERT_TRUE(torch::cuda::is_available());
    test_softabs_metric(torch::kCUDA);
}

TEST(GHMC, HamiltonianCUDA)
{
    ASSERT_TRUE(torch::cuda::is_available());
    test_hamiltonian(torch::kCUDA);
}

TEST(GHMC, HamiltonianFlowCUDA)
{
    ASSERT_TRUE(torch::cuda::is_available());
    test_hamiltonian_flow(torch::kCUDA);
}
