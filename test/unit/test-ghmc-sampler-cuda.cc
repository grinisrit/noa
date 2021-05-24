#include "test-ghmc-sampler.hh"


TEST(GHMC, FunnelHessianCUDA)
{
    ASSERT_TRUE(torch::cuda::is_available());
    test_funnel_hessian(torch::kCUDA);
}

TEST(GHMC, FunnelHessianCUDA)
{
    ASSERT_TRUE(torch::cuda::is_available());
    test_softabs_metric(torch::kCUDA);
}

TEST(GHMC, FisherInfoCUDA)
{
    ASSERT_TRUE(torch::cuda::is_available());
    test_fisher_info(torch::kCUDA);
}

TEST(GHMC, SymplecticFlowCUDA)
{
    ASSERT_TRUE(torch::cuda::is_available());
    test_symplectic_flow(torch::kCUDA);
}