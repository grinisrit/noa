#include "test-ghmc-sampler.hh"


TEST(GHMC, FunnelHessian)
{
    test_funnel_hessian();
}

TEST(GHMC, SoftAbsMetric)
{
    test_softabs_metric();
}

TEST(GHMC, Hamiltonian)
{
    test_hamiltonian();
}

TEST(GHMC, HamiltonianFlow)
{
    test_hamiltonian_flow();
}
