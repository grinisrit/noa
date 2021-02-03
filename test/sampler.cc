#include "sampler.hh"

TEST(Sampler, AutogradFunnel)
{
    auto theta = _theta.clone().requires_grad_();
    auto val = alog_funnel(theta);
    ASSERT_NEAR((val - expected_val).item<double>(), 0., 1e-3);

    auto grad = torch::autograd::grad({val}, {theta})[0];
    ASSERT_NEAR((grad - expected_grad).abs().sum().item<double>(), 0., 1e-3);
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
    auto theta = _theta.clone().requires_grad_();
    auto log_prob = alog_funnel(theta);
    auto metric = ghmc::softabs_map(log_prob, theta, 0.);
    ASSERT_TRUE(metric.has_value());
    auto [spec, Q] = metric.value();
    auto err = (spec - expected_spectrum).abs().sum().item<double>();
    auto orth = (Q.mm(Q.t()) - torch::eye(theta.numel())).abs().sum().item<double>();
    ASSERT_NEAR(err, 0., 1e-3);
    ASSERT_NEAR(orth, 0., 1e-3);
}

TEST(Sampler, Hamiltonian)
{
    auto theta = _theta.clone().requires_grad_();
    auto momentum = _momentum.clone().requires_grad_();
    auto ham_ = ghmc::hamiltonian(alog_funnel, theta, momentum, 0.);
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
