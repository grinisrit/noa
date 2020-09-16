#include <gtest/gtest.h>
#include <torch/torch.h>

#include <ghmc/ghmc.hh>

auto alog_funnel = [](const auto &theta) {
    auto dim = theta.numel() - 1;
    return -0.5 * ((torch::exp(theta[0]) * theta.slice(0, 1, dim + 1).pow(2).sum()) +
                   (theta[0].pow(2) / 9) - dim * theta[0]);
};

auto _theta = torch::tensor({0.8745, -0.4383, 0.5938, 0.1020});
auto _momentum = torch::tensor({-0.4262, -0.5880, 0.0718, 1.0845});

auto expected_val = 0.6038;
auto expected_grad = torch::tensor({0.7373, 1.0509, -1.4237, -0.2446});

auto expected_fisher = torch::tensor(
    {{0.7766, -1.0509, 1.4237, 0.2446},
     {-1.0509, 2.3977, 0.0000, 0.0000},
     {1.4237, 0.0000, 2.3977, 0.0000},
     {0.2446, 0.0000, 0.0000, 2.3977}});

auto expected_spectrum = torch::tensor({0.3745, 2.3977, 2.3977, 3.5488});

auto expected_energy = torch::tensor({1.2519});

auto expected_flow_theta = torch::tensor({0.6839, -0.5335, 0.6796, 0.1806});
auto expected_flow_moment = torch::tensor({-0.5150, -0.3777, -0.1253, 0.9208});

TEST(Unit, AutogradFunnel)
{
    auto theta = _theta.clone().requires_grad_();
    auto val = alog_funnel(theta);
    ASSERT_NEAR((val - expected_val).item<double>(), 0., 1e-3);

    auto grad = torch::autograd::grad({val}, {theta})[0];
    ASSERT_NEAR((grad - expected_grad).abs().sum().item<double>(), 0., 1e-3);
}

void test_fisher_info(torch::DeviceType device = torch::kCPU)
{
    auto theta = _theta.clone().to(device).requires_grad_();
    auto log_prob = alog_funnel(theta);
    auto fisher = ghmc::fisher_info(log_prob, theta);
    ASSERT_TRUE(fisher.has_value());
    ASSERT_TRUE(fisher.value().device().type() == device);
    auto res = fisher.value().to(torch::kCPU);
    auto err = (res - expected_fisher).abs().sum().item<double>();
    ASSERT_NEAR(err, 0., 1e-3);
}

TEST(Unit, FisherInfo)
{
    test_fisher_info();
}

TEST(Unit, FisherInfoCUDA)
{
    ASSERT_TRUE(torch::cuda::is_available());
    test_fisher_info(torch::kCUDA);
}

void benchmark_fisher_info(torch::Tensor log_prob, torch::Tensor theta)
{
    auto start = std::chrono::steady_clock::now();
    auto fisher = ghmc::fisher_info(log_prob, theta);
    auto end = std::chrono::steady_clock::now();
    printf("Execution took %.3f sec\n",
           std::chrono::duration<double>(end - start).count());
    ASSERT_TRUE(fisher.has_value());
}

TEST(Unit, FisherInfo_Benchmark)
{
    auto theta = torch::randn(1000).requires_grad_();
    auto log_prob = alog_funnel(theta);
    for (int i = 0; i < 3; i++)
    {
        benchmark_fisher_info(log_prob, theta);
    }
}

TEST(Unit, SoftAbsMap)
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

TEST(Unit, Hamiltonian)
{
    auto theta = _theta.clone().requires_grad_();
    auto momentum = _momentum.clone().requires_grad_();
    auto ham_ = ghmc::hamiltonian(alog_funnel, theta, momentum, 0.);
    ASSERT_TRUE(ham_.has_value());
    torch::Tensor energy = std::get<0>(ham_.value());
    auto err = (energy - expected_energy).abs().sum().item<double>();
    ASSERT_NEAR(err, 0., 1e-3);
}

void test_symplectic_flow(torch::DeviceType device = torch::kCPU)
{
    auto theta = _theta.clone().to(device);
    auto momentum = _momentum.clone().to(device);

    auto flow = ghmc::symplectic_flow(/*log_probability_density=*/alog_funnel,
                                      /*params=*/theta, /*momentum=*/momentum,
                                      /*leap_steps=*/1, /*epsilon=*/0.14, /*binding_const=*/10.,
                                      /*jitter=*/0.00001, /*jitter_max=*/0);

    ASSERT_TRUE(flow.has_value());
    auto [p_flow_, m_flow_] = flow.value();
    ASSERT_TRUE(p_flow_.device().type() == device);
    auto p_flow = p_flow_.to(torch::kCPU);
    ASSERT_TRUE(m_flow_.device().type() == device);
    auto m_flow = m_flow_.to(torch::kCPU);
    auto err = (p_flow[-1] - expected_flow_theta).abs().sum().item<double>();
    ASSERT_NEAR(err, 0., 1e-2);
    err = (m_flow[-1] - expected_flow_moment).abs().sum().item<double>();
    ASSERT_NEAR(err, 0., 1e-2);
}

TEST(Unit, SymplecticFlow)
{
    test_symplectic_flow();
}

TEST(Unit, SymplecticFlowCUDA)
{
    ASSERT_TRUE(torch::cuda::is_available());
    test_symplectic_flow(torch::kCUDA);
}
