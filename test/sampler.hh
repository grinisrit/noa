#pragma once

#include <ghmc/ghmc.hh>

#include <gtest/gtest.h>
#include <torch/torch.h>


inline const auto alog_funnel = [](const auto &theta) {
    auto dim = theta.numel() - 1;
    return -0.5 * ((torch::exp(theta[0]) * theta.slice(0, 1, dim + 1).pow(2).sum()) +
                   (theta[0].pow(2) / 9) - dim * theta[0]);
};

inline const auto _theta = torch::tensor({0.8745, -0.4383, 0.5938, 0.1020});
inline const auto _momentum = torch::tensor({-0.4262, -0.5880, 0.0718, 1.0845});

inline const auto expected_val = 0.6038;
inline const auto expected_grad = torch::tensor({0.7373, 1.0509, -1.4237, -0.2446});

inline const auto expected_fisher = torch::tensor(
    {{0.7766, -1.0509, 1.4237, 0.2446},
     {-1.0509, 2.3977, 0.0000, 0.0000},
     {1.4237, 0.0000, 2.3977, 0.0000},
     {0.2446, 0.0000, 0.0000, 2.3977}});

inline const auto expected_spectrum = torch::tensor({0.3745, 2.3977, 2.3977, 3.5488});

inline const auto expected_energy = torch::tensor({1.2519});

inline const auto expected_flow_theta = torch::tensor({0.6839, -0.5335, 0.6796, 0.1806});
inline const auto expected_flow_moment = torch::tensor({-0.5150, -0.3777, -0.1253, 0.9208});

inline void test_fisher_info(torch::DeviceType device = torch::kCPU)
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

inline void test_symplectic_flow(torch::DeviceType device = torch::kCPU)
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