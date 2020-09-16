#include <gtest/gtest.h>
#include <torch/torch.h>

#include <ghmc/ghmc.hh>

void test_normal_dist(torch::DeviceType device = torch::kCPU)
{
    auto mean = torch::tensor({0., 10., 5.}, torch::device(device));
    auto sigma = torch::tensor({.5, 1., 2.}, torch::device(device));
    auto alog_prob_normal = [&mean, &sigma](const auto &theta) {
        return -0.5 * ((theta - mean) / sigma).pow(2).sum();
    };
    auto params_init = torch::zeros(3, torch::device(device));
    auto conf = ghmc::SampleConf{}
                    .set_num_iter(200)
                    .set_leap_steps(5)
                    .set_epsilon(0.3);
    auto result = ghmc::sample(alog_prob_normal, params_init, conf);

    ASSERT_TRUE(result.has_value());
    // save result for analysis
    auto [acc_rate, sample] = result.value();
    ASSERT_TRUE(sample.device().type() == device);
    torch::save(sample, "normal_ghmc_sample.pt");

    ASSERT_GE(acc_rate, 0.9);
    auto [s_sigma, s_mean] = torch::std_mean(sample.slice(0, sample.size(0) / 10, sample.size(0)), 0, true, true);
    auto err_mean = (s_mean - mean).abs().sum().item<double>();
    auto err_sigma = (s_sigma - sigma).abs().sum().item<double>();
    ASSERT_NEAR(err_mean, 0., 1.);
    ASSERT_NEAR(err_sigma, 0., 0.5);
}

TEST(Functional, NormalDist)
{
    test_normal_dist();
}

TEST(Functional, NormalDistCUDA)
{
    ASSERT_TRUE(torch::cuda::is_available());
    test_normal_dist(torch::kCUDA);
}

TEST(Functional, DISABLED_Funnel)
{
    auto alog_funnel = [](const auto &w) {
        auto dim = w.numel() - 1;
        return -0.5 * ((torch::exp(w[0]) * w.slice(0, 1, dim + 1).pow(2).sum()) +
                       (w[0].pow(2) / 9) - dim * w[0]);
    };
    auto params_init = torch::ones(11);
    params_init[0] = 0.;
    auto conf = ghmc::SampleConf{}
                    .set_num_iter(100)
                    .set_leap_steps(25)
                    .set_epsilon(0.14)
                    .set_binding_const(10.);
    auto result = ghmc::sample(alog_funnel, params_init, conf);
    ASSERT_TRUE(result.has_value());
    // save result for analysis
    auto [acc_rate, sample] = result.value();
    torch::save(sample, "funnel_ghmc_sample.pt");
    ASSERT_GE(acc_rate, 0.8);
}