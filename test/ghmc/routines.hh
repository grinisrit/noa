#pragma once

#include <noa/ghmc.hh>
#include <noa/utils/common.hh>

#include <torch/torch.h>

using namespace noa;
using namespace noa::utils;

/*
 * Save result for analysis. One can explore the sample in python:
 * next(torch.jit.load(<@param save_result_to>).parameters())
 */
inline void save_result(const torch::Tensor &sample, const Path &save_result_to)
{
    std::cout << "Saving result to " << save_result_to << "\n";
    torch::save(sample, save_result_to);
}


inline Status sample_normal_dist(const Path &save_result_to,
                                        torch::DeviceType device = torch::kCPU)
{
    torch::manual_seed(utils::SEED);

    // Sample from the 3 dimensional Gaussian distribution
    // HMC requires the log density up to additive constants
    auto mean = torch::tensor({0., 10., 5.}, torch::device(device));
    auto sigma = torch::tensor({.5, 1., 2.}, torch::device(device));
    auto alog_prob_normal = [&mean, &sigma](const auto &theta) {
        return -0.5 * ((theta - mean) / sigma).pow(2).sum();
    };

    std::cout << "Sampling Normal distribution:\n"
              << " mean =\n  "
              << mean.view({1, 3}) << "\n sigma =\n  "
              << sigma.view({1, 3}) << "\n";

    // Initialise parameters
    auto params_init = torch::zeros(3, torch::device(device));

    // Create sampler configuration
    auto conf = ghmc::SampleConf{}
                    .set_num_iter(200)
                    .set_leap_steps(5)
                    .set_epsilon(0.3);

    // Run sampler
    auto result = ghmc::sample(alog_prob_normal, params_init, conf);
    if (!result.has_value())
    {
        std::cerr << "Sampler failed\n";
        return false;
    }
    auto [acc_rate, sample] = result.value();

    save_result(sample, save_result_to);

    auto [s_sigma, s_mean] = torch::std_mean(sample.slice(0, sample.size(0) / 10, sample.size(0)), 0, true, true);

    std::cout << "Sample statistics:\n"
              << " mean =\n  "
              << s_mean << "\n sigma =\n  "
              << s_sigma << "\n";
    return true;
}

inline Status sample_funnel_dist(const Path &save_result_to,
                                 torch::DeviceType device = torch::kCPU)
{
    torch::manual_seed(utils::SEED);

    // Sample from the 10 dimensional Funnel distribution
    // HMC requires the log density up to additive constants
    auto alog_funnel = [](const auto &w) {
        auto dim = w.numel() - 1;
        return -0.5 * ((torch::exp(w[0]) * w.slice(0, 1, dim + 1).pow(2).sum()) +
                       (w[0].pow(2) / 9) - dim * w[0]);
    };

    // Initialise parameters
    auto params_init = torch::ones(11, torch::device(device));
    params_init[0] = 0.;

    // Create sampler configuration
    auto conf = ghmc::SampleConf{}
                    .set_num_iter(100)
                    .set_leap_steps(25)
                    .set_epsilon(0.14)
                    .set_binding_const(10.);

    // Run sampler
    auto result = ghmc::sample(alog_funnel, params_init, conf);
    if (!result.has_value())
    {
        std::cerr << "Sampler failed\n";
        return false;
    };

    save_result(std::get<1>(result.value()), save_result_to);

    return true;
}