#pragma once

#include <noa/ghmc.hh>
#include <noa/utils/common.hh>

#include <torch/torch.h>
#include <chrono>

using namespace std::chrono;
using namespace noa;
using namespace noa::ghmc;
using namespace noa::utils;

/*
 * Save result for analysis. One can explore the sample in python:
 * next(torch.jit.load(<@param save_result_to>).parameters())
 */
inline void save_result(const Tensor &sample, const Path &save_result_to) {
    std::cout << "Saving result to " << save_result_to << "\n";
    torch::save(sample, save_result_to);
}

inline Status sample_normal_distribution(const Path &save_result_to,
                                         torch::DeviceType device = torch::kCPU) {
    torch::manual_seed(SEED);

    // Sample from the 3 dimensional Gaussian distribution
    // HMC requires the log density up to additive constants
    const auto mean = torch::tensor({0.f, 10.f, 5.f}, torch::device(device));
    const auto sigma = torch::tensor({.5f, 1.f, 2.f}, torch::device(device));

    const auto log_prob_normal = [&mean, &sigma](const Parameters &theta_) {
        const auto theta = theta_.at(0).detach().requires_grad_(true);
        const auto log_prob = -((theta - mean) / sigma).pow(2).sum() / 2;
        return LogProbabilityGraph{log_prob, {theta}};
    };

    std::cout << "Sampling Normal distribution:\n"
              << " mean =\n  "
              << mean.view({1, 3}) << "\n sigma =\n  "
              << sigma.view({1, 3}) << "\n";

    // Initialise parameters
    const auto params_init = Parameters{torch::zeros(3, torch::device(device))};

    // Create sampler
    const auto normal_sampler = sampler(
            log_prob_normal,
            Configuration<float>{}
                    .set_max_flow_steps(10)
                    .set_step_size(0.05).set_verbosity(true));

    // Run sampler
    const auto begin = steady_clock::now();
    const auto samples = normal_sampler(params_init, 2);
    const auto end = steady_clock::now();
    std::cout << "GHMC: sampler took " << duration_cast<microseconds>(end - begin).count() / 1E+6
              << " seconds" << std::endl;
    const auto num_samples = samples.size();
    if (num_samples <= 1) {
        std::cerr << "Sampler failed\n";
        return false;
    }

    auto result_ = Tensors{};
    result_.reserve(num_samples);
    for(const auto &sample: samples)
        result_.push_back(sample.at(0));
    const auto result = torch::stack(result_);
    save_result(result, save_result_to);

    /*
    const auto[s_sigma, s_mean] = torch::std_mean(result.slice(0, result.size(0) / 10, result.size(0)), 0, true, true);

    std::cout << "Sample statistics:\n"
              << " mean =\n  "
              << s_mean << "\n sigma =\n  "
              << s_sigma << "\n";
    */
    return true;

}
/*
inline Status sample_funnel_distribution(const Path &save_result_to,
                                         torch::DeviceType device = torch::kCPU) {
    torch::manual_seed(utils::SEED);

    // Sample from the 10 dimensional Funnel distribution
    // HMC requires the log density up to additive constants
    const auto alog_funnel = [](const auto &w) {
        const auto dim = w.numel() - 1;
        return -((torch::exp(w[0]) * w.slice(0, 1, dim + 1).pow(2).sum()) +
                 (w[0].pow(2) / 9) - dim * w[0]) / 2;
    };

    // Initialise parameters
    auto params_init = torch::ones(11, torch::device(device));
    params_init[0] = 0.;

    // Create sampler
    const auto sampler = ghmc::sampler(
            alog_funnel,
            ghmc::Configuration<float>{}
                    .set_max_flow_steps(25)
                    .set_jitter(0.001)
                    .set_step_size(0.14f)
                    .set_binding_const(10.f).set_verbosity(true));

    // Run sampler
    const auto begin = steady_clock::now();
    const auto sample = sampler(params_init, 100);
    const auto end = steady_clock::now();
    std::cout << "GHMC: sampler took " << duration_cast<microseconds>(end - begin).count() / 1E+6
              << " seconds" << std::endl;
    if (sample.size() <= 1) {
        std::cerr << "Sampler failed\n";
        return false;
    }

    save_result(torch::stack(sample), save_result_to);

    return true;
}
*/
/////////////////////////////////////////////////////////////////////////////////////

inline Status sample_normal_dist(const Path &save_result_to,
                                 torch::DeviceType device = torch::kCPU) {
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
    if (!result.has_value()) {
        std::cerr << "Sampler failed\n";
        return false;
    }
    auto[acc_rate, sample] = result.value();

    save_result(sample, save_result_to);

    auto[s_sigma, s_mean] = torch::std_mean(sample.slice(0, sample.size(0) / 10, sample.size(0)), 0, true, true);

    std::cout << "Sample statistics:\n"
              << " mean =\n  "
              << s_mean << "\n sigma =\n  "
              << s_sigma << "\n";
    return true;
}

inline Status sample_funnel_dist(const Path &save_result_to,
                                 torch::DeviceType device = torch::kCPU) {
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
    if (!result.has_value()) {
        std::cerr << "Sampler failed\n";
        return false;
    };

    save_result(std::get<1>(result.value()), save_result_to);

    return true;
}