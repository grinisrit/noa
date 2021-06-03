#pragma once

#include "test-data.hh"

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
                    .set_max_flow_steps(5)
                    .set_step_size(0.3f).set_verbosity(true));

    // Run sampler
    const auto begin = steady_clock::now();
    const auto samples = normal_sampler(params_init, 200);
    const auto end = steady_clock::now();
    std::cout << "GHMC: sampler took " << duration_cast<microseconds>(end - begin).count() / 1E+6
              << " seconds" << std::endl;
    const auto num_samples = samples.size();
    if (num_samples <= 1) {
        std::cerr << "Sampler failed\n";
        return false;
    }

    const auto result = stack(samples);
    save_result(result, save_result_to);
    const auto[s_sigma, s_mean] = torch::std_mean(result.slice(0, result.size(0) / 10, result.size(0)), 0, true, true);

    std::cout << "Sample statistics:\n"
              << " mean =\n  "
              << s_mean << "\n sigma =\n  "
              << s_sigma << "\n";

    return true;
}


inline Status sample_funnel_distribution(const Path &save_result_to,
                                         torch::DeviceType device = torch::kCPU) {
    torch::manual_seed(SEED);

    std::cout << "Sample from the 10 dimensional Funnel distribution:\n";

    // Initialise parameters
    auto params_init = torch::ones(11, torch::device(device));
    params_init[0] = 0.;

    // Create sampler
    const auto funnel_sampler = ghmc::sampler(
            log_funnel,
            Configuration<float>{}
                    .set_max_flow_steps(25)
                    .set_jitter(0.001f)
                    .set_step_size(0.14f)
                    .set_binding_const(10.f).set_verbosity(true));

    // Run sampler
    const auto begin = steady_clock::now();
    const auto samples = funnel_sampler(Parameters{params_init}, 100);
    const auto end = steady_clock::now();
    std::cout << "GHMC: sampler took " << duration_cast<microseconds>(end - begin).count() / 1E+6
              << " seconds" << std::endl;
    if (samples.size() <= 1) {
        std::cerr << "Sampler failed\n";
        return false;
    }

    const auto result = stack(samples);
    save_result(result, save_result_to);

    return true;
}


inline Status sample_bayesian_net(const Path &save_result_to,
                                  torch::DeviceType device = torch::kCPU) {
    torch::manual_seed(SEED);

    std::cout << "Bayesian Deep Learning regression example:\n";

    auto module = load_module(jit_net_pt);
    if (!module.has_value())
        return false;

    const auto n_tr = 6;
    const auto n_val = 300;
    const auto n_epochs = 250;

    const auto x_val = torch::linspace(-5.f, 5.f, n_val, torch::device(device)).view({-1, 1});
    const auto y_val = torch::sin(x_val);

    const auto x_train = torch::linspace(-3.14f, 3.14f, n_tr, torch::device(device)).view({-1, 1});
    const auto y_train = torch::sin(x_train) + 0.1f * torch::randn_like(x_train);

    auto &net = module.value();
    net.train();
    net.to(device);
    const auto params_init = parameters(net);
    const auto inputs_val = std::vector<torch::jit::IValue>{x_val};
    const auto inputs_train = std::vector<torch::jit::IValue>{x_train};

    auto loss_fn = torch::nn::MSELoss{};
    auto optimizer = torch::optim::Adam{params_init, torch::optim::AdamOptions(0.005)};


    auto preds = Tensors{};
    preds.reserve(n_epochs);

    std::cout << " Running Adam gradient descent optimisation ...\n";
    for (uint32_t i = 0; i < n_epochs; i++) {

        optimizer.zero_grad();
        const auto output = net.forward(inputs_train).toTensor();
        const auto loss = loss_fn(output, y_train);
        loss.backward();
        optimizer.step();

        preds.push_back(net.forward(inputs_val).toTensor().detach());
    }

    std::cout << " Initial MSE loss:\n" << loss_fn(preds.front(), y_val) << "\n"
              << " Optimal MSE loss:\n" << loss_fn(preds.back(), y_val) << "\n";


    const auto log_prob_bnet = [&net, &inputs_train, &y_train](const Parameters &theta) {
        uint32_t i = 0;
        auto log_prob = torch::tensor(0, y_train.options());
        for (const auto &param: net.parameters()) {
            param.set_data(theta.at(i).detach());
            log_prob += param.pow(2).sum();
            i++;
        }
        const auto output = net.forward(inputs_train).toTensor();
        log_prob = -50 * (y_train - output).pow(2).sum() - log_prob / 2;
        return LogProbabilityGraph{log_prob, parameters(net)};
    };

    const auto conf_bnet = Configuration<float>{}
            .set_max_flow_steps(10)
            .set_jitter(0.01f)
            .set_step_size(0.001f)
            .set_binding_const(10.f)
            .set_verbosity(true);

    const auto bnet_sampler = sampler(log_prob_bnet, conf_bnet);

    // Run sampler
    const auto begin = steady_clock::now();
    const auto samples = bnet_sampler(params_init, 0);
    const auto end = steady_clock::now();
    std::cout << "GHMC: sampler took " << duration_cast<microseconds>(end - begin).count() / 1E+6
              << " seconds" << std::endl;
    if (samples.size() <= 1) {
        std::cerr << "Sampler failed\n";
        return false;
    }

    const auto result = stack(samples);
    save_result(result, save_result_to);
    //const auto result = load_tensor(save_result_to).value();

    set_flat_parameters(net, result.slice(0, result.size(0) / 5).mean(0));
    const auto posterior_pred = net.forward(inputs_val).toTensor();

    std::cout << " Posterior MSE loss:\n" << loss_fn(posterior_pred, y_val) << "\n";

    return true;
}

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