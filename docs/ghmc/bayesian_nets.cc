#include "noa/ghmc.hh"
#include "noa/utils/common.hh"

#include <iostream>
#include <torch/extension.h>

using namespace noa;
using namespace noa::utils;

torch::Tensor run_ghmc() {
    torch::manual_seed(utils::SEED);

    // Sample from the 3 dimensional Gaussian distribution
    // HMC requires the log density up to additive constants
    auto mean = torch::tensor({0., 5., 10.});
    auto sigma = torch::tensor({.5, 1., 2.});
    auto alog_prob_normal = [&mean, &sigma](const auto &theta) {
        return -0.5 * ((theta - mean) / sigma).pow(2).sum();
    };

    // Initialise parameters
    auto params_init = torch::zeros(3);

    // Create sampler configuration
    auto conf = ghmc::SampleConf{}
            .set_num_iter(200)
            .set_leap_steps(5)
            .set_epsilon(0.3);

    // Run sampler
    auto result = ghmc::sample(alog_prob_normal, params_init, conf);

    // Check result
    if (!result.has_value())
    {
        return torch::Tensor{};
    }
    return std::get<1>(result.value()).detach();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_ghmc", &run_ghmc, py::call_guard<py::gil_scoped_release>(), "GHMC example");
}