#include "noa/ghmc.hh"
#include "noa/utils/common.hh"

#include <torch/extension.h>

using namespace noa;
using namespace noa::ghmc;
using namespace noa::utils;


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> generate_data(int n_tr, int n_val){
    torch::manual_seed(SEED);
    const auto x_val = torch::linspace(-4.f, 4.f, n_val).view({-1, 1});
    const auto y_val = torch::sin(x_val);

    const auto x_train = torch::linspace(-3.54f, 3.54f, n_tr).view({-1, 1});
    const auto y_train = torch::sin(x_train) + 0.1f * torch::randn_like(x_train);

    return std::make_tuple(x_train, y_train, x_val, y_val);
}

std::tuple<torch::Tensor, torch::Tensor> train_jit_module(
        std::string jit_model_pt,
        torch::Tensor x_train,
        torch::Tensor y_train,
        torch::Tensor x_val,
        int nepochs) {
    torch::manual_seed(SEED);
    auto module = load_module(jit_model_pt);
    if (!module.has_value())
        return std::make_tuple(torch::Tensor{},torch::Tensor{}) ;
    auto &net = module.value();
    net.train();

    auto loss_fn = torch::nn::MSELoss{};
    auto optimizer = torch::optim::Adam{parameters(net), torch::optim::AdamOptions(0.005)};

    auto adam_preds = std::vector<at::Tensor>{};
    adam_preds.reserve(nepochs);

    for (int i = 0; i < nepochs; i++) {

        optimizer.zero_grad();
        auto output = net({x_train}).toTensor();
        auto loss = loss_fn(output, y_train);
        loss.backward();
        optimizer.step();

        adam_preds.push_back(net({x_val}).toTensor().detach());
    }

    return std::make_tuple(flat_parameters(net, true), torch::stack(adam_preds));
}

torch::Tensor sample_jit_module(
        std::string jit_model_pt,
        std::string save_sample_pt,
        torch::Tensor x_train,
        torch::Tensor y_train,
        torch::Tensor prior_flat_params,
        float model_variance,
        float params_variance,
        int niter,
        int max_flow_steps,
        float step_size) {

    torch::manual_seed(SEED);

    auto module = load_module(jit_model_pt);
    if (!module.has_value())
        return torch::Tensor{};
    auto &net = module.value();
    net.train();

    set_flat_parameters(net, prior_flat_params, true);
    const auto prior_params = parameters(net, true);

    const auto log_prob_bnet = numerics::regression_log_probability(
            net, model_variance, prior_params, params_variance)(x_train, y_train);

    const auto conf_bnet = Configuration<float>{}
            .set_max_flow_steps(max_flow_steps)
            .set_step_size(step_size)
            .set_verbosity(true);

    const auto net_params = parameters(net);
    const auto ham_dym = euclidean_dynamics(
            log_prob_bnet, identity_metric_like(net_params), metropolis_criterion, conf_bnet);
    const auto bnet_sampler = sampler(ham_dym, full_trajectory, conf_bnet);

    const auto samples = bnet_sampler(net_params, niter);

    const auto result = stack(samples);
    torch::save(result, save_sample_pt);

    return result;
}

torch::Tensor compute_posterior_mean_prediction(
        std::string jit_model_pt, torch::Tensor x_val, torch::Tensor sample, int burn) {

    auto module = load_module(jit_model_pt);
    if (!module.has_value())
        return torch::Tensor{};
    auto &net = module.value();

    const auto stationary_sample = sample.slice(0, sample.size(0) / burn);
    set_flat_parameters(net, stationary_sample.mean(0));
    const auto posterior_mean_pred = net({x_val}).toTensor().detach();

    return posterior_mean_pred;
}

torch::Tensor compute_bayes_predictions(
        std::string jit_model_pt, torch::Tensor x_val, torch::Tensor sample) {

    auto module = load_module(jit_model_pt);
    if (!module.has_value())
        return torch::Tensor{};
    auto &net = module.value();

    auto bayes_preds_ = Tensors{};
    bayes_preds_.reserve(sample.size(0));
    for (uint32_t i = 0; i < sample.size(0); i++) {
        set_flat_parameters(net, sample[i]);
        bayes_preds_.push_back(net({x_val}).toTensor().detach());
    }
    const auto bayes_preds = torch::stack(bayes_preds_);

    return bayes_preds;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("generate_data", &generate_data, py::call_guard<py::gil_scoped_release>(),
          "Generate synthetic data");
    m.def("train_jit_module", &train_jit_module, py::call_guard<py::gil_scoped_release>(),
          "Likelihood training for a TorchScript module");
    m.def("sample_jit_module", &sample_jit_module, py::call_guard<py::gil_scoped_release>(),
          "Bayesian training for a TorchScript module");
    m.def("compute_posterior_mean_prediction", &compute_posterior_mean_prediction,
          py::call_guard<py::gil_scoped_release>(),
          "Compute posterior mean prediction");
    m.def("compute_bayes_predictions", &compute_bayes_predictions,
          py::call_guard<py::gil_scoped_release>(),
          "Compute Bayes sample predictions");
}