#include "noa/ghmc.hh"
#include "noa/utils/common.hh"

#include <torch/extension.h>

using namespace noa;
using namespace noa::ghmc;
using namespace noa::utils;


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> generate_data(int n_tr, int n_val){
    torch::manual_seed(SEED);
    const auto x_val = torch::linspace(-5.f, 5.f, n_val).view({-1, 1});
    const auto y_val = torch::sin(x_val);

    const auto x_train = torch::linspace(-3.14f, 3.14f, n_tr).view({-1, 1});
    const auto y_train = torch::sin(x_train) + 0.1f * torch::randn_like(x_train);

    return std::make_tuple(x_train, y_train, x_val, y_val);
}

torch::Tensor train_jit_module(
        std::string jit_model_pt,
        torch::Tensor x_train,
        torch::Tensor y_train,
        torch::Tensor x_val,
        int nepochs) {

    auto module = load_module(jit_model_pt);
    if (!module.has_value())
        return torch::Tensor{};

    auto net = module.value();
    net.train();

    auto inputs_val = std::vector<torch::jit::IValue>{x_val};
    auto inputs_train = std::vector<torch::jit::IValue>{x_train};

    auto loss_fn = torch::nn::MSELoss{};
    auto optimizer = torch::optim::Adam{parameters(net), torch::optim::AdamOptions(0.005)};

    auto adam_preds = std::vector<at::Tensor>{};
    adam_preds.reserve(nepochs);

    for (int i = 0; i < nepochs; i++) {

        optimizer.zero_grad();
        auto output = net.forward(inputs_train).toTensor();
        auto loss = loss_fn(output, y_train);
        loss.backward();
        optimizer.step();

        adam_preds.push_back(net.forward(inputs_val).toTensor().detach());
    }

    return torch::stack(adam_preds);
}

torch::Tensor sample_jit_module(
        std::string jit_model_pt,
        std::string save_sample_pt,
        torch::Tensor x_train,
        torch::Tensor y_train,
        int niter) {

    auto module = load_module(jit_model_pt);
    if (!module.has_value())
        return torch::Tensor{};

    auto net = module.value();

    auto inputs_train = std::vector<torch::jit::IValue>{x_train};

    auto params_init = parameters(net);

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

    const auto samples = bnet_sampler(params_init, niter);

    const auto result = stack(samples);
    torch::save(result, save_sample_pt);

    return result;
}

torch::Tensor compute_posterior_mean_prediction(
        std::string jit_model_pt, torch::Tensor x_val, torch::Tensor sample) {

    auto module = load_module(jit_model_pt);
    if (!module.has_value())
        return torch::Tensor{};

    auto net = module.value();
    auto inputs_val = std::vector<torch::jit::IValue>{x_val};

    const auto stationary_sample = sample.slice(0, sample.size(0) / 10);

    set_flat_parameters(net, stationary_sample.mean(0));
    const auto posterior_mean_pred = net.forward(inputs_val).toTensor().detach();

    return posterior_mean_pred;
}

torch::Tensor compute_bayes_predictions(
        std::string jit_model_pt, torch::Tensor x_val, torch::Tensor sample) {

    auto module = load_module(jit_model_pt);
    if (!module.has_value())
        return torch::Tensor{};

    auto net = module.value();
    auto inputs_val = std::vector<torch::jit::IValue>{x_val};

    const auto stationary_sample = sample.slice(0, sample.size(0) / 5);

    auto bayes_preds_ = Tensors{};
    bayes_preds_.reserve(stationary_sample.size(0));
    for (uint32_t i = 0; i < stationary_sample.size(0); i++) {
        set_flat_parameters(net, stationary_sample[i]);
        bayes_preds_.push_back(net.forward(inputs_val).toTensor().detach());
    }
    const auto bayes_preds = torch::stack(bayes_preds_);

    return bayes_preds;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("generate_data", &generate_data, py::call_guard<py::gil_scoped_release>(),
          "Generate synthetic data");
    m.def("train_jit_module", &train_jit_module, py::call_guard<py::gil_scoped_release>(),
          "SGD training for a TorchScript module");
    m.def("sample_jit_module", &sample_jit_module, py::call_guard<py::gil_scoped_release>(),
          "Bayesian training for a TorchScript module");
    m.def("compute_posterior_mean_prediction", &compute_posterior_mean_prediction,
          py::call_guard<py::gil_scoped_release>(),
          "Compute posterior mean prediction");
    m.def("compute_bayes_predictions", &compute_bayes_predictions,
          py::call_guard<py::gil_scoped_release>(),
          "Compute Bayes sample predictions");
}