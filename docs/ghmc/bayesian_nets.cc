#include "noa/ghmc.hh"
#include "noa/utils/common.hh"

#include <torch/extension.h>

using namespace noa;
using namespace noa::utils;


torch::Tensor train_jit_module(
        std::string jit_model_pt,
        torch::Tensor x_val,
        torch::Tensor x_train,
        torch::Tensor y_train,
        int nepochs) {

    auto module = load_module(jit_model_pt);
    if (!module.has_value())
        return torch::Tensor{};

    auto net = module.value();
    auto params = parameters(net);

    auto loss_fn = torch::nn::MSELoss{};
    auto optimizer = torch::optim::Adam{params, torch::optim::AdamOptions(0.005)};

    auto inputs_val = std::vector<torch::jit::IValue>{x_val};
    auto inputs_train = std::vector<torch::jit::IValue>{x_train};

    auto preds = std::vector<at::Tensor>{};
    preds.reserve(nepochs);

    net.train();

    for (int i = 0; i < nepochs; i++) {

        optimizer.zero_grad();
        auto output = net.forward(inputs_train).toTensor();
        auto loss = loss_fn(output, y_train);
        loss.backward();
        optimizer.step();

        preds.push_back(net.forward(inputs_val).toTensor().detach());
    }

    return torch::stack(preds);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("train_jit_module", &train_jit_module, py::call_guard<py::gil_scoped_release>(),
          "Train a TorchScript module");
}