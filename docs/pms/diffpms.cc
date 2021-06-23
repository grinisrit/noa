#include <noa/utils/common.hh>
#include <noa/pms/constants.hh>
#include <noa/pms/dcs.hh>
#include <noa/ghmc.hh>

#include <torch/extension.h>

using namespace noa::ghmc;
using namespace noa::pms;
using namespace noa::utils;

inline const auto PI = 2.f * torch::acos(torch::tensor(0.f));

inline Tensor mix_density(const Tensor &states, const Tensor &node) {
    return torch::exp(-(states - node.slice(0, 0, 2)).pow(2).sum(-1) / node[2].abs());
}

inline Tensor rot(const Tensor &theta) {
    const auto n = theta.numel();
    const auto c = torch::cos(theta);
    const auto s = torch::sin(theta);
    return torch::stack({c, -s, s, c}).t().view({n, 2, 2});
}

inline std::tuple<Tensor, Tensor> backward_mc(
        Tensor theta,
        Tensor node,
        const int npar) {

    const auto detector = torch::zeros(2);
    const auto material_A = 0.9f;
    const auto material_B = 0.01f;

    const auto length1 = 1.f - 0.2f * torch::rand(npar);
    const auto rot1 = rot(theta);
    auto step1 = torch::stack({torch::zeros(npar), length1}).t();
    step1 = rot1.matmul(step1.view({npar, 2, 1})).view({npar, 2});
    const auto state1 = detector + step1;

    auto biasing = torch::randint(0, 2, {npar});
    auto density = mix_density(state1, node);
    auto weights =
            torch::where(biasing > 0,
                         (density / 0.5) * material_A,
                         ((1 - density) / 0.5) * material_B) * torch::exp(-0.1f * length1);

    const auto length2 = 1.f - 0.2f * torch::rand(npar);
    const auto rot2 = rot(0.05f * PI * (torch::rand(npar) - 0.5f));
    auto step2 = length2.view({npar, 1}) * step1 / length1.view({npar, 1});
    step2 = rot2.matmul(step2.view({npar, 2, 1})).view({npar, 2});
    const auto state2 = state1 + step2;

    biasing = torch::randint(0, 2, {npar});
    density = mix_density(state2, node);
    weights *=
            torch::where(biasing > 0,
                         (density / 0.5) * material_A,
                         ((1 - density) / 0.5) * material_B) * torch::exp(-0.1f * length2);

    return std::make_tuple(weights, torch::stack({state1, state2}).transpose(0, 1));
}

inline auto log_prob_flux(const Tensor &thetas,
                          const Tensor &observed_flux,
                          const float &inv_variance_flux,
                          const Tensor &mean_prior,
                          const float &inv_variance_prior,
                          const int npar) {
    return [thetas, observed_flux, inv_variance_flux, mean_prior, inv_variance_prior, npar](
            const Parameters &parameters) {
        const auto params = parameters.at(0).detach().requires_grad_(true);
        auto log_prob = torch::tensor(0.f);
        const auto num_thetas = thetas.numel();
        for (int i = 0; i < num_thetas; i++)
            log_prob -= inv_variance_flux *
                        (observed_flux[i] - std::get<0>(backward_mc(thetas[i], params, npar)).mean()).pow(2).sum() / 2;
        log_prob -= inv_variance_prior * (mean_prior - params).pow(2).sum() / 2;
        return LogProbabilityGraph{log_prob, {params}};
    };
}

inline std::tuple<Tensor, Tensor> grad_convergence_test(
        Tensor node_params,
        Tensor thetas,
        Tensor observed_flux,
        float inv_variance_flux,
        Tensor mean_prior,
        float inv_variance_prior,
        std::vector<int> num_particles) {
    auto log_probs = Tensors{};
    log_probs.reserve(num_particles.size());
    auto params_grads = Tensors{};
    params_grads.reserve(num_particles.size());

    for (const auto npar : num_particles) {
        const auto log_prob_density =
                log_prob_flux(thetas, observed_flux, inv_variance_flux, mean_prior, inv_variance_prior, npar);
        const auto[log_prob, params] = log_prob_density({node_params});
        const auto log_prob_grad = torch::autograd::grad({log_prob}, params)[0];
        log_probs.push_back(log_prob.detach());
        params_grads.push_back(log_prob_grad);
    }

    return std::make_tuple(torch::stack(log_probs), torch::stack(params_grads));

}

inline std::tuple<Tensor, Tensor> gradient_descent_bmc(
        Tensor thetas,
        Tensor observed_flux,
        Tensor initial_params,
        int num_epochs) {
    float learning_rate = 0.05f;
    const auto log_prob_density =
            log_prob_flux(thetas, observed_flux, 1000.f, initial_params, 0.f, 100000);

    auto log_probs = Tensors{};
    log_probs.reserve(num_epochs);
    auto node_flow = Tensors{};
    node_flow.reserve(num_epochs);

    auto node_params = initial_params.detach().clone();
    std::cout << "Running SGD over " << num_epochs << " iterations with initial data:\n"
              << node_params << "\n";

    for (int i = 0; i < num_epochs; i++) {
        const auto[log_prob, params] = log_prob_density({node_params});

        const auto log_prob_grad = torch::autograd::grad({log_prob}, params)[0];

        node_params += log_prob_grad * learning_rate;

        if (i % 50 == 0)
            std::cout << "log probability:\n" << log_prob
                      << "\nnode " << i << " :\n" << node_params << "\n";

        log_probs.push_back(log_prob.detach());
        node_flow.push_back(node_params.clone());
    }

    return std::make_tuple(torch::stack(log_probs), torch::stack(node_flow));
}

inline Tensor bayesian_backward_mc(
        Tensor thetas,
        Tensor observed_flux,
        float inv_variance_flux,
        Tensor mean_prior,
        float inv_variance_prior,
        int npar,
        int niter,
        int max_flow_steps,
        float step_size,
        float jitter,
        float binding_const) {

    torch::manual_seed(SEED);

    const auto conf = Configuration<float>{}
            .set_max_flow_steps(max_flow_steps)
            .set_step_size(step_size)
            .set_jitter(jitter)
            .set_binding_const(binding_const)
            .set_verbosity(true);

    const auto log_prob =
            log_prob_flux(thetas, observed_flux, inv_variance_flux, mean_prior, inv_variance_prior, npar);

    const auto params_init = Parameters{mean_prior.detach()};

    const auto ham_dym = riemannian_dynamics(
            log_prob, softabs_metric(conf), metropolis_criterion, conf);

    const auto pms_sampler = sampler(ham_dym, full_trajectory, conf);

    const auto samples = pms_sampler(params_init, niter);

    const auto result = stack(samples);
    torch::save(result, "pms_bayesian_sample.pt");

    return result;
}

inline std::tuple<Tensor, Tensor> backward_mc_grad(
        Tensor theta,
        Tensor node) {
    const auto npar = 1;
    const auto detector = torch::zeros(2);
    auto bmc_grad = torch::zeros_like(node);

    const auto material_A = 0.9f;
    const auto material_B = 0.01f;

    const auto length1 = 1.f - 0.2f * torch::rand(npar);
    const auto rot1 = rot(theta);
    auto step1 = torch::stack({torch::zeros(npar), length1}).t();
    step1 = rot1.matmul(step1.view({npar, 2, 1})).view({npar, 2});
    const auto state1 = detector + step1;

    auto biasing = torch::randint(0, 2, {npar});

    auto node_leaf = node.detach().requires_grad_();
    auto density = mix_density(state1, node_leaf);
    auto weights_leaf =
            torch::where(biasing > 0,
                         (density / 0.5) * material_A,
                         ((1 - density) / 0.5) * material_B) * torch::exp(-0.01f * length1);
    bmc_grad += torch::autograd::grad({weights_leaf}, {node_leaf})[0];
    auto weights = weights_leaf.detach();

    const auto length2 = 1.f - 0.2f * torch::rand(npar);
    const auto rot2 = rot(0.05f * PI * (torch::rand(npar) - 0.5f));
    auto step2 = length2.view({npar, 1}) * step1 / length1.view({npar, 1});
    step2 = rot2.matmul(step2.view({npar, 2, 1})).view({npar, 2});
    const auto state2 = state1 + step2;

    biasing = torch::randint(0, 2, {npar});

    node_leaf = node.detach().requires_grad_();
    density = mix_density(state2, node_leaf);
    weights_leaf =
            torch::where(biasing > 0,
                         (density / 0.5) * material_A,
                         ((1 - density) / 0.5) * material_B) * torch::exp(-0.01f * length2);
    const auto weight2 = weights_leaf.detach();
    bmc_grad = weights * torch::autograd::grad({weights_leaf}, {node_leaf})[0] + weight2 * bmc_grad;
    weights *= weight2;

    return std::make_tuple(weights, bmc_grad);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward_mc", &backward_mc, py::call_guard<py::gil_scoped_release>(),
          "Backward MC example");
    m.def("grad_convergence_test", &grad_convergence_test, py::call_guard<py::gil_scoped_release>(),
          "Test convergence for the log probability function and its gradient");
    m.def("gradient_descent_bmc", &gradient_descent_bmc, py::call_guard<py::gil_scoped_release>(),
          "Stochastic gradient descent over backward MC example");
    m.def("bayesian_backward_mc", &bayesian_backward_mc, py::call_guard<py::gil_scoped_release>(),
          "Bayesian Backward MC example");
    m.def("backward_mc_grad", &backward_mc_grad, py::call_guard<py::gil_scoped_release>(),
          "Backward MC with adjoint sensitivity");
}