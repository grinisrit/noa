/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Roland Grinis, GrinisRIT ltd. (roland.grinis@grinisrit.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "noa/utils/numerics.hh"

#include <iostream>
#include <chrono>

#include <torch/torch.h>

namespace noa::ghmc {

    using Parameters = torch::Tensor;
    using Momentum = torch::Tensor;
    using MomentumOpt = std::optional<Momentum>;
    using LogProbability = torch::Tensor;
    using Spectrum = torch::Tensor;
    using Rotation = torch::Tensor;
    using MetricDecomposition = std::tuple<Spectrum, Rotation>;
    using MetricDecompositionOpt = std::optional<MetricDecomposition>;
    using Energy = torch::Tensor;
    using PhaseSpaceFoliation = std::tuple<Parameters, Momentum, Energy>;
    using PhaseSpaceFoliationOpt = std::optional<PhaseSpaceFoliation>;
    using ParametersFlow = std::vector<Parameters>;
    using MomentumFlow = std::vector<Momentum>;
    using EnergyFluctuation = std::vector<Energy>;
    using HamiltonianFlow = std::tuple<ParametersFlow, MomentumFlow, EnergyFluctuation>;
    using ParametersGradient = torch::Tensor;
    using MomentumGradient = torch::Tensor;
    using HamiltonianGradient = std::tuple<ParametersGradient, MomentumGradient>;
    using HamiltonianGradientOpt = std::optional<HamiltonianGradient>;
    using Samples = std::vector<Parameters>;

    //////////////////////
    using SafeResult = std::optional<torch::Tensor>;
    using SampleResult = std::optional<std::tuple<double, torch::Tensor>>;
    using Params = torch::Tensor;
    using LogProb = torch::Tensor;
    using FisherInfo = std::optional<torch::Tensor>;
    using SoftAbsMap = std::optional<std::tuple<torch::Tensor, torch::Tensor>>;
    using HamiltonianRef = std::optional<std::tuple<torch::Tensor, std::optional<Momentum>>>;
    using SymplecticFlow = std::optional<std::tuple<torch::Tensor, torch::Tensor>>;
    ///////////////////////


    template<typename Dtype>
    struct Configuration {
        uint32_t max_flow_steps = 3;
        Dtype step_size = 0.1;
        Dtype binding_const = 100.;
        Dtype cutoff = 1e-12;
        Dtype jitter = 1e-6;
        Dtype softabs_const = 1e6;
        bool verbose = false;

        inline Configuration &set_max_flow_steps(const Dtype &max_flow_steps_) {
            max_flow_steps = max_flow_steps_;
            return *this;
        }

        inline Configuration &set_step_size(const Dtype &step_size_) {
            step_size = step_size_;
            return *this;
        }

        inline Configuration &set_binding_const(const Dtype &binding_const_) {
            binding_const = binding_const_;
            return *this;
        }

        inline Configuration &set_cutoff(const Dtype &cutoff_) {
            cutoff = cutoff_;
            return *this;
        }

        inline Configuration &set_jitter(const Dtype &jitter_) {
            jitter = jitter_;
            return *this;
        }

        inline Configuration &set_softabs_const(const Dtype &softabs_const_) {
            softabs_const = softabs_const_;
            return *this;
        }

        inline Configuration &set_verbosity(bool verbose_) {
            verbose = verbose_;
            return *this;
        }
    };

    template<typename Configurations>
    inline auto softabs_metric(const Configurations &conf) {
        return [&conf](const LogProbability &log_prob, const Parameters &params) {
            const auto hess_ = utils::numerics::hessian(log_prob, params);
            if (!hess_.has_value()) {
                if (conf.verbose)
                    std::cerr << "GHMC: failed to compute hessian for log probability:\n"
                              << log_prob << "\n";
                return MetricDecompositionOpt{};
            }
            const auto n = params.numel();
            auto[eigs, rotation] =
            torch::symeig(
                    -hess_.value() + conf.jitter * torch::eye(n, params.options()) * torch::rand_like(params),
                    true);
            eigs = torch::where(eigs.abs() >= conf.cutoff, eigs, torch::tensor(conf.cutoff, params.options()));
            const auto spectrum = torch::abs((1 / torch::tanh(conf.softabs_const * eigs)) * eigs);

            return MetricDecompositionOpt{MetricDecomposition{spectrum, rotation}};
        };
    }

    template<typename LogProbabilityDensity, typename Configurations>
    inline auto hamiltonian(const LogProbabilityDensity &log_prob_density, const Configurations &conf) {
        const auto local_metric = softabs_metric(conf);
        return [log_prob_density, local_metric, conf](
                const Parameters &parameters,
                const MomentumOpt &momentum_ = std::nullopt) {
            const auto params = parameters.detach().requires_grad_(true);
            const auto log_prob = log_prob_density(params);
            const auto metric = local_metric(log_prob, params);
            if (!metric.has_value()) {
                if (conf.verbose)
                    std::cerr << "GHMC: failed to compute local metric for log probability:\n"
                              << log_prob << "\n";
                return PhaseSpaceFoliationOpt{};
            }
            const auto&[spectrum, rotation] = metric.value();

            const auto momentum_lift = momentum_.has_value()
                                       ? momentum_.value()
                                       : rotation.mv(torch::sqrt(spectrum) * torch::randn_like(parameters));
            const auto momentum = momentum_lift.detach().requires_grad_(true);

            const auto first_order_term = spectrum.log().sum() / 2;
            const auto mass = rotation.mm(torch::diag(1 / spectrum)).mm(rotation.t());
            const auto second_order_term = momentum.dot(mass.mv(momentum)) / 2;
            return PhaseSpaceFoliationOpt{
                    PhaseSpaceFoliation{params, momentum, -log_prob + first_order_term + second_order_term}};
        };
    }

    template<typename Configurations>
    inline auto hamiltonian_gradient(const Configurations &conf) {
        return [conf](const PhaseSpaceFoliationOpt &foliation) {
            if (!foliation.has_value()) {
                if (conf.verbose)
                    std::cerr << "GHMC: no phase space foliation provided.\n";
                return HamiltonianGradientOpt{};
            }
            const auto &[params, momentum, energy] = foliation.value();
            const auto ham_grad = torch::autograd::grad({energy}, {params, momentum});

            const auto params_grad = ham_grad[0].detach();
            const auto check_params = params_grad.sum();
            if (torch::isnan(check_params).item<bool>() || torch::isinf(check_params).item<bool>()) {
                if (conf.verbose)
                    std::cerr << "GHMC: failed to compute parameters gradient for Hamiltonian:\n"
                              << energy << "\n";
                return HamiltonianGradientOpt{};
            }

            const auto momentum_grad = ham_grad[1].detach();
            const auto check_momentum = momentum_grad.sum();
            if (torch::isnan(check_momentum).item<bool>() || torch::isinf(check_momentum).item<bool>()) {
                if (conf.verbose)
                    std::cerr << "GHMC: failed to compute momentum gradient for Hamiltonian:\n"
                              << energy << "\n";
                return HamiltonianGradientOpt{};
            }

            return HamiltonianGradientOpt{HamiltonianGradient{params_grad, momentum_grad}};
        };
    }

    template<typename LogProbabilityDensity, typename Configurations>
    inline auto hamiltonian_flow(const LogProbabilityDensity &log_prob_density, const Configurations &conf) {
        const auto ham = hamiltonian(log_prob_density, conf);
        const auto ham_grad = hamiltonian_gradient(conf);
        const auto theta = 2 * conf.binding_const * conf.step_size;
        const auto rot = std::make_tuple(cos(theta), sin(theta));
        return [ham, ham_grad, conf, rot](const Parameters &parameters,
                                          const MomentumOpt &momentum_ = std::nullopt) {

            auto params_flow = ParametersFlow{};
            params_flow.reserve(conf.max_flow_steps + 1);

            auto momentum_flow = MomentumFlow{};
            momentum_flow.reserve(conf.max_flow_steps + 1);

            auto energy_fluctuation = EnergyFluctuation{};
            energy_fluctuation.reserve(conf.max_flow_steps + 1);

            auto foliation = ham(parameters, momentum_);
            if (!foliation.has_value()) {
                if (conf.verbose)
                    std::cerr << "GHMC: failed to initialise Hamiltonian flow.\n";
                return HamiltonianFlow{params_flow, momentum_flow, energy_fluctuation};
            }

            const auto &[initial_params, initial_momentum, energy_level] = foliation.value();
            params_flow.push_back(initial_params.detach());
            momentum_flow.push_back(initial_momentum.detach());
            energy_fluctuation.push_back(energy_level.detach());

            uint32_t iter_step = 0;
            if (iter_step >= conf.max_flow_steps)
                return HamiltonianFlow{params_flow, momentum_flow, energy_fluctuation};

            const auto error_msg = [&iter_step, &conf]() {
                if (conf.verbose)
                    std::cerr << "GHMC: failed to evolve flow at step: "
                              << iter_step << "/" << conf.max_flow_steps << "\n";
            };

            auto dynamics = ham_grad(foliation);
            if (!dynamics.has_value()) {
                error_msg();
                return HamiltonianFlow{params_flow, momentum_flow, energy_fluctuation};
            }

            const auto delta = conf.step_size / 2;
            const auto &[c, s] = rot;

            auto params = params_flow[0];
            auto momentum_copy = momentum_flow[0];
            auto params_copy = params + std::get<1>(dynamics.value()) * delta;
            auto momentum = momentum_copy - std::get<0>(dynamics.value()) * delta;

            for (iter_step = 0; iter_step < conf.max_flow_steps; iter_step++) {

                foliation = ham(params_copy, momentum);
                dynamics = ham_grad(foliation);
                if (!dynamics.has_value()) {
                    error_msg();
                    break;
                }

                params = params + std::get<1>(dynamics.value()) * delta;
                momentum_copy = momentum_copy - std::get<0>(dynamics.value()) * delta;

                params = (params + params_copy + c * (params - params_copy) + s * (momentum - momentum_copy)) / 2;
                momentum = (momentum + momentum_copy - s * (params - params_copy) + c * (momentum - momentum_copy)) / 2;
                params_copy = (params + params_copy - c * (params - params_copy) - s * (momentum - momentum_copy)) / 2;
                momentum_copy =
                        (momentum + momentum_copy + s * (params - params_copy) - c * (momentum - momentum_copy)) / 2;

                foliation = ham(params_copy, momentum);
                dynamics = ham_grad(foliation);
                if (!dynamics.has_value()) {
                    error_msg();
                    break;
                }
                params = params + std::get<1>(dynamics.value()) * delta;
                momentum_copy = momentum_copy - std::get<0>(dynamics.value()) * delta;

                foliation = ham(params, momentum_copy);
                dynamics = ham_grad(foliation);
                if (!dynamics.has_value()) {
                    error_msg();
                    break;
                }
                params_copy = params_copy + std::get<1>(dynamics.value()) * delta;
                momentum = momentum - std::get<0>(dynamics.value()) * delta;

                foliation = ham(params, momentum);
                if (!foliation.has_value()) {
                    error_msg();
                    break;
                }

                params_flow.push_back(params);
                momentum_flow.push_back(momentum);
                energy_fluctuation.push_back(std::get<2>(foliation.value()).detach());

                if (iter_step < conf.max_flow_steps - 1) {
                    const auto rho = -torch::relu(energy_fluctuation[iter_step + 1] - energy_fluctuation[0]);
                    if ((rho >= torch::log(torch::rand_like(rho))).item<bool>()) {
                        params_copy = params_copy + std::get<1>(dynamics.value()) * delta;
                        momentum = momentum - std::get<0>(dynamics.value()) * delta;
                    } else {
                        if (conf.verbose)
                            std::cout << "GHMC: rejecting sample at iteration: "
                                      << iter_step + 1 << "/" << conf.max_flow_steps << "\n";
                        break;
                    }
                }
            }
            return HamiltonianFlow{params_flow, momentum_flow, energy_fluctuation};
        };
    }


    template<typename LogProbabilityDensity, typename Configurations>
    inline auto sampler(const LogProbabilityDensity &log_prob_density, const Configurations &conf) {
        const auto ham_flow = hamiltonian_flow(log_prob_density, conf);
        return [ham_flow, conf](const Parameters &initial_parameters, const uint32_t num_iterations) {
            const auto max_num_samples = conf.max_flow_steps * num_iterations;

            auto samples = Samples{};
            samples.reserve(max_num_samples + 1);

            if (conf.verbose)
                std::cout << "GHMC: Riemannian HMC simulation\n"
                          << "GHMC: generating MCMC chain of maximum length "
                          << max_num_samples <<  " ...\n";

            auto params = initial_parameters.detach().clone();
            samples.push_back(params);

            for(uint32_t iter; iter < num_iterations; iter++){
                auto flow = ham_flow(params);
                const auto &params_flow = std::get<0>(flow);
                if(params_flow.size() > 1){
                    samples.insert(samples.end(), params_flow.begin() + 1, params_flow.end());
                    params = params_flow.back().clone();
                }
            }

            if (conf.verbose)
                std::cout << "GHMC: generated "
                          << samples.size() <<  " samples.\n";

            return samples;

        };
    }

    ///////////////////////////////////////////////////
    inline FisherInfo fisher_info(LogProb log_prob, Params params) {
        auto n = params.numel();
        auto res = log_prob.new_zeros({n, n});
        auto grad = torch::autograd::grad({log_prob}, {params}, {}, torch::nullopt, true)[0];
        int i = 0;
        for (int j = 0; j < n; j++) {
            auto row = grad[j].requires_grad()
                       ? torch::autograd::grad({grad[i]}, {params}, {}, true, true, true)[0].slice(0, j, n)
                       : grad[j].new_zeros(n - j, torch::device(params.device()));
            res[i].slice(0, i, n).add_(row.type_as(res));
            i++;
        }
        auto check = torch::triu(res).sum();
        return (torch::isnan(check).item<bool>() || torch::isinf(check).item<bool>())
               ? FisherInfo{}
               : FisherInfo{-(res + torch::triu(res, 1).t())};
    }

    inline SoftAbsMap softabs_map(LogProb log_prob, Params params, double jitter = 0.001, double softabs_const = 1e6) {
        auto fisher_ = fisher_info(log_prob, params);
        if (!fisher_.has_value())
            return SoftAbsMap{};
        auto fisher = fisher_.value();
        auto n = params.numel();
        fisher += (torch::eye(n, torch::device(params.device())) *
                   torch::rand(n, torch::device(params.device())) * jitter);
        auto[eigs, rotation] = torch::symeig(fisher, true);
        auto spectrum = torch::abs((1. / torch::tanh(softabs_const * eigs)) * eigs);
        return (torch::isinf(spectrum.sum()).item<bool>() || torch::isinf((1 / spectrum).sum()).item<bool>())
               ? SoftAbsMap{}
               : SoftAbsMap{std::make_tuple(spectrum, rotation)};
    }

    template<typename LogProbabilityDensity>
    HamiltonianRef hamiltonianref(LogProbabilityDensity log_probability_density,
                                  Params params, std::optional<Momentum> momentum_,
                                  double jitter = 0.001, double softabs_const = 1e6) {
        torch::Tensor log_prob = log_probability_density(params);
        if (torch::isnan(log_prob).item<bool>() || torch::isinf(log_prob).item<bool>())
            return HamiltonianRef{};
        auto metric = softabs_map(log_prob, params, jitter, softabs_const);
        if (!metric.has_value())
            return HamiltonianRef{};
        auto[spectrum, rotation] = metric.value();
        auto momentum = momentum_.has_value()
                        ? momentum_.value()
                        : rotation.mv(
                        torch::sqrt(spectrum) * torch::randn(params.numel(), torch::device(params.device())));
        auto first_order_term = 0.5 * spectrum.log().sum();
        auto mass = rotation.mm(torch::diag(1 / spectrum)).mm(rotation.t());
        auto second_order_term = 0.5 * momentum.dot(mass.mv(momentum));
        auto energy = -log_prob + first_order_term + second_order_term;
        return HamiltonianRef{
                std::make_tuple(energy, momentum_.has_value() ? std::nullopt : std::make_optional(momentum))};
    }

    template<typename LogProbabilityDensity>
    SymplecticFlow symplectic_flow(LogProbabilityDensity log_probability_density,
                                   Params params, Momentum momentum,
                                   int leap_steps = 10, double epsilon = 0.1, double binding_const = 100,
                                   double jitter = 0.01, int jitter_max = 10, double softabs_const = 1e6) {
        auto ham_grad_params_ = [&log_probability_density, jitter, softabs_const](torch::Tensor params_,
                                                                                  torch::Tensor momentum_) {
            params_ = params_.detach().requires_grad_();
            auto ham_ = hamiltonianref(log_probability_density, params_, momentum_.detach(), jitter, softabs_const);
            if (!ham_.has_value())
                return SafeResult{};
            torch::Tensor ham = std::get<0>(ham_.value());
            auto params_grad = torch::autograd::grad({ham}, {params_})[0];
            auto check = params_grad.sum();
            return (torch::isnan(check).item<bool>() || torch::isinf(check).item<bool>())
                   ? SafeResult{}
                   : SafeResult{params_grad};
        };

        auto ham_grad_params = [&ham_grad_params_, jitter_max](torch::Tensor params_, torch::Tensor momentum_) {
            SafeResult params_grad = ham_grad_params_(params_, momentum_);
            int tries = 0;
            while (!params_grad.has_value()) {
                tries += 1;
                if (tries > jitter_max)
                    break;
                params_grad = ham_grad_params_(params_, momentum_);
            }
            return params_grad;
        };

        auto ham_grad_momentum = [&log_probability_density, jitter, softabs_const](torch::Tensor params_,
                                                                                   torch::Tensor momentum_) {
            momentum_ = momentum_.detach().requires_grad_();
            params_ = params_.detach().requires_grad_();
            auto ham_ = hamiltonianref(log_probability_density, params_, momentum_, jitter, softabs_const);
            if (!ham_.has_value())
                return SafeResult{};
            torch::Tensor ham = std::get<0>(ham_.value());
            auto moment_grad = torch::autograd::grad({ham}, {momentum_})[0];
            auto check = moment_grad.sum();
            return (torch::isnan(check).item<bool>() || torch::isinf(check).item<bool>())
                   ? SafeResult{}
                   : SafeResult{moment_grad};
        };

        auto n = params.numel();
        auto ret_params = torch::zeros({leap_steps, n}, torch::dtype(params.dtype()).device(params.device()));
        auto ret_moment = torch::zeros({leap_steps, n}, torch::dtype(params.dtype()).device(params.device()));
        auto params_copy = params.clone();
        auto momentum_copy = momentum.clone();
        auto c = torch::cos(torch::tensor({2. * binding_const * epsilon}, torch::device(params.device())));
        auto s = torch::sin(torch::tensor({2 * binding_const * epsilon}, torch::device(params.device())));
        SafeResult H;
        for (int i = 0; i < leap_steps; i++) {
            // Leapfrog step
            H = ham_grad_params(params, momentum_copy);
            if (!H.has_value())
                return SymplecticFlow{};
            momentum = momentum - 0.5 * epsilon * H.value();

            H = ham_grad_momentum(params, momentum_copy);
            if (!H.has_value())
                return SymplecticFlow{};
            params_copy = params_copy + 0.5 * epsilon * H.value();

            H = ham_grad_momentum(params_copy, momentum);
            if (!H.has_value())
                return SymplecticFlow{};
            params = params + 0.5 * epsilon * H.value();

            H = ham_grad_params(params_copy, momentum);
            if (!H.has_value())
                return SymplecticFlow{};
            momentum_copy = momentum_copy - 0.5 * epsilon * H.value();

            params = 0.5 * (params + params_copy + c * (params - params_copy) + s * (momentum - momentum_copy));
            momentum = 0.5 * (momentum + momentum_copy - s * (params - params_copy) + c * (momentum - momentum_copy));
            params_copy = 0.5 * (params + params_copy - c * (params - params_copy) - s * (momentum - momentum_copy));
            momentum_copy =
                    0.5 * (momentum + momentum_copy + s * (params - params_copy) - c * (momentum - momentum_copy));

            H = ham_grad_momentum(params_copy, momentum);
            if (!H.has_value())
                return SymplecticFlow{};
            params = params + 0.5 * epsilon * H.value();

            H = ham_grad_params(params_copy, momentum);
            if (!H.has_value())
                return SymplecticFlow{};
            momentum_copy = momentum_copy - 0.5 * epsilon * H.value();

            H = ham_grad_params(params, momentum_copy);
            if (!H.has_value())
                return SymplecticFlow{};
            momentum = momentum - 0.5 * epsilon * H.value();

            H = ham_grad_momentum(params, momentum_copy);
            if (!H.has_value())
                return SymplecticFlow{};
            params_copy = params_copy + 0.5 * epsilon * H.value();

            ret_params[i] = params;
            ret_moment[i] = momentum;
        }

        return SymplecticFlow{std::make_tuple(ret_params, ret_moment)};
    }

    struct SampleConf {
        int num_iter = 10;    // need automatic convergence check
        int leap_steps = 10;  // should be determined by NUTS
        double epsilon = 0.1; // step size in symplectic flow integrator
        double binding_const = 100.;
        double jitter = 0.001;
        int jitter_max = 10;
        double softabs_const = 1e6; // SoftAbsMetric

        inline SampleConf &set_num_iter(int num_iter_) {
            num_iter = num_iter_;
            return *this;
        }

        inline SampleConf &set_leap_steps(int leap_steps_) {
            leap_steps = leap_steps_;
            return *this;
        }

        inline SampleConf &set_epsilon(double epsilon_) {
            epsilon = epsilon_;
            return *this;
        }

        inline SampleConf &set_binding_const(double binding_const_) {
            binding_const = binding_const_;
            return *this;
        }

        inline SampleConf &set_jitter(double jitter_) {
            jitter = jitter_;
            return *this;
        }

        inline SampleConf &set_jitter_max(int jitter_max_) {
            jitter_max = jitter_max_;
            return *this;
        }

        inline SampleConf &set_softabs_const(double softabs_const_) {
            softabs_const = softabs_const_;
            return *this;
        }
    };

    template<typename LogProbabilityDensity>
    SampleResult sample(LogProbabilityDensity log_probability_density,
                        Params params_init, // vector of initial values
                        SampleConf conf = SampleConf{}) {
        if (params_init.ndimension() != 1) {
            printf("GHMC error: params_init must be a vector\n");
            return SampleResult{};
        }

        auto chain_length = conf.num_iter * conf.leap_steps;
        auto result = torch::zeros({chain_length, params_init.numel()},
                                   torch::dtype(params_init.dtype()).device(params_init.device()));
        int num_rejected = 0;
        auto params = params_init.clone();

        auto skip_iter = [&result, &num_rejected, &params, conf](int iter) {
            if (iter == 0) {
                printf("GHMC error: bad initial value\n");
                return true;
            }
            num_rejected += 1;
            params = result[(iter - 1) * conf.leap_steps];
            result.slice(0, iter * conf.leap_steps, (iter + 1) * conf.leap_steps) = result.slice(0, (iter - 1) *
                                                                                                    conf.leap_steps,
                                                                                                 iter *
                                                                                                 conf.leap_steps);
            return false;
        };

        printf("GHMC: Riemannian HMC simulation\n");
        printf("GHMC: Generating MCMC chain of length %d ...\n", chain_length);
        auto start = std::chrono::steady_clock::now();
        for (int iter = 0; iter < conf.num_iter; iter++) {
            params = params.detach().requires_grad_();
            auto ham_ = hamiltonianref(log_probability_density, params, std::nullopt, conf.jitter, conf.softabs_const);
            if (!ham_.has_value()) {
                printf("GHMC: Failed to compute hamiltonianref\n");
                if (skip_iter(iter))
                    return SampleResult{};
                continue;
            }
            auto[ham, momentum] = ham_.value();
            auto flow_ = symplectic_flow(log_probability_density, params, momentum.value(),
                                         conf.leap_steps, conf.epsilon, conf.binding_const, conf.jitter,
                                         conf.jitter_max, conf.softabs_const);
            if (!flow_.has_value()) {
                printf("GHMC: Failed to evolve symplectic flow\n");
                if (skip_iter(iter))
                    return SampleResult{};
                continue;
            }
            auto[flow_params, flow_momenta] = flow_.value();

            params = flow_params[-1].detach().requires_grad_();
            auto new_ham = hamiltonianref(log_probability_density, params, flow_momenta[-1], conf.jitter,
                                          conf.softabs_const);
            if (!new_ham.has_value()) {
                printf("GHMC: Failed to compute proposed hamiltonianref\n");
                if (skip_iter(iter))
                    return SampleResult{};
                continue;
            }

            torch::Tensor rho = -torch::relu(std::get<0>(new_ham.value()) - ham).to(torch::kCPU);
            if ((rho >= torch::log(torch::rand(1))).item<bool>()) {
                result.slice(0, iter * conf.leap_steps, (iter + 1) * conf.leap_steps) = flow_params;
            } else {
                printf("GHMC: Rejecting proposed sample\n");
                if (skip_iter(iter))
                    return SampleResult{};
            }
        }
        auto end = std::chrono::steady_clock::now();
        printf("GHMC: Took %.3f sec\n", std::chrono::duration<double>(end - start).count());
        double acceptance_rate = 1. - (num_rejected / (double) conf.num_iter);
        printf("GHMC: Acceptance rate %.2f\n", acceptance_rate);
        return SampleResult{std::make_tuple(acceptance_rate, result)};
    }

} // namespace noa::ghmc