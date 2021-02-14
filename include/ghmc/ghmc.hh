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

#include <iostream>
#include <chrono>

#include <torch/torch.h>

namespace ghmc
{
    using SafeResult = std::optional<torch::Tensor>;
    using SampleResult = std::optional<std::tuple<double, torch::Tensor>>;
    using Params = torch::Tensor;
    using Momentum = torch::Tensor;
    using LogProb = torch::Tensor;
    using FisherInfo = std::optional<torch::Tensor>;
    using SoftAbsMap = std::optional<std::tuple<torch::Tensor, torch::Tensor>>;
    using Hamiltonian = std::optional<std::tuple<torch::Tensor, std::optional<Momentum>>>;
    using SymplecticFlow = std::optional<std::tuple<torch::Tensor, torch::Tensor>>;

    inline FisherInfo fisher_info(LogProb log_prob, Params params)
    {
        auto n = params.numel();
        auto res = log_prob.new_zeros({n, n});
        auto grad = torch::autograd::grad({log_prob}, {params}, {}, torch::nullopt, true)[0];
        int i = 0;
        for (int j = 0; j < n; j++)
        {
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

    inline SoftAbsMap softabs_map(LogProb log_prob, Params params, double jitter = 0.001, double softabs_const = 1e6)
    {
        auto fisher_ = fisher_info(log_prob, params);
        if (!fisher_.has_value())
            return SoftAbsMap{};
        auto fisher = fisher_.value();
        auto n = params.numel();
        fisher += (torch::eye(n, torch::device(params.device())) *
                   torch::rand(n, torch::device(params.device())) * jitter);
        auto [eigs, rotation] = torch::symeig(fisher, true);
        auto spectrum = torch::abs((1. / torch::tanh(softabs_const * eigs)) * eigs);
        return (torch::isinf(spectrum.sum()).item<bool>() || torch::isinf((1 / spectrum).sum()).item<bool>())
                   ? SoftAbsMap{}
                   : SoftAbsMap{std::make_tuple(spectrum, rotation)};
    }

    template <typename LogProbabilityDensity>
    Hamiltonian hamiltonian(LogProbabilityDensity log_probability_density,
                            Params params, std::optional<Momentum> momentum_,
                            double jitter = 0.001, double softabs_const = 1e6)
    {
        torch::Tensor log_prob = log_probability_density(params);
        if (torch::isnan(log_prob).item<bool>() || torch::isinf(log_prob).item<bool>())
            return Hamiltonian{};
        auto metric = softabs_map(log_prob, params, jitter, softabs_const);
        if (!metric.has_value())
            return Hamiltonian{};
        auto [spectrum, rotation] = metric.value();
        auto momentum = momentum_.has_value()
                            ? momentum_.value()
                            : rotation.mv(torch::sqrt(spectrum) * torch::randn(params.numel(), torch::device(params.device())));
        auto first_order_term = 0.5 * spectrum.log().sum();
        auto mass = rotation.mm(torch::diag(1 / spectrum)).mm(rotation.t());
        auto second_order_term = 0.5 * momentum.dot(mass.mv(momentum));
        auto energy = -log_prob + first_order_term + second_order_term;
        return Hamiltonian{std::make_tuple(energy, momentum_.has_value() ? std::nullopt : std::make_optional(momentum))};
    }

    template <typename LogProbabilityDensity>
    SymplecticFlow symplectic_flow(LogProbabilityDensity log_probability_density,
                                   Params params, Momentum momentum,
                                   int leap_steps = 10, double epsilon = 0.1, double binding_const = 100,
                                   double jitter = 0.01, int jitter_max = 10, double softabs_const = 1e6)
    {
        auto ham_grad_params_ = [&log_probability_density, jitter, softabs_const](torch::Tensor params_, torch::Tensor momentum_) {
            params_ = params_.detach().requires_grad_();
            auto ham_ = hamiltonian(log_probability_density, params_, momentum_.detach(), jitter, softabs_const);
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
            while (!params_grad.has_value())
            {
                tries += 1;
                if (tries > jitter_max)
                    break;
                params_grad = ham_grad_params_(params_, momentum_);
            }
            return params_grad;
        };

        auto ham_grad_momentum = [&log_probability_density, jitter, softabs_const](torch::Tensor params_, torch::Tensor momentum_) {
            momentum_ = momentum_.detach().requires_grad_();
            params_ = params_.detach().requires_grad_();
            auto ham_ = hamiltonian(log_probability_density, params_, momentum_, jitter, softabs_const);
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
        for (int i = 0; i < leap_steps; i++)
        {
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
            momentum_copy = 0.5 * (momentum + momentum_copy + s * (params - params_copy) - c * (momentum - momentum_copy));

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

    struct SampleConf
    {
        int num_iter = 10;    // need automatic convergence check
        int leap_steps = 10;  // should be determined by NUTS
        double epsilon = 0.1; // step size in symplectic flow integrator
        double binding_const = 100.;
        double jitter = 0.001;
        int jitter_max = 10;
        double softabs_const = 1e6; // SoftAbsMetric

        inline SampleConf &set_num_iter(int num_iter_)
        {
            num_iter = num_iter_;
            return *this;
        }
        inline SampleConf &set_leap_steps(int leap_steps_)
        {
            leap_steps = leap_steps_;
            return *this;
        }
        inline SampleConf &set_epsilon(double epsilon_)
        {
            epsilon = epsilon_;
            return *this;
        }
        inline SampleConf &set_binding_const(double binding_const_)
        {
            binding_const = binding_const_;
            return *this;
        }
        inline SampleConf &set_jitter(double jitter_)
        {
            jitter = jitter_;
            return *this;
        }
        inline SampleConf &set_jitter_max(int jitter_max_)
        {
            jitter_max = jitter_max_;
            return *this;
        }
        inline SampleConf &set_softabs_const(double softabs_const_)
        {
            softabs_const = softabs_const_;
            return *this;
        }
    };

    template <typename LogProbabilityDensity>
    SampleResult sample(LogProbabilityDensity log_probability_density,
                        Params params_init, // vector of initial values
                        SampleConf conf = SampleConf{})
    {
        if (params_init.ndimension() != 1)
        {
            printf("GHMC error: params_init must be a vector\n");
            return SampleResult{};
        }

        auto chain_length = conf.num_iter * conf.leap_steps;
        auto result = torch::zeros({chain_length, params_init.numel()}, torch::dtype(params_init.dtype()).device(params_init.device()));
        int num_rejected = 0;
        auto params = params_init.clone();

        auto skip_iter = [&result, &num_rejected, &params, conf](int iter) {
            if (iter == 0)
            {
                printf("GHMC error: bad initial value\n");
                return true;
            }
            num_rejected += 1;
            params = result[(iter - 1) * conf.leap_steps];
            result.slice(0, iter * conf.leap_steps, (iter + 1) * conf.leap_steps) = result.slice(0, (iter - 1) * conf.leap_steps, iter * conf.leap_steps);
            return false;
        };

        printf("GHMC: Riemannian HMC simulation\n");
        printf("GHMC: Generating MCMC chain of length %d ...\n", chain_length);
        auto start = std::chrono::steady_clock::now();
        for (int iter = 0; iter < conf.num_iter; iter++)
        {
            params = params.detach().requires_grad_();
            auto ham_ = hamiltonian(log_probability_density, params, std::nullopt, conf.jitter, conf.softabs_const);
            if (!ham_.has_value())
            {
                printf("GHMC: Failed to compute hamiltonian\n");
                if (skip_iter(iter))
                    return SampleResult{};
                continue;
            }
            auto [ham, momentum] = ham_.value();
            auto flow_ = symplectic_flow(log_probability_density, params, momentum.value(),
                                         conf.leap_steps, conf.epsilon, conf.binding_const, conf.jitter, conf.jitter_max, conf.softabs_const);
            if (!flow_.has_value())
            {
                printf("GHMC: Failed to evolve symplectic flow\n");
                if (skip_iter(iter))
                    return SampleResult{};
                continue;
            }
            auto [flow_params, flow_momenta] = flow_.value();

            params = flow_params[-1].detach().requires_grad_();
            auto new_ham = hamiltonian(log_probability_density, params, flow_momenta[-1], conf.jitter, conf.softabs_const);
            if (!new_ham.has_value())
            {
                printf("GHMC: Failed to compute proposed hamiltonian\n");
                if (skip_iter(iter))
                    return SampleResult{};
                continue;
            }

            torch::Tensor rho = -torch::relu(std::get<0>(new_ham.value()) - ham).to(torch::kCPU);
            if ((rho >= torch::log(torch::rand(1))).item<bool>())
            {
                result.slice(0, iter * conf.leap_steps, (iter + 1) * conf.leap_steps) = flow_params;
            }
            else
            {
                printf("GHMC: Rejecting proposed sample\n");
                if (skip_iter(iter))
                    return SampleResult{};
            }
        }
        auto end = std::chrono::steady_clock::now();
        printf("GHMC: Took %.3f sec\n", std::chrono::duration<double>(end - start).count());
        double acceptance_rate = 1. - (num_rejected / (double)conf.num_iter);
        printf("GHMC: Acceptance rate %.2f\n", acceptance_rate);
        return SampleResult{std::make_tuple(acceptance_rate, result)};
    }

} // namespace ghmc