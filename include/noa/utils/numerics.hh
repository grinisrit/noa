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

#include "noa/utils/common.hh"

#include <optional>

namespace noa::utils::numerics {

    inline TensorOpt hessian(const torch::Tensor &function_value, const torch::Tensor &parameter) {
        if ((function_value.dim() > 0) || (parameter.dim() > 1)) {
            std::cerr << "Invalid arguments to noa::utils::numerics::hessian : "
                      << "expecting 0D function value and 1D parameter\n";
            return std::nullopt;
        }
        auto n = parameter.numel();
        auto res = function_value.new_zeros({n, n});
        auto grad = torch::autograd::grad({function_value}, {parameter}, {}, torch::nullopt, true)[0];
        int i = 0;
        for (int j = 0; j < n; j++) {
            auto row = grad[j].requires_grad()
                       ? torch::autograd::grad({grad[i]}, {parameter}, {}, true, true, true)[0].slice(0, j, n)
                       : grad[j].new_zeros(n - j);
            res[i].slice(0, i, n).add_(row);
            i++;
        }
        auto check = torch::triu(res.detach()).sum();
        return (torch::isnan(check).item<bool>() || torch::isinf(check).item<bool>())
               ? std::nullopt
               : std::make_optional(res + torch::triu(res, 1).t());
    }

    // https://pomax.github.io/bezierinfo/legendre-gauss.html
    template<typename Dtype, typename Function>
    inline Dtype legendre_gaussian_quadrature(const Dtype &lower_bound,
                                              const Dtype &upper_bound,
                                              const Function &function,
                                              const int32_t min_points,
                                              const int32_t order,
                                              const Dtype *abscissa,
                                              const Dtype *weight) {
        const int32_t n_itv = (min_points + order - 1) / order;
        const Dtype h = (upper_bound - lower_bound) / n_itv;
        const int32_t N = n_itv * order;
        Dtype res = 0;
        Dtype x0 = lower_bound;
        for (int32_t i = 0; i < N; i++) {
            int32_t j = i % order;
            res += function(x0 + h * ((i / order) + abscissa[j])) * h * weight[j];
        }
        return res;
    }

    template<typename Dtype, typename Function>
    inline Dtype quadrature6(const Dtype &lower_bound,
                             const Dtype &upper_bound,
                             const Function &function,
                             const int32_t min_points = 1) {
        constexpr int32_t N_GQ = 6;
        const Dtype xGQ[N_GQ] = {(Dtype) 0.03376524, (Dtype) 0.16939531, (Dtype) 0.38069041,
                                 (Dtype) 0.61930959, (Dtype) 0.83060469, (Dtype) 0.96623476};
        const Dtype wGQ[N_GQ] = {(Dtype) 0.08566225, (Dtype) 0.18038079, (Dtype) 0.23395697,
                                 (Dtype) 0.23395697, (Dtype) 0.18038079, (Dtype) 0.08566225};

        return legendre_gaussian_quadrature<Dtype>(
                lower_bound,
                upper_bound,
                function,
                min_points,
                N_GQ, xGQ, wGQ);
    }

    template<typename Dtype, typename Function>
    inline Dtype quadrature8(const Dtype &lower_bound,
                             const Dtype &upper_bound,
                             const Function &function,
                             const int32_t min_points = 1) {
        constexpr int32_t N_GQ = 8;
        const Dtype xGQ[N_GQ] = {(Dtype) 0.01985507, (Dtype) 0.10166676, (Dtype) 0.2372338,
                                 (Dtype) 0.40828268, (Dtype) 0.59171732, (Dtype) 0.7627662,
                                 (Dtype) 0.89833324, (Dtype) 0.98014493};
        const Dtype wGQ[N_GQ] = {(Dtype) 0.05061427, (Dtype) 0.11119052, (Dtype) 0.15685332,
                                 (Dtype) 0.18134189, (Dtype) 0.18134189, (Dtype) 0.15685332,
                                 (Dtype) 0.11119052, (Dtype) 0.05061427};

        return legendre_gaussian_quadrature(
                lower_bound,
                upper_bound,
                function,
                min_points,
                N_GQ, xGQ, wGQ);
    }

    template<typename Dtype, typename Function>
    inline Dtype quadrature9(const Dtype &lower_bound,
                             const Dtype &upper_bound,
                             const Function &function,
                             const int32_t min_points = 1) {
        constexpr int32_t N_GQ = 9;
        const Dtype xGQ[N_GQ] = {(Dtype) 0.0000000000000000, (Dtype) -0.8360311073266358,
                                 (Dtype) 0.8360311073266358, (Dtype) -0.9681602395076261, (Dtype) 0.9681602395076261,
                                 (Dtype) -0.3242534234038089, (Dtype) 0.3242534234038089, (Dtype) -0.6133714327005904,
                                 (Dtype) 0.6133714327005904};
        const Dtype wGQ[N_GQ] = {(Dtype) 0.3302393550012598, (Dtype) 0.1806481606948574,
                                 (Dtype) 0.1806481606948574, (Dtype) 0.0812743883615744, (Dtype) 0.0812743883615744,
                                 (Dtype) 0.3123470770400029, (Dtype) 0.3123470770400029, (Dtype) 0.2606106964029354,
                                 (Dtype) 0.2606106964029354};

        return legendre_gaussian_quadrature(
                lower_bound,
                upper_bound,
                function,
                min_points,
                N_GQ, xGQ, wGQ);
    }

    //https://en.wikipedia.org/wiki/Ridders%27_method
    template<typename Dtype, typename Function>
    inline std::optional<Dtype> ridders_root(
            Dtype xa, //The lower bound of the search interval.
            Dtype xb, //The upper bound of the search interval.
            //The objective function to resolve.
            const Function &function,
            // The initial value at *a*,
            const std::optional<Dtype> &fa_ = std::nullopt,
            // The initial value at *b*
            const std::optional<Dtype> &fb_ = std::nullopt,
            //The absolute & relative tolerance on the root value.
            const Dtype &xtol = TOLERANCE,
            const Dtype &rtol = TOLERANCE,
            const int32_t max_iter = 100) {
        //  Check the initial values
        Dtype fa = (fa_.has_value()) ? fa_.value() : function(xa);
        Dtype fb = (fb_.has_value()) ? fb_.value() : function(xb);

        if (fa * fb > 0)
            return std::nullopt;
        if (fa == 0)
            return xa;
        if (fb == 0)
            return xb;

        // Set the tolerance for the root finding
        const Dtype tol =
                xtol + rtol * std::min(std::abs(xa), std::abs(xb));

        // Do the bracketing using Ridder's update rule.
        Dtype xn = 0.;
        for (int32_t i = 0; i < max_iter; i++) {
            Dtype dm = 0.5 * (xb - xa);
            const Dtype xm = xa + dm;
            const Dtype fm = function(xm);
            Dtype sgn = (fb > fa) ? 1. : -1.;
            Dtype dn = sgn * dm * fm / sqrt(fm * fm - fa * fb);
            sgn = (dn > 0.) ? 1. : -1.;
            dn = std::abs(dn);
            dm = std::abs(dm) - 0.5 * tol;
            if (dn < dm)
                dm = dn;
            xn = xm - sgn * dm;
            const Dtype fn = function(xn);
            if (fn * fm < 0.0) {
                xa = xn;
                fa = fn;
                xb = xm;
                fb = fm;
            } else if (fn * fa < 0.0) {
                xb = xn;
                fb = fn;
            } else {
                xa = xn;
                fa = fn;
            }
            if (fn == 0.0 || std::abs(xb - xa) < tol)
                // A valid bracketing was found
                return xn;
        }

        /* The maximum number of iterations was reached*/
        return std::nullopt;
    }

} // namespace noa::utils::numerics