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

#include <optional>

namespace kotik::utils::numerics
{

    // https://pomax.github.io/bezierinfo/legendre-gauss.html
    template <typename Dtype, typename Function>
    inline Dtype legendre_gaussian_quadrature(const Dtype &lower_bound,
                                              const Dtype &upper_bound,
                                              const Function &function,
                                              const int min_points,
                                              const int order,
                                              const Dtype *abscissa,
                                              const Dtype *weight)
    {
        const int n_itv = (min_points + order - 1) / order;
        const Dtype h = (upper_bound - lower_bound) / n_itv;
        const int N = n_itv * order;
        Dtype res = 0;
        Dtype x0 = lower_bound;
        for (int i = 0; i < N; i++)
        {
            int j = i % order;
            res += function(x0 + h * ((i / order) + abscissa[j])) * h * weight[j];
        }
        return res;
    }

    template <typename Dtype, typename Function>
    inline Dtype quadrature6(const Dtype &lower_bound,
                             const Dtype &upper_bound,
                             const Function &function,
                             const int min_points = 1)
    {
        constexpr int N_GQ = 6;
        const Dtype xGQ[N_GQ] = {0.03376524, 0.16939531, 0.38069041,
                                 0.61930959, 0.83060469, 0.96623476};
        const Dtype wGQ[N_GQ] = {0.08566225, 0.18038079, 0.23395697,
                                 0.23395697, 0.18038079, 0.08566225};

        return legendre_gaussian_quadrature<Dtype>(
            lower_bound,
            upper_bound,
            function,
            min_points,
            N_GQ, xGQ, wGQ);
    }

    template <typename Dtype, typename Function>
    inline Dtype quadrature8(const Dtype &lower_bound,
                             const Dtype &upper_bound,
                             const Function &function,
                             const int min_points = 1)
    {
        constexpr int N_GQ = 8;
        const Dtype xGQ[N_GQ] = {0.01985507, 0.10166676, 0.2372338,
                                 0.40828268, 0.59171732, 0.7627662, 0.89833324, 0.98014493};
        const Dtype wGQ[N_GQ] = {0.05061427, 0.11119052, 0.15685332,
                                 0.18134189, 0.18134189, 0.15685332, 0.11119052, 0.05061427};

        return legendre_gaussian_quadrature(
            lower_bound,
            upper_bound,
            function,
            min_points,
            N_GQ, xGQ, wGQ);
    }

    template <typename Dtype, typename Function>
    inline Dtype quadrature9(const Dtype &lower_bound,
                             const Dtype &upper_bound,
                             const Function &function,
                             const int min_points = 1)
    {
        constexpr int N_GQ = 9;
        const Dtype xGQ[N_GQ] = {0.0000000000000000, -0.8360311073266358,
                                 0.8360311073266358, -0.9681602395076261, 0.9681602395076261,
                                 -0.3242534234038089, 0.3242534234038089, -0.6133714327005904,
                                 0.6133714327005904};
        const Dtype wGQ[N_GQ] = {0.3302393550012598, 0.1806481606948574,
                                 0.1806481606948574, 0.0812743883615744, 0.0812743883615744,
                                 0.3123470770400029, 0.3123470770400029, 0.2606106964029354,
                                 0.2606106964029354};

        return legendre_gaussian_quadrature(
            lower_bound,
            upper_bound,
            function,
            min_points,
            N_GQ, xGQ, wGQ);
    }

    //https://en.wikipedia.org/wiki/Ridders%27_method
    template <typename Dtype, typename Function>
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
        const int max_iter = 100)
    {
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
        for (int i = 0; i < max_iter; i++)
        {
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
            if (fn * fm < 0.0)
            {
                xa = xn;
                fa = fn;
                xb = xm;
                fb = fm;
            }
            else if (fn * fa < 0.0)
            {
                xb = xn;
                fb = fn;
            }
            else
            {
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

} // namespace kotik::utils::numerics