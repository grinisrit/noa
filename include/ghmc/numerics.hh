/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Roland Grinis, GrinisRIT ltd.
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

namespace ghmc::numerics
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

    template <typename Function>
    inline float quadrature_f6(const float &lower_bound,
                               const float &upper_bound,
                               const Function &function,
                               const int min_points = 1)
    {
        constexpr int N_GQ = 6;
        const float xGQ[N_GQ] = {0.03376524f, 0.16939531f, 0.38069041f,
                                  0.61930959f, 0.83060469f, 0.96623476f};
        const float wGQ[N_GQ] = {0.08566225f, 0.18038079f, 0.23395697f,
                                  0.23395697f, 0.18038079f, 0.08566225f};

        return legendre_gaussian_quadrature(
            lower_bound,
            upper_bound,
            function,
            min_points,
            N_GQ, xGQ, wGQ);
    }

    template <typename Function>
    inline float quadrature_f8(const float &lower_bound,
                               const float &upper_bound,
                               const Function &function,
                               const int min_points = 1)
    {
        constexpr int N_GQ = 8;
        const float xGQ[N_GQ] = {0.01985507f, 0.10166676f, 0.2372338f,
                                 0.40828268f, 0.59171732f, 0.7627662f, 0.89833324f, 0.98014493f};
        const float wGQ[N_GQ] = {0.05061427f, 0.11119052f, 0.15685332f,
                                 0.18134189f, 0.18134189f, 0.15685332f, 0.11119052f, 0.05061427f};

        return legendre_gaussian_quadrature(
            lower_bound,
            upper_bound,
            function,
            min_points,
            N_GQ, xGQ, wGQ);
    }

    template <typename Function>
    inline float quadrature_f9(const float &lower_bound,
                               const float &upper_bound,
                               const Function &function,
                               const int min_points = 1)
    {
        constexpr int N_GQ = 9;
        const float xGQ[N_GQ] = {0.0000000000000000f, -0.8360311073266358f,
                                 0.8360311073266358f, -0.9681602395076261f, 0.9681602395076261f,
                                 -0.3242534234038089f, 0.3242534234038089f, -0.6133714327005904f,
                                 0.6133714327005904f};
        const float wGQ[N_GQ] = {0.3302393550012598f, 0.1806481606948574f,
                                 0.1806481606948574f, 0.0812743883615744f, 0.0812743883615744f,
                                 0.3123470770400029f, 0.3123470770400029f, 0.2606106964029354f,
                                 0.2606106964029354f};

        return legendre_gaussian_quadrature(
            lower_bound,
            upper_bound,
            function,
            min_points,
            N_GQ, xGQ, wGQ);
    }
} // namespace ghmc::numerics