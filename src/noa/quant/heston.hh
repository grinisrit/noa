/*****************************************************************************
 *   Copyright (c) 2023, Roland Grinis, GrinisRIT ltd.                       *
 *   (roland.grinis@grinisrit.com)                                           *
 *   All rights reserved.                                                    *
 *   See the file COPYING for full copying permissions.                      *
 *                                                                           *
 *   This program is free software: you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation, either version 3 of the License, or       *
 *   (at your option) any later version.                                     *
 *                                                                           *
 *   This program is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the            *
 *   GNU General Public License for more details.                            *
 *                                                                           *
 *   You should have received a copy of the GNU General Public License       *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.   *
 *****************************************************************************/
/**
 * Implemented by: Maksim Sosnin
 *
 * References:
 *     - [Grzelak2019] Oosterlee, C. W., & Grzelak, L. A. (2019). Mathematical
 *       modeling and computation in finance: with exercises and Python and
 *       MATLAB compute codes. World Scientific.
 *
 *     - [Andersen2007] Andersen, L.B., 2007. Efficient simulation of the Heston
 *       stochastic volatility model. Available at SSRN 946405.
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <tuple>
#include <stdexcept>

#include <torch/torch.h>

namespace noa::quant {

using namespace torch::indexing;

/**
 * Generates samples from a noncentral chi-square distribution using approximate
 * Quadratic Exponential scheme from [Andersen2007].
 *
 * @param df Degrees of freedom, must be > 0.
 * @param nonc Non-centrality parameter, must be >= 0.
 * @param size Length of output tensor.
 * @return Tensor with generated samples. Shape: same as `df` and `nonc`, if
 *     they have the same shape.
 */
torch::Tensor
noncentral_chisquare(const torch::Tensor& df, const torch::Tensor& nonc) {
    // algorithm is summarized in [Andersen2007, section 3.2.4]
    double THRESHOLD = 1.5;  // threshold value for switching between sampling algorithms
    torch::Tensor m = df + nonc;
    torch::Tensor s2 = 2*df + 4*nonc;
    torch::Tensor psi = s2 / m.pow(2);
    // quadratic
    torch::Tensor psi_inv = 1 / psi;
    torch::Tensor b2 = 2*psi_inv - 1 + (2*psi_inv).sqrt() * (2*psi_inv - 1).sqrt();
    torch::Tensor a = m / (1 + b2);
    torch::Tensor sample_quad = a * (b2.sqrt() + torch::randn_like(a)).pow(2);
    // exponential
    torch::Tensor p = (psi - 1) / (psi + 1);
    torch::Tensor beta = (1 - p) / m;
    torch::Tensor rand = torch::rand_like(p);
    torch::Tensor sample_exp = torch::where(
            (p < rand) & (rand <= 1),
            beta.pow(-1)*torch::log((1-p)/(1-rand)),
            torch::zeros_like(rand));
    return torch::where(psi <= THRESHOLD, sample_quad, sample_exp);
}

/**
 * Generates paths of Cox-Ingersoll-Ross (CIR) process.
 *
 * CIR process is described by the SDE:
 *
 * dv(t) = κ(θ - v(t)) dt + ξ sqrt(v(t)) dW(t)
 *
 * (see [Grzelak2019, section 8.1.2]).
 *
 * For path generation, Andersen's Quadratic Exponential scheme is used
 * (see [Andersen2007], [Grzelak2019, section 9.3.4]).
 *
 * @param n_paths Number of simulated paths.
 * @param n_steps Number of time steps.
 * @param dt Time step.
 * @param init_state Initial states of the paths, i.e. v(0). Shape: (n_paths,).
 * @param kappa Parameter κ.
 * @param theta Parameter θ.
 * @param xi Parameter ξ.
 * @return Simulated paths of CIR process. Shape: (n_paths, n_steps + 1).
 */
torch::Tensor
generate_cir(int64_t n_paths, int64_t n_steps, double dt,
             const torch::Tensor& init_state,
             double kappa, double theta, double xi)
{
    if (init_state.sizes() != torch::IntArrayRef{n_paths})
        throw std::invalid_argument("Shape of `init_state` must be (n_paths,)");

    torch::Tensor paths = torch::empty({n_paths, n_steps + 1},init_state.dtype());
    paths.index_put_({Slice(), 0}, init_state);

    torch::Tensor delta = 4*kappa*theta/(xi*xi) * torch::ones_like(init_state);
    double exp = std::exp(-kappa*dt);
    double c_bar = 1/(4*kappa)*xi*xi * (1 - exp);
    for (int64_t i = 0; i < n_steps; i++) {
        torch::Tensor v_cur = paths.index({Slice(), i});
        torch::Tensor kappa_bar = v_cur * 4*kappa*exp / (xi*xi*(1 - exp));
        // [Grzelak2019, definition 8.1.1]
        torch::Tensor v_next = c_bar * noncentral_chisquare(delta, kappa_bar);
        paths.index_put_({Slice(), i+1}, v_next);
    }
    return paths;
}

/**
 * Generates time series following Heston model:
 *
 * dS(t) = sqrt(v(t))·S(t)·dW_1(t);
 * dv(t) = κ(θ - v(t)) dt + ξ sqrt(v(t)) dW_2(t),
 *
 * using Andersen's Quadratic Exponential scheme [Andersen2007].
 * Also see [Grzelak, section 9.4.3].
 *
 * @param n_paths Number of simulated paths.
 * @param n_steps Number of time steps.
 * @param dt Time step.
 * @param init_state_price Initial states of the price paths, i.e. S(0). Shape: (n_paths).
 * @param init_state_var Initial states of the variance paths, i.e. v(0). Shape: (n_paths).
 * @param kappa Parameter κ.
 * @param theta Parameter θ.
 * @param xi Parameter ξ.
 * @param rho Correlation between underlying Brownian motions for S(t) and v(t).
 * @return Two tensors: simulated paths for price, simulated paths for variance.
 *     Both tensors have shape (n_paths, n_steps + 1).
 */
std::tuple<torch::Tensor, torch::Tensor>
generate_heston(int64_t n_paths, int64_t n_steps, double dt,
                const torch::Tensor& init_state_price,
                const torch::Tensor& init_state_var,
                double kappa, double theta, double xi, double rho)
{
    if (init_state_price.sizes() != torch::IntArrayRef{n_paths})
        throw std::invalid_argument("Shape of `init_state_price` must be (n_paths,)");
    if (init_state_var.sizes() != torch::IntArrayRef{n_paths})
        throw std::invalid_argument("Shape of `init_state_var` must be (n_paths,)");

    double k0 = -rho / xi * kappa * theta * dt;
    double k1 = (rho*kappa/xi - 0.5) * dt - rho/xi;
    double k2 = rho / xi;
    double k3 = (1 - rho*rho) * dt;

    torch::Tensor var = generate_cir(n_paths, n_steps, dt, init_state_var, kappa, theta, xi);
    torch::Tensor log_paths = torch::empty({n_paths, n_steps + 1}, init_state_price.dtype());
    log_paths.index_put_({Slice(), 0}, init_state_price.log());

    for (int64_t i = 0; i < n_steps; i++) {
        torch::Tensor v_i = var.index({Slice(), i});
        torch::Tensor next_vals = log_paths.index({Slice(), i}) + k0 +
                k1*v_i + k2*var.index({Slice(), i+1}) +
                torch::sqrt(k3*v_i) * torch::randn_like(v_i);
        log_paths.index_put_({Slice(), i+1}, next_vals);
    }
    return std::make_tuple(log_paths.exp(), var);
}

} // namespace noa::quant
