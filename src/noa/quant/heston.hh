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
 * @param size Length of the output tensor.
 * @return Tensor with generated samples. Shape: same as `df` and `nonc`, if
 *     they have the same shape.
 */
torch::Tensor
noncentral_chisquare(const torch::Tensor& df, const torch::Tensor& nonc) {
    // algorithm is summarized in [Andersen2007, section 3.2.4]
    double PSI_CRIT = 1.5;  // threshold value for switching between sampling algorithms
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
    return torch::where(psi <= PSI_CRIT, sample_quad, sample_exp);
}

/**
 * Generates paths of Cox-Ingersoll-Ross (CIR) process.
 *
 * CIR process is described by the SDE:
 *
 * dv(t) = κ·(θ - v(t))·dt + ε·sqrt(v(t))·dW(t)
 *
 * (see [Grzelak2019, section 8.1.2]).
 *
 * For path generation, Andersen's Quadratic Exponential scheme is used
 * (see [Andersen2007], [Grzelak2019, section 9.3.4]).
 *
 * @param n_paths Number of paths to simulate.
 * @param n_steps Number of time steps.
 * @param dt Time step.
 * @param init_state Initial states of the paths, i.e. v(0). Shape: (n_paths,).
 * @param kappa Parameter κ.
 * @param theta Parameter θ.
 * @param eps Parameter ε.
 * @return Simulated paths of CIR process. Shape: (n_paths, n_steps + 1).
 */
torch::Tensor
generate_cir(int64_t n_paths, int64_t n_steps, double dt,
             const torch::Tensor& init_state,
             double kappa, double theta, double eps)
{
    if (init_state.sizes() != torch::IntArrayRef{n_paths})
        throw std::invalid_argument("Shape of `init_state` must be (n_paths,)");

    torch::Tensor paths = torch::empty({n_paths, n_steps + 1},init_state.dtype());
    paths.index_put_({Slice(), 0}, init_state);

    torch::Tensor delta = 4 * kappa * theta / (eps * eps) * torch::ones_like(init_state);
    double exp = std::exp(-kappa*dt);
    double c_bar = 1 / (4*kappa) * eps * eps * (1 - exp);
    for (int64_t i = 0; i < n_steps; i++) {
        torch::Tensor v_cur = paths.index({Slice(), i});
        torch::Tensor kappa_bar = v_cur * 4*kappa*exp / (eps * eps * (1 - exp));
        // [Grzelak2019, definition 8.1.1]
        torch::Tensor v_next = c_bar * noncentral_chisquare(delta, kappa_bar);
        paths.index_put_({Slice(), i+1}, v_next);
    }
    return paths;
}

/**
 * Generates time series following the Heston model:
 *
 * dS(t) = μ·S(t)·dt + sqrt(v(t))·S(t)·dW_1(t);
 * dv(t) = κ(θ - v(t))·dt + ε·sqrt(v(t))·dW_2(t),
 *
 * using Andersen's Quadratic Exponential scheme [Andersen2007].
 * Also see [Grzelak, section 9.4.3].
 *
 * @param n_paths Number of simulated paths.
 * @param n_steps Number of time steps.
 * @param dt Time step.
 * @param init_state_price Initial states of the price paths, i.e. S(0). Shape: (n_paths).
 * @param init_state_var Initial states of the variance paths, i.e. v(0). Shape: (n_paths).
 * @param kappa Parameter κ - the rate at which v(t) reverts to θ.
 * @param theta Parameter θ - long-run average variance.
 * @param eps Parameter ε - volatility of variance.
 * @param rho Correlation between underlying Brownian motions for S(t) and v(t).
 * @param drift Drift parameter μ.
 * @return Two tensors: simulated paths for price, simulated paths for variance.
 *     Both tensors have shape (n_paths, n_steps + 1).
 */
std::tuple<torch::Tensor, torch::Tensor>
generate_heston(int64_t n_paths, int64_t n_steps, double dt,
                const torch::Tensor& init_state_price,
                const torch::Tensor& init_state_var,
                double kappa, double theta, double eps, double rho, double drift)
{
    if (init_state_price.sizes() != torch::IntArrayRef{n_paths})
        throw std::invalid_argument("Shape of `init_state_price` must be (n_paths,)");
    if (init_state_var.sizes() != torch::IntArrayRef{n_paths})
        throw std::invalid_argument("Shape of `init_state_var` must be (n_paths,)");

    double gamma2 = 0.5;
    // regularity condition [Andersen 2007, section 4.3.2]
    if (rho > 0) { // always satisfied when rho <= 0
        double L = rho*dt*(kappa/eps - 0.5*rho);
        double R = 2*kappa/(eps*eps*(1 - std::exp(-kappa*dt))) - rho/eps;
        if (R<=0 || L==0 || (L<0 && R>=0)) {
            // When (L<0 && R<=0), L/R is always < 0.5.
            // (L>0 && R<=0) never happens.
            // In other cases, regularity condition is always satisfied.
        }
        else if (L > 0) {
            gamma2 = std::min(0.5, R / L * 0.9); // multiply by 0.9 to have some margin
        }
    }
    double gamma1 = 1.0 - gamma2;

    double k0 = -rho * kappa * theta * dt / eps;
    double k1 = gamma1 * dt * (kappa * rho / eps - 0.5) - rho / eps;
    double k2 = gamma2 * dt * (kappa * rho / eps - 0.5) + rho / eps;
    double k3 = gamma1 * dt * (1 - rho * rho);
    double k4 = gamma2 * dt * (1 - rho * rho);

    torch::Tensor var = generate_cir(n_paths, n_steps, dt, init_state_var, kappa, theta, eps);
    torch::Tensor log_paths = torch::empty({n_paths, n_steps + 1}, init_state_price.dtype());
    log_paths.index_put_({Slice(), 0}, init_state_price.log());

    for (int64_t i = 0; i < n_steps; i++) {
        torch::Tensor v_i = var.index({Slice(), i});
        torch::Tensor v_next = var.index({Slice(), i+1});
        torch::Tensor next_vals = drift*dt +
                log_paths.index({Slice(), i}) + k0 + k1*v_i + k2*v_next +
                torch::sqrt(k3*v_i + k4*v_next) * torch::randn_like(v_i);
        log_paths.index_put_({Slice(), i+1}, next_vals);
    }
    return std::make_tuple(log_paths.exp(), var);
}

} // namespace noa::quant
