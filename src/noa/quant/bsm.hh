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
 * Implemented by: Maksim Sosnin (based on numba implementation by Ivan Novikov)
 *
 * References:
 *     - [BrennanSchwartz1977] Brennan, M. J., Schwartz, E. S. (1977).
 *       The valuation of American put options. The Journal of Finance, 32(2), 449-462.
 *
 *     - [Seydel2017] Seydel, R.U. Tools for Computational Finance,
 *       6th edition, Springer V., London (2017).
 */

#pragma once

#include <iostream>
#include <tuple>
#include <vector>
#include <cmath>
#include <cstdint>

#include <torch/torch.h>

namespace noa::quant {

using Slice = torch::indexing::Slice;

    namespace bsm_impl {

    /**
     * Computes solution to Ax - b >= 0 ; x >= g and (Ax-b)'(x-g)=0.
     * A is tridiagonal matrix with alpha, beta, gamma coefficients.
     *
     * @param alpha Main diagonal of A.  Shape: (npoints_S,)
     * @param beta  Upper diagonal of A. Shape: (npoints_S - 1,)
     * @param gamma Lower diagonal of A. Shape: (npoints_S - 1,)
     */
    torch::Tensor
    brennan_schwartz(const torch::Tensor& alpha, const torch::Tensor& beta,
                     const torch::Tensor& gamma, const torch::Tensor& b,
                     const torch::Tensor& g) {
        int64_t n = alpha.sizes()[0];
        torch::Tensor alpha_hat = torch::zeros(n, torch::kFloat64);
        torch::Tensor b_hat = torch::zeros(n, torch::kFloat64);
        torch::Tensor x = torch::zeros(n, torch::kFloat64);

        auto alpha_a = alpha.accessor<double, 1>();
        auto beta_a = beta.accessor<double, 1>();
        auto gamma_a = gamma.accessor<double, 1>();
        auto b_a = b.accessor<double, 1>();
        auto g_a = g.accessor<double, 1>();
        auto alpha_hat_a = alpha_hat.accessor<double, 1>();
        auto b_hat_a = b_hat.accessor<double, 1>();
        auto x_a = x.accessor<double, 1>();

        alpha_hat_a[n-1] = alpha_a[n-1];
        b_hat_a[n-1] = b_a[n-1];
        for (int64_t i = n - 2; i >= 0; i--) {
            alpha_hat_a[i] = alpha_a[i] - beta_a[i] * gamma_a[i] / alpha_hat_a[i+1];
            b_hat_a[i] = b_a[i] - beta_a[i] * b_hat_a[i+1] / alpha_hat_a[i+1];
        }
        x_a[0] = std::max(b_hat_a[0] / alpha_hat_a[0], g_a[0]);
        for (int64_t i = 1; i < n; i++)
            x_a[i] = std::max((b_hat_a[i] - gamma_a[i-1] * x_a[i-1]) / alpha_hat_a[i], g_a[i]);
        return x;
    }


    /**
     * Computes diagonals of the A matrix (for implicit step).
     * @return alpha, beta, gamma for `noa::quant::brennan_schwartz()`
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    get_A_diags(int64_t size, double lambda) {
        torch::Tensor alpha = (1 + lambda) * torch::ones(size + 1, torch::kFloat64);
        torch::Tensor beta = -0.5 * lambda * torch::ones(size, torch::kFloat64);
        torch::Tensor gamma = -0.5 * lambda * torch::ones(size, torch::kFloat64);
        return std::make_tuple(alpha, beta, gamma);
    }


    torch::Tensor
    g_func(const torch::Tensor& tau, const torch::Tensor& x, double k) {
        torch::Tensor t1 = torch::exp(0.25 * tau * std::pow(k+1, 2));
        torch::Tensor t2 = torch::maximum(
                torch::zeros_like(x, torch::kFloat64),
                torch::exp(0.5*(k-1)*x) - torch::exp(0.5*(k+1)*x));
        return t1 * t2;
    }

    double g_func(double tau, double x, double k) {
        double t1 = std::exp(0.25 * tau * std::pow(k+1, 2));
        double t2 = std::max(0.0, std::exp(0.5*(k-1)*x) - std::exp(0.5*(k+1)*x));
        return t1 * t2;
    }


    torch::Tensor
    get_crank_B_matrix(int64_t n_points, double lambda) {
        torch::Tensor B = torch::zeros({n_points, n_points}, torch::kFloat64);
        auto B_a = B.accessor<double, 2>();
        for (int64_t i = 0; i < n_points - 1; i++) {
            B_a[i][i] = 1 - lambda;
            B_a[i][i + 1] = lambda / 2;
            B_a[i + 1][i] = lambda / 2;
        }
        B_a[n_points - 1][n_points - 1] = 1 - lambda;
        return B;
    }


    /**
     * @param w_matrix  Shape: (npoints_S, npoints_t)
     * @param x_array   Shape: (npoints_S,)
     * @param tau_array Shape: (npoints_t,)
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    transform(const torch::Tensor& w_matrix, const torch::Tensor& x_array,
              const torch::Tensor& tau_array,
              double K, double T, double r, double sigma) {
        double k = 2 * r / std::pow(sigma, 2);
        double coef = 0.25 * std::pow(k+1, 2);
        torch::Tensor V = torch::zeros_like(w_matrix, torch::kFloat64);

        auto w_matrix_a = w_matrix.accessor<double, 2>();
        auto x_array_a = x_array.accessor<double, 1>();
        auto tau_array_a = tau_array.accessor<double, 1>();
        auto V_a = V.accessor<double, 2>();

        for (int64_t n = 0; n < tau_array.sizes()[0]; n++)
            for (int64_t m = 0; m < x_array.sizes()[0]; m++)
                V_a[m][n] = K * w_matrix_a[m][n] *
                            std::exp(0.5 * (1-k) * x_array_a[m] - coef * tau_array_a[n]);
        torch::Tensor t_array = T - 2 * tau_array / std::pow(sigma, 2);
        t_array.index_put_({-1}, 0.0);
        torch::Tensor S_array = K * torch::exp(x_array);
        return std::make_tuple(V, S_array, t_array);
    }
    }  // namespace bsm_impl


/**
 * Calculates the value of American put option under the Black-Scholes model,
 * using the Brennan-Schwartz algorithm.
 *
 * @param K Strike price.
 * @param T Time to expriy in years.
 * @param r Risk-free rate.
 * @param sigma Volatility.
 * @param S_min Minimum underlying price for the (S, t) grid.
 * @param S_max Maximum underlying price for the (S, t) grid.
 * @param npoints_S Number of underlying price points on the grid.
 * @param npoints_t Number of time points on the grid.
 * @return 1) Values of the option on the (S, t) grid. Shape: (npoints_S, npoints_t).
 *         2) Underlying price values from the grid. Shape: (npoints_S,).
 *         3) Time values from the grid. Shape: (npoints_t,).
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
price_american_put_bs(double K, double T, double r, double sigma,
                      double S_min, double S_max,
                      int64_t npoints_S = 1000, int64_t npoints_t = 1000) {
    double tau_max = 0.5 * T * std::pow(sigma, 2);
    double x_min = std::log(S_min / K);
    double x_max = std::log(S_max / K);
    double delta_tau = tau_max / (static_cast<double>(npoints_t) - 1.0);
    double delta_x = (x_max - x_min) / (static_cast<double>(npoints_S) - 1.0);
    double lambda = delta_tau / std::pow(delta_x, 2);
    double k = 2*r / std::pow(sigma, 2);
    std::cout << "price_american_put_bs(): delta_tau / (delta_x)^2 = " << lambda << std::endl;

    torch::Tensor x = torch::linspace(x_min, x_max, npoints_S, torch::kFloat64);
    torch::Tensor tau_array = torch::linspace(0, tau_max, npoints_t, torch::kFloat64);
    auto tau_array_a = tau_array.accessor<double, 1>();
    torch::Tensor w_matrix = torch::zeros({npoints_S, npoints_t}, torch::kFloat64);
    auto [alpha, beta, gamma] = bsm_impl::get_A_diags(npoints_S - 1, lambda);
    torch::Tensor B = bsm_impl::get_crank_B_matrix(npoints_S, lambda);

    // setting initial and boundary conditions
    w_matrix.index_put_({Slice(), 0},
                        bsm_impl::g_func(torch::zeros_like(x), x, k));
    w_matrix.index_put_({0, Slice()},
                        bsm_impl::g_func(tau_array, torch::full_like(tau_array, x_min), k));

    for (int64_t nu = 0; nu < npoints_t - 1; nu++) {
        torch::Tensor w = w_matrix.index({Slice(), nu});
        torch::Tensor d = torch::zeros_like(w);
        d.index_put_({0}, 0.5 * lambda * (bsm_impl::g_func(tau_array_a[nu], x_min, k) +
                                                   bsm_impl::g_func(tau_array_a[nu+1], x_min, k)));
        // explicit step
        torch::Tensor f = torch::matmul(B, w) + d;
        f = f.squeeze();
        // implicit step
        torch::Tensor w_ = bsm_impl::brennan_schwartz(alpha, beta, gamma, f,
                bsm_impl::g_func(torch::full_like(x, tau_array_a[nu+1]), x, k));
        w_matrix.index_put_({Slice(), nu+1}, w_);
    }
    return bsm_impl::transform(w_matrix, x, tau_array, K, T, r, sigma);
}


/**
 * Calculates the early exercise curve for American put option under the
 * Black-Scholes model.
 *
 * For reference see [Seydel2017, section 4.5].
 *
 * @param V Values of American put option on the (S, t) grid. Shape: (npoints_S, npoints_t).
 * @param S_array Underlying price values from the grid. Shape: (npoints_S,).
 * @param t_array Time values from the grid. Shape: (npoints_t,).
 * @param K Strike price.
 * @param tol Absolute tolerance paramater.
 * @return 1) Option values corresponding to early exercise price
 *            of the underlying at each moment of time. Shape: (npoints_t,).
 *         2) Early exercise price of the underlying at each moment
 *            of time. Shape: (npoints_t,).
 */
std::tuple<std::vector<double>, std::vector<double>>
find_early_exercise(const torch::Tensor& V, const torch::Tensor& S_array,
                    const torch::Tensor& t_array,
                    double K, double tol = 1e-5) {
    std::vector<double> stop_S_values = {K};
    std::vector<double> stop_V_values = {0};
    torch::Tensor euro_payoff;
    for (int64_t i = 1; i < t_array.sizes()[0]; i++) {
        euro_payoff = torch::maximum(K - S_array + tol,torch::zeros_like(S_array));
        auto stop_idx = torch::argmax(
                (V.index({Slice(), i}) > euro_payoff).to(torch::kInt64)).item<int64_t>();
        stop_S_values.push_back(S_array[stop_idx].item<double>());
        stop_V_values.push_back(V[stop_idx][i].item<double>());
    }
    return std::make_tuple(stop_V_values, stop_S_values);
}

} // namespace noa::quant
