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
 *     - Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options
 *       by simulation: a simple least-squares approach. The review of
 *       financial studies, 14(1), 113-147.
 *
 *     - Seydel, Rüdiger. Tools for computational finance. Sixth edition.
 *       Springer, 2017. Section 3.6.3.
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <torch/torch.h>

namespace noa::quant {

using Slice = torch::indexing::Slice;

/**
 * Calculate the price of American put option using the Longstaff-Schwartz method.
 *
 * @param paths Paths of underlying asset, starting from the same point S0 at initial moment of time.
 *     Shape: (N, M + 1), where N is the number of generated paths, M is the number of time steps.
 *     Dtype: torch::kFloat64.
 * @param dt Time step.
 * @param strike Option strike price.
 * @param rate Risk-free rate. Note that the paths must be generated with the
 *     same risk-free rate.
 * @return Price of the option at initial moment of time.
 */
double
price_american_put_lsm(const torch::Tensor& paths, double dt, double strike, double rate) {
    if (paths.dtype() != torch::kFloat64)
        throw std::invalid_argument("Dtype of `paths` must be `torch::kFloat64`");

    int64_t POLY_DEGREE = 3;
    int64_t n_paths = paths.sizes()[0];
    int64_t n_steps = paths.sizes()[1] - 1;

    torch::Tensor cashflow = torch::where(paths.index({Slice(), -1}) < strike,
                                          strike - paths.index({Slice(), -1}),
                                          0);
    torch::Tensor tau = n_steps * torch::ones(n_paths, torch::kInt64);

    for (int64_t j = n_steps - 1; j >= 1; j--) {
        torch::Tensor itm_mask = paths.index({Slice(), j}) < strike;
        torch::Tensor paths_itm = paths.index({Slice(), j}).index({itm_mask});

        torch::Tensor A = torch::vander(paths_itm, POLY_DEGREE + 1);
        torch::Tensor y = torch::exp(-rate * (tau.index({itm_mask}) - j) * dt) * cashflow.index({itm_mask});
        torch::Tensor fit_params;
        std::tie(fit_params, std::ignore, std::ignore, std::ignore) =
                torch::linalg::lstsq(A, y, torch::nullopt, torch::nullopt);
        torch::Tensor C_hat = torch::matmul(A, fit_params);

        torch::Tensor payoff_itm_now = strike - paths_itm;
        torch::Tensor stop_now_mask = (payoff_itm_now >= C_hat);
        torch::Tensor stop_indices = itm_mask.nonzero().squeeze().index({stop_now_mask});

        cashflow.index_put_({stop_indices}, payoff_itm_now.index({stop_now_mask}));
        tau.index_put_({stop_indices}, j);
    }
    auto C0_hat = torch::mean(torch::exp(-rate * tau * dt) * cashflow).item<double>();
    auto S0 = paths.index({0, 0}).item<double>();
    return std::fmax(strike - S0, C0_hat);
}

}
