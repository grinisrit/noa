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
#include <tuple>

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
 * @param rate Risk-free rate. Note that the input paths must be generated with
 *     the same risk-free rate as the value of this argument.
 * @return Price of the option at initial moment of time.
 */
// double
// price_american_put_lsm(const torch::Tensor& paths, double dt, double strike, double rate) {
//     if (paths.dtype() != torch::kFloat64)
//         throw std::invalid_argument("Dtype of `paths` must be `torch::kFloat64`");

//     int64_t POLY_DEGREE = 3;
//     int64_t n_paths = paths.sizes()[0];
//     int64_t n_steps = paths.sizes()[1] - 1;

//     torch::Tensor cashflow = torch::where(paths.index({Slice(), -1}) < strike,
//                                           strike - paths.index({Slice(), -1}),
//                                           0);
//     torch::Tensor tau = n_steps * torch::ones(n_paths, torch::kInt64);

//     for (int64_t j = n_steps - 1; j >= 1; j--) {
//         torch::Tensor itm_mask = paths.index({Slice(), j}) < strike;
//         torch::Tensor paths_itm = paths.index({Slice(), j}).index({itm_mask});

//         torch::Tensor A = torch::vander(paths_itm, POLY_DEGREE + 1);
//         torch::Tensor y = torch::exp(-rate * (tau.index({itm_mask}) - j) * dt) * cashflow.index({itm_mask});
//         torch::Tensor fit_params;
//         std::tie(fit_params, std::ignore, std::ignore, std::ignore) =
//                 torch::linalg::lstsq(A, y, torch::nullopt, torch::nullopt);
//         torch::Tensor C_hat = torch::matmul(A, fit_params);

//         torch::Tensor payoff_itm_now = strike - paths_itm;
//         torch::Tensor stop_now_mask = (payoff_itm_now >= C_hat);
//         torch::Tensor stop_indices = itm_mask.nonzero().squeeze().index({stop_now_mask});

//         cashflow.index_put_({stop_indices}, payoff_itm_now.index({stop_now_mask}));
//         tau.index_put_({stop_indices}, j);
//     }
//     auto C0_hat = torch::mean(torch::exp(-rate * tau * dt) * cashflow).item<double>();
//     auto S0 = paths.index({0, 0}).item<double>();
//     return std::fmax(strike - S0, C0_hat);
// }

struct LSMResult {
    torch::Tensor option_price;
    torch::Tensor reg_poly_coefs;
    torch::Tensor initial_cont_value;

    //std::optional<std::list<torch::Tensor> reg_x_vals; return_extra flag for cpp ignored  
    //std::optional<std::list<torch::Tensor> reg_x_vals;    
};

LSMResult
_lsm_regression_step(
    const torch::Tensor& paths,
    double& dt,
    const torch::Tensor& strike,
    const torch::Tensor& rate,
    int64_t reg_poly_degree,
    bool return_extra
)
{
    int64_t n_paths = paths.sizes()[0];
    int64_t n_steps = paths.sizes()[1] - 1;

    torch::Tensor cashflow = torch::where(paths.index({Slice(), -1}) < strike,
                                          strike - paths.index({Slice(), -1}),
                                          0);
    torch::Tensor tau = n_steps * torch::ones(n_paths, torch::kInt64);

    torch::Tensor reg_poly_coefs = torch::zeros({paths.sizes()[1], reg_poly_degree + 1});
    // no nan in libtorch...
    // reg_poly_coefs.index({-1}) *= torch::nan;  // continuation value is not defined at expiration
    // reg_poly_coefs.index({0}) *= torch::nan;

    if (return_extra) {
    }

    for (int64_t j = n_steps - 1; j > 0; j--) {
        torch::Tensor itm_mask = paths.index({Slice(), j}) < strike;
        if (torch::sum(itm_mask).item<int64_t>() == 0) {
            continue;
        }
        torch::Tensor paths_itm = paths.index({Slice(), j}).index({itm_mask});

        torch::Tensor A = torch::vander(paths_itm, reg_poly_degree + 1);
        torch::Tensor y = torch::exp(-rate * (tau.index({itm_mask}) - j) * dt) * cashflow.index({itm_mask});
        torch::Tensor fit_params;
        std::tie(fit_params, std::ignore, std::ignore, std::ignore) =
                torch::linalg::lstsq(A, y, torch::nullopt, torch::nullopt);
        torch::Tensor C_hat = torch::matmul(A, fit_params);
        reg_poly_coefs.index({j}) = fit_params;

        if (return_extra) {
        }

        torch::Tensor payoff_itm_now = strike - paths_itm;
        torch::Tensor stop_now_mask = (payoff_itm_now >= C_hat);
        cashflow.index_put_({itm_mask}, torch::where(stop_now_mask, payoff_itm_now, cashflow.index({itm_mask})));
        tau.index_put_({itm_mask}, torch::where(stop_now_mask, j, tau.index({itm_mask})));
    }

    torch::Tensor C_hat = torch::mean(torch::exp(-rate * tau * dt) * cashflow);
    torch::Tensor payoff_now = torch::maximum(strike - paths.index({0, 0}), torch::tensor(0.0));
    torch::Tensor option_price = torch::maximum(payoff_now, C_hat);

    LSMResult result{option_price, reg_poly_coefs, C_hat};
    if (return_extra) {
    }

    return result;
}

LSMResult
_lsm_pricing_step(
    const torch::Tensor& paths,
    double& dt,
    const torch::Tensor& strike,
    const torch::Tensor& rate,
    int64_t reg_poly_degree,
    LSMResult& result_reg_step
)
{
    int64_t n_paths = paths.sizes()[0];
    int64_t n_steps = paths.sizes()[1] - 1;
    torch::Tensor payoff = torch::zeros(n_paths);
    torch::Tensor stopped_mask = torch::zeros(n_paths, torch::TensorOptions().dtype(torch::kBool));
    torch::Tensor payoff_now = torch::maximum(strike - paths.index({0, 0}), torch::tensor(0.0));

    torch::Tensor option_price;
    if (torch::all(payoff_now > result_reg_step.initial_cont_value).item<bool>()) {
        option_price = payoff_now;
    } else {
        for (int64_t j = 1; j < n_steps - 1; ++j) {
            torch::Tensor itm_mask = (paths.index({Slice(), j}) < strike) & (~stopped_mask);
            if (itm_mask.numel() == 0) {
                continue;
            }
            torch::Tensor paths_itm = paths.index({Slice(), j}).index({itm_mask});
            
            torch::Tensor vander = torch::vander(paths_itm, reg_poly_degree + 1);
            torch::Tensor cont_value = torch::matmul(vander, result_reg_step.reg_poly_coefs.index({j}));  // continuation value

            torch::Tensor payoff_itm_now = strike - paths_itm;
            torch::Tensor stop_now_mask = (payoff_itm_now >= cont_value);
            payoff.index_put_({itm_mask}, torch::where(
                stop_now_mask,
                payoff_itm_now * torch::exp(-rate * j * dt),
                payoff.index({itm_mask})
            ));

            stopped_mask.index_put_({itm_mask}, stopped_mask.index({itm_mask}) | stop_now_mask); 
        }

        // last step - expiration time
        torch::Tensor stop_now_mask = ~stopped_mask;
        payoff_now = torch::maximum(strike - paths.index({Slice(), -1}), torch::zeros(n_paths));
        payoff = torch::where(
            stop_now_mask,
            payoff_now * torch::exp(-rate * n_steps * dt),
            payoff
        );
        option_price = torch::mean(payoff);
    }

    result_reg_step.option_price = option_price;
    return result_reg_step;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
price_american_put_lsm(
    const torch::Tensor& paths_regression,
    const torch::Tensor& paths_pricing,
    double dt,
    const torch::Tensor& strike,
    const torch::Tensor& rate,
    int64_t reg_poly_degree,
    bool return_extra)
{
    if (!(torch::all(paths_regression.index({Slice{}, 0}) == paths_regression.index({0, 0})).item<bool>()
        && torch::all(paths_pricing.index({Slice{}, 0}) == paths_pricing.index({0, 0})).item<bool>())) {
        throw std::invalid_argument("Paths of the underlying must start from the same value at initial moment of time");
    }

    if (paths_regression.sizes()[1] != paths_pricing.sizes()[1]) {
        throw std::invalid_argument("`paths1` and `paths2` must have the same number of time steps");
    }

    LSMResult result_reg_step;
    {
        torch::NoGradGuard no_grad;
        result_reg_step = _lsm_regression_step(
            paths_regression, dt, strike, rate, reg_poly_degree, return_extra);
    }

    result_reg_step = _lsm_pricing_step(paths_pricing, dt, strike, rate, reg_poly_degree, result_reg_step);
    return {result_reg_step.option_price, result_reg_step.reg_poly_coefs, result_reg_step.initial_cont_value};
}


}
