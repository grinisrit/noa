"""
References:
    [Seydel2017] Seydel, Rüdiger. Tools for computational finance.
    Sixth edition. Springer, 2017. Section 3.6.3.
"""


import torch
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class LSMResult:
    """Computation results of LSM algorithm.

    Attributes:
        option_price:
            Price of the option at the initial moment of time. Shape: (1, ).
        reg_poly_coefs:
            Polynomial coefficients that were fitted on the regression step.
            Shape: (N_STEPS + 1, REG_POLY_DEGREE + 1). `reg_poly_coefs[0]` and
            `reg_poly_coefs[-1]` are always NaN for the following reasons.
            At the initial moment of time, all paths have the same value, so 
            continuation value can only be calculated for this single point. This
            value is contained in the attribute `initial_cont_value`.
            At the final moment of time, the option is exercised, so continuation
            value is not defined.
        initial_cont_value:
            Continuation value at the initial moment of time. Shape: (1, ).
            At this moment, all paths have the same value, so continuation value
            can only be calculated for this single point.
        reg_x_vals:
            Values of in-the-money subset of underlying paths at each moment of time.
            If there were no ITM paths at some moment of time, the corresponding
            list item will be `None`. Length: N_STEPS + 1.
        reg_y_vals:
            Values of dependent variable for regression for in-the-money
            subset of underlying paths at each moment of time. If there were
            no ITM paths at some moment of time, the corresponding list item
            will be `None`. Length: N_STEPS + 1.

    Notes:
        - `N_STEPS` is the number of time steps in paths of the underlying asset
            which were passed as the argument for `price_american_put_lsm()`
            function.
        - `REG_POLY_DEGREE` is the value of the eponymous argument of
            `price_american_put_lsm()` function.
        - For each index `i`, the tensors `reg_x_vals[i]` and `reg_y_vals[i]`
            have the same length. For explanation of these variables,
            see [Seydel2017], section 3.6.3, algorithm 3.14, item (c).
    """
    option_price: torch.Tensor
    reg_poly_coefs: torch.Tensor
    initial_cont_value: torch.Tensor
    reg_x_vals: Optional[list[Union[torch.Tensor, None]]] = None
    reg_y_vals: Optional[list[Union[torch.Tensor, None]]] = None


def price_american_put_lsm(
    paths_regression: torch.Tensor,
    paths_pricing: torch.Tensor,
    dt: torch.Tensor,
    strike: torch.Tensor,
    rate: torch.Tensor,
    reg_poly_degree: int = 3,
    return_extra: bool = False
) -> LSMResult:
    """Calculates the price of American put option using the LSM algorithm.

    This is the modification of LSM algorithm where it's divided into two steps:
    1. Regression. Find approximation of continuation value C(S, t) via regression.
    2. Pricing. Use the continuation value from step 1 as a sort of barrier:
       when the payoff for a given path crosses C(S, t), exercise the option on this path.

    Two separate sets of paths are used on each step to reduce the bias. These
    sets may have different number of paths, but they must have the same number
    of time steps.

    Gradient computation is disabled on the regression step.

    Args:
        paths_regression: Paths of underlying asset, starting from the same
            point S0 at initial moment of time. Used on regression step.
        paths_pricing: Paths of underlying asset, starting from the same
            point S0 at initial moment of time. Used on pricing step.
        dt: Time step.
        strike: Option strike price.
        rate: Risk-free rate. Note that the input paths must be generated with
            the same risk-free rate as the value of this parameter.
        reg_poly_degree: Degree of polynomial for regression.
        return_extra: If `True`, in addition to other values return the
            in-the-money points which are used as the data for regression.

    Shape:
        - paths_regression, paths_pricing: (N, M + 1), where N is the number of
            generated paths, M is the number of time steps.
        - dt: (1, )
        - strike: (1, )
        - rate: (1, )

    Returns:
        Computation result of LSM algorithm, represented as `LSMResult` object.
        Contains computed option price, polynomial coefficients that were fitted
        on the regression step and continuation value at initial moment of time.
        If `return_extra` is `True`, also contains the in-the-money points which
        were used as the data for regression.
    """
    if not (torch.all(paths_regression[:, 0] == paths_regression[0, 0])
            and torch.all(paths_pricing[:, 0] == paths_pricing[0, 0])):
        raise ValueError('Paths of the underlying must start from the same value at initial moment of time')

    if paths_regression.shape[1] != paths_pricing.shape[1]:
        raise ValueError('`paths1` and `paths2` must have the same number of time steps')

    with torch.no_grad():
        result_reg_step = _lsm_regression_step(
            paths_regression, dt, strike, rate, reg_poly_degree, return_extra)

    return _lsm_pricing_step(paths_pricing, dt, strike, rate, reg_poly_degree, result_reg_step)


def _lsm_regression_step(
    paths: torch.Tensor,
    dt: torch.Tensor,
    strike: torch.Tensor,
    rate: torch.Tensor,
    reg_poly_degree: int,
    return_extra: bool = False
) -> LSMResult:
    """Implementation of algorithm 3.15 from [Seydel2017]."""
    n_paths = paths.shape[0]
    n_steps = paths.shape[1] - 1

    cashflow = torch.where(paths[:, -1] < strike, strike - paths[:, -1], 0)
    tau = n_steps * torch.ones(n_paths, dtype=torch.int64)

    reg_poly_coefs = torch.zeros((paths.shape[1], reg_poly_degree + 1))
    reg_poly_coefs[-1] *= torch.nan  # continuation value is not defined at expiration
    reg_poly_coefs[0] *= torch.nan

    if return_extra:
        reg_x_vals = [None] * n_steps
        reg_y_vals = [None] * n_steps

    for j in range(n_steps - 1, 0, -1):
        itm_mask = paths[:, j] < strike
        if torch.sum(itm_mask) == 0:
            continue
        paths_itm = paths[:, j][itm_mask]

        A = torch.vander(paths_itm, N=reg_poly_degree + 1)
        y = torch.exp(-rate * (tau[itm_mask] - j) * dt) * cashflow[itm_mask]
        fit_params = torch.linalg.lstsq(A, y).solution
        C_hat = torch.matmul(A, fit_params)  # continuation value
        reg_poly_coefs[j] = fit_params

        if return_extra:
            reg_x_vals[j] = paths_itm
            reg_y_vals[j] = y

        payoff_itm_now = strike - paths_itm
        stop_now_mask = (payoff_itm_now >= C_hat)
        cashflow[itm_mask] = torch.where(stop_now_mask, payoff_itm_now, cashflow[itm_mask])
        tau[itm_mask] = torch.where(stop_now_mask, j, tau[itm_mask])

    C_hat = torch.mean(torch.exp(-rate * tau * dt) * cashflow)
    S0 = torch.mean(paths[:, 0])
    payoff_now = torch.maximum(strike - S0, torch.tensor(0.0))
    option_price = torch.maximum(payoff_now, C_hat)

    result = LSMResult(option_price, reg_poly_coefs, C_hat)
    if return_extra:
        result.reg_x_vals = reg_x_vals
        result.reg_y_vals = reg_y_vals
    return result


def _lsm_pricing_step(
    paths: torch.Tensor,
    dt: torch.Tensor,
    strike: torch.Tensor,
    rate: torch.Tensor,
    reg_poly_degree: int,
    result_reg_step: LSMResult
) -> LSMResult:
    n_paths = paths.shape[0]
    n_steps = paths.shape[1] - 1
    payoff = torch.zeros(n_paths)
    stopped_mask = torch.zeros(n_paths, dtype=torch.bool)

    S0 = torch.mean(paths[:, 0])
    payoff_now = torch.maximum(strike - S0, torch.tensor(0.0))
    if torch.all(payoff_now > result_reg_step.initial_cont_value):
        option_price = payoff_now
    else:
        for j in range(1, n_steps - 1):
            itm_mask = paths[:, j] < strike
            if torch.sum(itm_mask) == 0:
                continue
            paths_itm = paths[:, j][itm_mask]

            vander = torch.vander(paths_itm, N=reg_poly_degree + 1)
            cont_value = torch.matmul(vander, result_reg_step.reg_poly_coefs[j])  # continuation value

            payoff_itm_now = strike - paths_itm
            stop_now_mask = (payoff_itm_now >= cont_value)
            payoff[itm_mask] = torch.where(
                stop_now_mask & (~stopped_mask[itm_mask]),
                payoff_itm_now * torch.exp(-rate * j * dt),
                payoff[itm_mask]
            )
            stopped_mask[itm_mask] |= stop_now_mask

        # last step - expiration time
        stop_now_mask = ~stopped_mask
        payoff_now = torch.maximum(strike - paths[:, -1], torch.zeros(n_paths))
        payoff = torch.where(
            stop_now_mask,
            payoff_now * torch.exp(-rate * n_steps * dt),
            payoff
        )
        option_price = torch.mean(payoff)
    result_reg_step.option_price = option_price
    return result_reg_step
