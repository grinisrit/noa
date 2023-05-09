"""
References:
    [Seydel2017] Seydel, Rüdiger. Tools for computational finance.
    Sixth edition. Springer, 2017. Section 3.6.3.
"""


import torch
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class LSMResult:
    """Represents the computation result of LSM algorithm.

    Attributes:
        option_price:
            Price of the option at initial moment of time. Shape: (1, ).
        stop_S_vals:
            Price values of the underlying for early exercise curve.
        stop_t_vals:
            Time values for early exercise curve.
        C_hats:
            Continuation value approximations for in-the-money subset of
            underlying paths at each moment of time. Length: N_STEPS + 1
        y_vals:
            Values of dependent variable for regression for in-the-money
            subset of underlying paths at each moment of time Length: N_STEPS + 1.
        paths_itm_vals:
            Values of in-the-money subset of underlying paths at each moment of time.
            Length: N_STEPS + 1.

    Notes:
        - `stop_S_vals` and `stop_t_vals` have the same length which becomes known
        only after algorithm is executed.
        - `N_STEPS` is the number of time steps in paths of the underlying asset
        (input for the LSM algorithm).
        - For each index `i`, the tensors `C_hats[i]`, `paths_itm_vals[i]` and `y_vals[i]`
        have the same length. For explanation of these variables, see [Seydel2017],
        section 3.6.3, algorithm 3.14, item (c).
    """
    option_price:    torch.Tensor
    stop_S_vals:     torch.Tensor
    stop_t_vals:     torch.Tensor
    C_hats:          Optional[List[torch.Tensor]] = None
    y_vals:          Optional[List[torch.Tensor]] = None
    paths_itm_vals:  Optional[List[torch.Tensor]] = None


def price_american_put_lsm(
        paths: torch.Tensor,
        dt: torch.Tensor,
        strike: torch.Tensor,
        rate: torch.Tensor,
        return_extra: bool = False
) -> LSMResult:
    """Calculates the price of American put option using the Longstaff-Schwartz method.

    Args:
        paths: Paths of underlying asset, starting from the same point S0 at
            initial moment of time.
        dt: Time step.
        strike: Option strike price.
        rate: Risk-free rate. Note that the input paths must be generated with
            the same risk-free rate as the value of this parameter.
        return_extra: If `True`, store and return continuation value curves and
            in-the-money points which are used as data for regression.

    Shape:
        - paths: (N, M + 1), where N is the number of generated paths,
              M is the number of time steps.
        - dt: (1, )
        - strike: (1, )
        - rate: (1, )

    Returns:
        Computation result of LSM algorithm, represented as `LSMResult` object.
        Contains price of the option at initial moment of time and early-exercise
        curve. If `return_extra` is `True`, also contains continuation value curves
        and in-the-money points which are used as data for regression.
    """
    if not torch.all(paths[:, 0] == paths[0, 0]):
        raise ValueError('Paths of the underlying must start from the same value '
                         'at initial moment of time')
    POLY_DEGREE = 3
    n_paths = paths.shape[0]
    n_steps = paths.shape[1] - 1

    cashflow = torch.where(paths[:, -1] < strike, strike - paths[:, -1], 0)
    tau = n_steps * torch.ones(n_paths, dtype=torch.int64)
    stop_S_vals = []
    stop_t_vals = []

    if return_extra:
        C_hats = [None] * (n_steps + 1)
        paths_itm_vals = [None] * (n_steps + 1)
        y_vals = [None] * (n_steps + 1)

    for j in range(n_steps - 1, 0, -1):
        itm_mask = paths[:, j] < strike
        paths_itm = paths[:, j][itm_mask]

        A = torch.vander(paths_itm, N=POLY_DEGREE + 1)
        y = torch.exp(-rate * (tau[itm_mask] - j) * dt) * cashflow[itm_mask]
        fit_params = torch.linalg.lstsq(A, y).solution
        C_hat = torch.matmul(A, fit_params)

        if return_extra:
            C_hats[j] = C_hat
            paths_itm_vals[j] = paths_itm
            y_vals[j] = y

        payoff_itm_now = strike - paths_itm
        stop_now_mask = (payoff_itm_now >= C_hat)
        cashflow[itm_mask] = torch.where(stop_now_mask, payoff_itm_now, cashflow[itm_mask])
        tau[itm_mask] = torch.where(stop_now_mask, j, tau[itm_mask])
        # early-exercise curve
        if torch.any(stop_now_mask):
            stop_S_vals.append(torch.max(paths_itm[stop_now_mask]).item())
            stop_t_vals.append(j * dt)

    C_hat = torch.mean(torch.exp(-rate * tau * dt) * cashflow)
    S0 = paths[0, 0].item()
    option_price = max(strike - S0, C_hat)
    stop_S_vals = torch.flip(torch.DoubleTensor(stop_S_vals), dims=(0,))
    stop_t_vals = torch.flip(torch.DoubleTensor(stop_t_vals), dims=(0,))

    result = LSMResult(option_price, stop_S_vals, stop_t_vals)
    if return_extra:
        result.C_hats = C_hats
        result.paths_itm_vals = paths_itm_vals
        result.y_vals = y_vals
    return result

