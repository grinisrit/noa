import torch
import math


def price_barrier_option(
        paths: torch.Tensor,
        strike: float,
        maturity: float,
        rate: float,
        barrier: float,
        barrier_type: str,
        call: bool
) -> float:
    """Compute the price of a barrier option.

    Args:
        paths: Simulated paths of the underlying. Shape: (N, M + 1), where
               N is the number of paths, M is the number of time steps.
               The last time point is assumed to be the expiration time.
        strike: Strike price.
        maturity: Expiration time in years (corresponding to `paths[:, -1]`).
        rate: Risk-free rate.
        barrier: Barrier price.
        barrier_type: One of 'up-in', 'up-out', 'down-in', 'down-out'.
        call: If `True`, price call option, otherwise price put option.
    Returns:
        Price of the option.
    """
    if barrier_type not in ('up-in', 'up-out', 'down-in', 'down-out'):
        raise ValueError("`barrier_type` must be one of: 'up-in', 'up-out', 'down-in', 'down-out'.")

    if call:
        payoff = torch.maximum(paths[:, -1] - strike, torch.zeros_like(paths[:, -1]))
    else:
        payoff = torch.maximum(strike - paths[:, -1], torch.zeros_like(paths[:, -1]))
    if barrier_type == 'up-in':
        condition = torch.max(paths, dim=1).values >= barrier
    elif barrier_type == 'up-out':
        condition = torch.max(paths, dim=1).values < barrier
    elif barrier_type == 'down-in':
        condition = torch.min(paths, dim=1).values <= barrier
    else:  # down-out
        condition = torch.min(paths, dim=1).values > barrier
    return math.exp(-rate*maturity) * torch.mean(payoff * condition).item()
