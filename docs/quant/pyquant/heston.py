"""
References:
    - [Grzelak2019] Oosterlee, C. W., & Grzelak, L. A. (2019). Mathematical
      modeling and computation in finance: with exercises and Python and
      MATLAB compute codes. World Scientific.

    - [Andersen2007] Andersen, L.B., 2007. Efficient simulation of the Heston
      stochastic volatility model. Available at SSRN 946405.
"""


import torch
from typing import Tuple

__all__ = ['noncentral_chisquare', 'generate_cir', 'generate_heston']


def noncentral_chisquare(
        df: torch.Tensor,
        nonc: torch.Tensor
) -> torch.Tensor:
    """ Generates samples from a noncentral chi-square distribution.
    Quadratic Exponential scheme from [Andersen2007] is used.

    Args:
        df: Degrees of freedom, must be > 0.
        nonc: Non-centrality parameter, must be >= 0.

    Returns:
        Tensor with generated tensor. Shape: same as `df` and `nonc`, if
        they have the same shape.
    """
    # algorithm is summarized in [Andersen2007, section 3.2.4]
    PSI_CRIT = 1.5  # threshold value for switching between sampling algorithms
    m = df + nonc
    s2 = 2*df + 4*nonc
    psi = s2 / m.pow(2)
    # quadratic
    psi_inv = 1 / psi
    b2 = 2*psi_inv - 1 + (2*psi_inv).sqrt() * (2*psi_inv - 1).sqrt()
    a = m / (1 + b2)
    sample_quad = a * (b2.sqrt() + torch.randn_like(a)).pow(2)
    # exponential
    p = (psi - 1) / (psi + 1)
    beta = (1 - p) / m
    rand = torch.rand_like(p)
    sample_exp = torch.where((p < rand) & (rand <= 1),
                             beta.pow(-1)*torch.log((1-p)/(1-rand)),
                             torch.zeros_like(rand))
    return torch.where(psi <= PSI_CRIT, sample_quad, sample_exp)


def generate_cir(
        n_paths: int,
        n_steps: int,
        dt: float,
        init_state: torch.Tensor,
        kappa: torch.Tensor,
        theta: torch.Tensor,
        eps: torch.Tensor
) -> torch.Tensor:
    """Generates paths of Cox-Ingersoll-Ross (CIR) process.

    CIR process is described by the SDE:
        dv(t) = κ·(θ - v(t))·dt + ε·sqrt(v(t))·dW(t)
    (see [Grzelak2019, section 8.1.2]).

    For path generation, Andersen's Quadratic Exponential scheme is used
    (see [Andersen2007], [Grzelak2019, section 9.3.4]).

    Args:
        n_paths: Number of paths to simulate.
        n_steps: Number of time steps.
        dt: Time step.
        init_state: Initial states of the paths, i.e. v(0). Shape: (n_paths,).
        kappa: Parameter κ.
        theta: Parameter θ.
        eps: Parameter ε.

    Returns:
        Simulated paths of CIR process. Shape: (n_paths, n_steps + 1).
    """
    if init_state.shape != torch.Size((n_paths,)):
        raise ValueError('Shape of `init_state` must be (n_paths,)')

    paths = torch.empty((n_paths, n_steps + 1), dtype=init_state.dtype)
    paths[:, 0] = init_state

    delta = 4 * kappa * theta / (eps * eps) * torch.ones_like(init_state)
    exp = torch.exp(-kappa*dt)
    c_bar = 1 / (4*kappa) * eps * eps * (1 - exp)
    for i in range(0, n_steps):
        v_cur = paths[:, i]
        kappa_bar = v_cur * 4*kappa*exp / (eps * eps * (1 - exp))
        # [Grzelak2019, definition 8.1.1]
        v_next = c_bar * noncentral_chisquare(delta, kappa_bar)
        paths[:, i+1] = v_next
    return paths


def generate_heston(
        n_paths: int,
        n_steps: int,
        dt: float,
        init_state_price: torch.Tensor,
        init_state_var: torch.Tensor,
        kappa: torch.Tensor,
        theta: torch.Tensor,
        eps: torch.Tensor,
        rho: torch.Tensor,
        drift: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates time series following the Heston model.

    Stochastic process of the Heston model is described by a system of SDE's:
        dS(t) = μ·S(t)·dt + sqrt(v(t))·S(t)·dW_1(t);
        dv(t) = κ(θ - v(t))·dt + ε·sqrt(v(t))·dW_2(t).

    Time series are generated using Andersen's Quadratic Exponential
    scheme [Andersen2007]. Also see [Grzelak, section 9.4.3].

    Args:
        n_paths: Number of simulated paths.
        n_steps: Number of time steps.
        dt: Time step.
        init_state_price: Initial states of the price paths, i.e. S(0). Shape: (n_paths,).
        init_state_var: Initial states of the variance paths, i.e. v(0). Shape: (n_paths,).
        kappa: Parameter κ - the rate at which v(t) reverts to θ.
        theta: Parameter θ - long-run average variance.
        eps: Parameter ε - volatility of variance.
        rho: Correlation between underlying Brownian motions for S(t) and v(t).
        drift: Drift parameter μ.

    Returns:
        Two tensors: 1) simulated paths for price, 2) simulated paths for variance.
        Both tensors have the shape (n_paths, n_steps + 1).
    """
    if init_state_price.shape != torch.Size((n_paths,)):
        raise ValueError('Shape of `init_state_price` must be (n_paths,)')
    if init_state_var.shape != torch.Size((n_paths,)):
        raise ValueError('Shape of `init_state_var` must be (n_paths,)')

    gamma2 = 0.5
    # regularity condition [Andersen 2007, section 4.3.2]
    if rho > 0:  # always satisfied when rho <= 0
        L = rho*dt*(kappa/eps - 0.5*rho)
        R = 2*kappa/(eps*eps*(1 - torch.exp(-kappa*dt))) - rho/eps
        if R<=0 or L==0 or (L<0 and R>=0):
            # When (L<0 && R<=0), L/R is always < 0.5.
            # (L>0 && R<=0) never happens.
            # In other cases, regularity condition is always satisfied.
            pass
        elif L > 0:
            gamma2 = min(0.5, R / L * 0.9)  # multiply by 0.9 to have some margin
    gamma1 = 1.0 - gamma2

    k0 = -rho * kappa * theta * dt / eps
    k1 = gamma1 * dt * (kappa * rho / eps - 0.5) - rho / eps
    k2 = gamma2 * dt * (kappa * rho / eps - 0.5) + rho / eps
    k3 = gamma1 * dt * (1 - rho * rho)
    k4 = gamma2 * dt * (1 - rho * rho)

    var_paths = generate_cir(n_paths, n_steps, dt, init_state_var, kappa, theta, eps)
    log_paths = torch.empty((n_paths, n_steps + 1), dtype=init_state_price.dtype)
    log_paths[:, 0] = init_state_price.log()

    for i in range(0, n_steps):
        v_i = var_paths[:, i]
        v_next = var_paths[:, i+1]
        next_vals = drift*dt + log_paths[:, i] + k0 + k1*v_i + k2*v_next + \
            torch.sqrt(k3*v_i + k4*v_next) * torch.randn_like(v_i)
        log_paths[:, i+1] = next_vals
    return log_paths.exp(), var_paths
