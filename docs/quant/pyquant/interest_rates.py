from .torch_spline import PchipSpline1D
from .heston_sim import generate_cir

import torch
import os
from dataclasses import dataclass
import pandas as pd
import click


def build_ois_yield_curve_from_now_starting(forwards, tenors):
    zcbs = 1 / (1 + tenors * forwards)
    return PchipSpline1D(tenors, zcbs)


def build_fwd_curve(forwards, time_to_maturity):
    return PchipSpline1D(time_to_maturity, forwards)


def caplet_premium_from_now_starting(vol_surface, key_rate_fwd_curve, ois_yield_curve):
    time_to_maturity = torch.tensor(vol_surface.time_to_maturity.values)
    tenor = torch.tensor(vol_surface.tenor.values)
    g_hat = torch.sqrt(time_to_maturity - (2/3) * tenor)

    strike = torch.tensor(vol_surface.strike.values)
    fwd = key_rate_fwd_curve.evaluate(time_to_maturity)
    discount = ois_yield_curve.evaluate(time_to_maturity)
    iv = torch.tensor(vol_surface.implied_normal_vol.values)

    sigma_moneyness = (fwd - strike) / (iv * g_hat)

    normal = torch.distributions.Normal(0,1)
    Phi = normal.cdf
    log_phi = normal.log_prob

    pv = tenor * discount * ( (fwd - strike) * Phi(sigma_moneyness) + iv * g_hat * torch.exp(log_phi(sigma_moneyness)) )
    return torch.maximum(pv,torch.tensor(0.))


def build_ifwd_curve_from_now_starting(forwards, tenors):
    zcbs = 1 / (1 + tenors * forwards)
    log_zcbs = torch.log(zcbs)
    return PchipSpline1D(tenors, - log_zcbs)


def build_ifwd_key_curve_from_now_starting(key_ifwd_values, key_fwd_values, tenors):
    clamp_ifwd_vals = torch.clamp(key_ifwd_values, min=key_fwd_values-0.01, max=key_fwd_values+0.01)
    return build_ifwd_curve_from_now_starting(clamp_ifwd_vals, tenors)


def id_from_years(years: torch.Tensor, timeline: torch.Tensor) -> torch.Tensor:
    """
    Convert years to indices in the timeline.
    
    Args:
        years: Tensor of time values in years (can be scalar or tensor)
        timeline: Timeline tensor (assumed to be in years, e.g., torch.linspace(0, 10., 3651))
    
    Returns:
        Tensor of indices (integers) corresponding to the given years
    """
    if timeline.numel() == 1:
        return torch.zeros_like(years, dtype=torch.long)
    
    # Calculate points per year from the timeline
    timeline_min = timeline.min()
    timeline_range = timeline.max() - timeline_min
    n_points = timeline.numel()
    
    if timeline_range > 0:
        # Convert years relative to timeline start, then to indices
        # Formula: index = ceil((years - timeline_min) * (n_points - 1) / timeline_range)
        # This ensures year=timeline_min maps to index 0, and year=timeline_max maps to index n_points-1
        points_per_year = (n_points - 1) / timeline_range
        indices = ((years - timeline_min) * points_per_year).ceil().long()
    else:
        indices = torch.zeros_like(years, dtype=torch.long)
    
    # Clamp to valid range [0, n_points - 1]
    indices = torch.clamp(indices, 0, n_points - 1)
    
    return indices


def zcb_yields(
    t0: torch.Tensor,
    timeline: torch.Tensor,
    zcbs: torch.Tensor
) -> torch.Tensor:
    """
    Compute simple yields from t0 for a timeline and corresponding ZCB values.
    
    Arguments:
        t0: Reference time point (in years)
        timeline: Time points (in years), shape (n,)
        zcbs: Zero-coupon bond values, shape (..., n) where last dimension corresponds to timeline
    
    Returns:
        Yields tensor with same shape as zcbs
    """
    # Compute time differences from t0
    # time_diffs: (n,)
    time_diffs = timeline - t0
    
    # Avoid division by zero for t0 itself
    # For points at or before t0, yield is undefined or zero
    # mask: (n,)
    mask = time_diffs > 0
    
    # Initialize yields tensor with same shape as zcbs
    yields = torch.zeros_like(zcbs)
    
    # Compute simple yields: y(t) = (1 - P(t0, t)) / (P(t0, t) * (t - t0))
    # Use ... indexing to operate on last dimension
    # time_diffs[mask] broadcasts automatically with zcbs[..., mask]
    yields[..., mask] = (1 - zcbs[..., mask]) / (zcbs[..., mask] * time_diffs[mask])
    
    return yields


def generate_hull_white(
        n_paths: int,
        timeline: torch.Tensor,
        init_state: torch.Tensor,
        lam: torch.Tensor,
        sigma: torch.Tensor
) -> torch.Tensor:
    dt_steps = timeline.diff()
    n_steps = dt_steps.shape[0]
    # Paths shape: (*init_state.shape, n_paths, n_steps + 1) so time dimension is at -2
    paths = torch.empty((*init_state.shape, n_paths, n_steps + 1), dtype=init_state.dtype)
    paths[..., 0] = init_state.unsqueeze(-1).expand(*init_state.shape, n_paths)
    for i in range(0, n_steps):
        x = paths[..., i].clone()
        Z = torch.randn_like(x)
        dt = dt_steps[i]
        x_next = x * torch.exp(- dt * lam) + Z * torch.sqrt( sigma * sigma * (1 - torch.exp(- 2 * dt * lam)) / (2 * lam) ) 
        paths[..., i+1] = x_next
    return paths


@dataclass
class HullWhiteModel:
    timeline: torch.Tensor
    lam: torch.Tensor
    sigma: torch.Tensor
    x0: torch.Tensor
    x_paths: torch.Tensor
    alpha_curve: torch.Tensor
    r_paths: torch.Tensor 
    sum_r_dt: torch.Tensor
    ois_zcbs: torch.Tensor


def create_hull_white_model(
    timeline: torch.Tensor,
    n_paths: int,
    ois_ifwd_curve: PchipSpline1D,
    r0: torch.Tensor,
    x0: torch.Tensor,
    lam: torch.Tensor,
    sigma: torch.Tensor
) -> HullWhiteModel:
    assert timeline.dim() == 1
    assert r0.dim() == 0
    assert x0.dim() == 0
    assert lam.dim() == 0
    assert sigma.dim() == 0

    # Generate Hull-White paths
    x_paths = generate_hull_white(n_paths, timeline, x0, lam, sigma)
    
    # Use full timeline[1:] for ZCB maturities
    dt = timeline.diff()
    zcb_ttm = timeline[1:]
    zcbs0 = torch.exp(-(dt * x_paths[:, :-1]).cumsum(1)).mean(0)
    
    # Build forward curve from ZCBs
    r0_fwd = PchipSpline1D(zcb_ttm, -torch.log(zcbs0))
    
 
    alpha_curve = torch.zeros_like(timeline)

    alpha_curve[0] = r0
    alpha_curve[1:] = ois_ifwd_curve.derivative(timeline[1:]) - r0_fwd.derivative(timeline[1:])
    
    # Compute full rate paths: r_t = alpha_t + x_t
    r_paths = alpha_curve + x_paths
    
    # Compute cumulative sum for discounting
    sum_r_dt = (dt * r_paths[:, :-1]).cumsum(-1)
    
    # Compute OIS zero-coupon bonds
    ois_zcbs = torch.exp(-sum_r_dt).mean(-2)
   
    
    return HullWhiteModel(
        timeline=timeline,
        lam=lam,
        sigma=sigma,
        x0=x0,
        x_paths=x_paths,
        alpha_curve=alpha_curve,
        r_paths=r_paths,
        sum_r_dt=sum_r_dt,
        ois_zcbs=ois_zcbs
    )


def hull_white_model_asof(
    as_of: torch.Tensor,
    n_paths: int,
    model: HullWhiteModel
) -> HullWhiteModel:
    
    assert as_of.dim() == 0
    # Map as_of to timeline index using id_from_years
    as_of_id = id_from_years(as_of, model.timeline)
    
    # Get the snapshot of x_paths at as_of_id
    # x_paths has shape (old_n_paths, n_steps + 1) when x0 is scalar
    x0_new = model.x_paths[:, as_of_id]  # shape: (old_n_paths,)
    
    # Use timeline[id:] for new_timeline
    new_timeline = model.timeline[as_of_id:]
    dt_new = new_timeline.diff()
    
    # Use alpha_curve[id:] for new_alpha_curve
    new_alpha_curve = model.alpha_curve[as_of_id:]
    
    # Branching:
    # x_paths_new will have shape (old_n_paths, n_paths, n_steps + 1)
    x_paths_new = generate_hull_white(n_paths, new_timeline, x0_new, model.lam, model.sigma)
    
    # Compute full rate paths: r_t = alpha_t + x_t
    # new_alpha_curve: (n_steps + 1,)
    # x_paths_new: (old_n_paths, n_paths, n_steps + 1)
    # Broadcast new_alpha_curve: (1, 1, n_steps + 1)
    alpha_expanded = new_alpha_curve.view(1, 1, -1)
    r_paths_new = alpha_expanded + x_paths_new  # shape: (old_n_paths, n_paths, n_steps + 1)
    
    # Compute cumulative sum for discounting
    # dt_new: (n_steps,) -> (1, 1, n_steps)
    dt_expanded = dt_new.view(1, 1, -1)
    sum_r_dt_new = (dt_expanded * r_paths_new[:, :, :-1]).cumsum(-1)  # shape: (old_n_paths, n_paths, n_steps)
    
    # Compute OIS zero-coupon bonds
    # Average over n_paths dimension (which is at -2)
    ois_zcbs_new = torch.exp(-sum_r_dt_new).mean(-2)  # shape: (old_n_paths, n_steps)
    
    return HullWhiteModel(
        timeline=new_timeline,
        lam=model.lam,
        sigma=model.sigma,
        x0=x0_new,
        x_paths=x_paths_new,
        alpha_curve=new_alpha_curve,
        r_paths=r_paths_new,
        sum_r_dt=sum_r_dt_new,
        ois_zcbs=ois_zcbs_new
    )


def generate_hull_white_heston(
        n_paths: int,
        timeline: torch.Tensor,
        init_state: torch.Tensor,
        lam: torch.Tensor,
        var: torch.Tensor
) -> torch.Tensor:
    dt_steps = timeline.diff()
    n_steps = dt_steps.shape[0]
    # Paths shape: (*init_state.shape, n_paths, n_steps + 1) so time dimension is at -2
    # var should have shape (*init_state.shape, n_paths, n_steps + 1)
    expected_var_shape = (*init_state.shape, n_paths, n_steps + 1)
    assert var.shape == expected_var_shape, f"var shape {var.shape} does not match expected shape {expected_var_shape}"
    paths = torch.empty((*init_state.shape, n_paths, n_steps + 1), dtype=init_state.dtype)
    paths[..., 0] = init_state.unsqueeze(-1).expand(*init_state.shape, n_paths)
    
    for i in range(0, n_steps):
        x = paths[..., i].clone()
        Z = torch.randn_like(x)
        dt = dt_steps[i]
        sigma_2 = var[..., i]
        # Variance term: sigma_2 * (1 - exp(-2*lam*dt)) / (2*lam)
        variance_term = sigma_2 * (1 - torch.exp(- 2 * dt * lam)) / (2 * lam)
        x_next = x * torch.exp(- dt * lam) + Z * torch.sqrt(variance_term)
        paths[..., i+1] = x_next
    return paths


@dataclass
class HullWhiteHestonModel:
    timeline: torch.Tensor
    lam: torch.Tensor
    kappa: torch.Tensor
    theta: torch.Tensor
    eps: torch.Tensor
    x0: torch.Tensor
    v0: torch.Tensor
    x_paths: torch.Tensor
    v_paths: torch.Tensor
    alpha_curve: torch.Tensor
    r_paths: torch.Tensor 
    sum_r_dt: torch.Tensor
    ois_zcbs: torch.Tensor


def create_hull_white_heston_model(
    timeline: torch.Tensor,
    n_paths: int,
    ois_ifwd_curve: PchipSpline1D,
    r0: torch.Tensor,
    x0: torch.Tensor,
    lam: torch.Tensor,
    v0: torch.Tensor,
    kappa: torch.Tensor,
    theta: torch.Tensor,
    eps: torch.Tensor
) -> HullWhiteHestonModel:
    assert timeline.dim() == 1
    assert r0.dim() == 0
    assert x0.dim() == 0
    assert lam.dim() == 0
    assert v0.dim() == 0
    assert kappa.dim() == 0
    assert theta.dim() == 0
    assert eps.dim() == 0

    # Generate CIR variance paths
    v_paths = generate_cir(n_paths, timeline, v0, kappa, theta, eps, 1e-6)
    
    # Generate Hull-White paths with stochastic volatility
    x_paths = generate_hull_white_heston(n_paths, timeline, x0, lam, v_paths)
    
    # Use full timeline[1:] for ZCB maturities
    dt = timeline.diff()
    zcb_ttm = timeline[1:]
    zcbs0 = torch.exp(-(dt * x_paths[:, :-1]).cumsum(1)).mean(0)
    
    # Build forward curve from ZCBs
    r0_fwd = PchipSpline1D(zcb_ttm, -torch.log(zcbs0))
    
    alpha_curve = torch.zeros_like(timeline)

    alpha_curve[0] = r0
    alpha_curve[1:] = ois_ifwd_curve.derivative(timeline[1:]) - r0_fwd.derivative(timeline[1:])
    
    # Compute full rate paths: r_t = alpha_t + x_t
    r_paths = alpha_curve + x_paths
    
    # Compute cumulative sum for discounting
    sum_r_dt = (dt * r_paths[:, :-1]).cumsum(-1)
    
    # Compute OIS zero-coupon bonds
    ois_zcbs = torch.exp(-sum_r_dt).mean(-2)
   
    return HullWhiteHestonModel(
        timeline=timeline,
        lam=lam,
        kappa=kappa,
        theta=theta,
        eps=eps,
        x0=x0,
        v0=v0,
        x_paths=x_paths,
        v_paths=v_paths,
        alpha_curve=alpha_curve,
        r_paths=r_paths,
        sum_r_dt=sum_r_dt,
        ois_zcbs=ois_zcbs
    )


def hull_white_heston_model_asof(
    as_of: torch.Tensor,
    n_paths: int,
    model: HullWhiteHestonModel
) -> HullWhiteHestonModel:
    
    assert as_of.dim() == 0
    # Map as_of to timeline index using id_from_years
    as_of_id = id_from_years(as_of, model.timeline)
    
    # Get the snapshot of x_paths and v_paths at as_of_id
    # x_paths has shape (old_n_paths, n_steps + 1) when x0 is scalar
    # v_paths has shape (old_n_paths, n_steps + 1) when v0 is scalar
    x0_new = model.x_paths[:, as_of_id]  # shape: (old_n_paths,)
    v0_new = model.v_paths[:, as_of_id]  # shape: (old_n_paths,)
    
    # Use timeline[id:] for new_timeline
    new_timeline = model.timeline[as_of_id:]
    dt_new = new_timeline.diff()
    
    # Use alpha_curve[id:] for new_alpha_curve
    new_alpha_curve = model.alpha_curve[as_of_id:]
    
    # Branching:
    # v_paths_new will have shape (old_n_paths, n_paths, n_steps + 1)
    v_paths_new = generate_cir(n_paths, new_timeline, v0_new, model.kappa, model.theta, model.eps, 1e-6)
    
    # x_paths_new will have shape (old_n_paths, n_paths, n_steps + 1)
    x_paths_new = generate_hull_white_heston(n_paths, new_timeline, x0_new, model.lam, v_paths_new)
    
    # Compute full rate paths: r_t = alpha_t + x_t
    # new_alpha_curve: (n_steps + 1,)
    # x_paths_new: (old_n_paths, n_paths, n_steps + 1)
    # Broadcast new_alpha_curve: (1, 1, n_steps + 1)
    alpha_expanded = new_alpha_curve.view(1, 1, -1)
    r_paths_new = alpha_expanded + x_paths_new  # shape: (old_n_paths, n_paths, n_steps + 1)
    
    # Compute cumulative sum for discounting
    # dt_new: (n_steps,) -> (1, 1, n_steps)
    dt_expanded = dt_new.view(1, 1, -1)
    sum_r_dt_new = (dt_expanded * r_paths_new[:, :, :-1]).cumsum(-1)  # shape: (old_n_paths, n_paths, n_steps)
    
    # Compute OIS zero-coupon bonds
    # Average over n_paths dimension (which is at -2)
    ois_zcbs_new = torch.exp(-sum_r_dt_new).mean(-2)  # shape: (old_n_paths, n_steps)
    
    return HullWhiteHestonModel(
        timeline=new_timeline,
        lam=model.lam,
        kappa=model.kappa,
        theta=model.theta,
        eps=model.eps,
        x0=x0_new,
        v0=v0_new,
        x_paths=x_paths_new,
        v_paths=v_paths_new,
        alpha_curve=new_alpha_curve,
        r_paths=r_paths_new,
        sum_r_dt=sum_r_dt_new,
        ois_zcbs=ois_zcbs_new
    )


'''
# example
model_params = torch.tensor(
    [
        0.3, #v0 - 0
        0.01, #kappa - 1
        0.3, #theta - 2
        0.1, #epsilon - 3
        0., #x0 - 4
        1., #lam - 5
        0., #k0 - 6
        1., #gamma - 7
        0.01, #xi - 8  
    ]
)
'''

@dataclass
class KeyRateModel:
    timeline: torch.Tensor
    params: torch.Tensor # as model_params example above
    x_paths: torch.Tensor
    v_paths: torch.Tensor
    k_paths: torch.Tensor
    f_curve: torch.Tensor
    s_curve: torch.Tensor
    r_paths: torch.Tensor
    sum_r_dt: torch.Tensor
    s_paths: torch.Tensor
    sum_s_dt: torch.Tensor
    ois_zcbs: torch.Tensor
    key_zcbs: torch.Tensor



def create_key_rate_model(
    timeline: torch.Tensor,
    n_paths: int,
    key_ifwd_curve: PchipSpline1D,
    ois_ifwd_curve: PchipSpline1D,
    r0: torch.Tensor,
    model_params: torch.Tensor
) -> KeyRateModel:
    """
    Create a KeyRateModel from model parameters.
    
    Args:
        timeline: Time steps tensor
        n_paths: Number of simulation paths
        key_ifwd_curve: Key rate instantaneous forward curve
        ois_ifwd_curve: OIS instantaneous forward curve
        r0: Initial OIS rate
        model_params: Tensor with 9 parameters:
            [0] v0 - initial variance
            [1] kappa - CIR mean reversion speed
            [2] theta - CIR long-term variance
            [3] eps - CIR volatility of variance
            [4] x0 - initial OIS rate deviation
            [5] lam - Hull-White mean reversion speed for x
            [6] k0 - initial key rate spread deviation
            [7] gamma - Hull-White mean reversion speed for k
            [8] xi - Hull-White volatility for k
    """
    assert timeline.dim() == 1
    assert r0.dim() == 0
    assert model_params.dim() == 1
    assert model_params.shape[0] == 9
    
    # Extract parameters from model_params tensor
    v0 = model_params[0]
    kappa = model_params[1]
    theta = model_params[2]
    eps = model_params[3]
    x0 = model_params[4]
    lam = model_params[5]
    k0 = model_params[6]
    gamma = model_params[7]
    xi = model_params[8]

    # Generate CIR variance paths
    v_paths = generate_cir(n_paths, timeline, v0, kappa, theta, eps, 1e-9)
    
    # Generate Hull-White paths with stochastic volatility
    x_paths = generate_hull_white_heston(n_paths, timeline, x0, lam, v_paths)
    
    # Generate Hull-White paths for key rate spread
    k_paths = generate_hull_white(n_paths, timeline, k0, gamma, xi)

    # Use full timeline[1:] for ZCB maturities
    dt = timeline.diff()
    zcb_ttm = timeline[1:]
    
    # Compute ZCBs for OIS rate (r)
    zcbs_r = torch.exp(-(dt * x_paths[:, :-1]).cumsum(1)).mean(0)
    r_fwd = PchipSpline1D(zcb_ttm, -torch.log(zcbs_r))
    
    # Compute ZCBs for key rate (a = r + s)
    zcbs_a = torch.exp(-(dt * (x_paths + k_paths)[:, :-1]).cumsum(1)).mean(0)
    a_fwd = PchipSpline1D(zcb_ttm, -torch.log(zcbs_a))

    # Compute f_curve and s_curve
    # f_curve is the OIS forward adjustment
    f_curve = torch.zeros_like(timeline)
    f_curve[0] = r0
    f_curve[1:] = ois_ifwd_curve.derivative(timeline[1:]) - r_fwd.derivative(timeline[1:])
    
    # s_curve is the key rate spread forward adjustment
    s_curve = torch.zeros_like(timeline)
    s_curve[0] = 0.0
    s_curve[1:] = key_ifwd_curve.derivative(timeline[1:]) - a_fwd.derivative(timeline[1:]) - f_curve[1:]

    # Compute full rate paths: r_t = f_curve_t + x_t, s_t = s_curve_t + k_t
    r_paths = f_curve.unsqueeze(0) + x_paths
    s_paths = s_curve.unsqueeze(0) + k_paths

    # Compute cumulative sum for discounting
    sum_r_dt = (dt * r_paths[:, :-1]).cumsum(-1)
    sum_s_dt = (dt * s_paths[:, :-1]).cumsum(-1)
    
    # Compute OIS zero-coupon bonds
    ois_zcbs = torch.exp(-sum_r_dt).mean(-2)
    
    # Compute key rate zero-coupon bonds (r + s)
    key_zcbs = torch.exp(-(sum_r_dt + sum_s_dt)).mean(-2)
    
    return KeyRateModel(
        timeline=timeline,
        params=model_params,
        x_paths=x_paths,
        v_paths=v_paths,
        k_paths=k_paths,
        f_curve=f_curve,
        s_curve=s_curve,
        r_paths=r_paths,
        sum_r_dt=sum_r_dt,
        s_paths=s_paths,
        sum_s_dt=sum_s_dt,
        ois_zcbs=ois_zcbs,
        key_zcbs=key_zcbs
    )


def key_rate_model_asof(
    as_of: torch.Tensor,
    n_paths: int,
    model: KeyRateModel
) -> KeyRateModel:
    
    assert as_of.dim() == 0
    # Map as_of to timeline index using id_from_years
    as_of_id = id_from_years(as_of, model.timeline)
    
    # Get the snapshot of x_paths, v_paths, and k_paths at as_of_id
    # x_paths has shape (old_n_paths, n_steps + 1) when x0 is scalar
    # v_paths has shape (old_n_paths, n_steps + 1) when v0 is scalar
    # k_paths has shape (old_n_paths, n_steps + 1) when k0 is scalar
    x0_new = model.x_paths[:, as_of_id]  # shape: (old_n_paths,)
    v0_new = model.v_paths[:, as_of_id]  # shape: (old_n_paths,)
    k0_new = model.k_paths[:, as_of_id]  # shape: (old_n_paths,)
    
    # Use timeline[id:] for new_timeline
    new_timeline = model.timeline[as_of_id:]
    dt_new = new_timeline.diff()
    
    # Use f_curve and s_curve[id:] for new curves
    new_f_curve = model.f_curve[as_of_id:]
    new_s_curve = model.s_curve[as_of_id:]
    
    # Extract model parameters
    kappa = model.params[1]
    theta = model.params[2]
    eps = model.params[3]
    lam = model.params[5]
    gamma = model.params[7]
    xi = model.params[8]
    
    # Branching:
    # v_paths_new will have shape (old_n_paths, n_paths, n_steps + 1)
    v_paths_new = generate_cir(n_paths, new_timeline, v0_new, kappa, theta, eps, 1e-9)
    
    # x_paths_new will have shape (old_n_paths, n_paths, n_steps + 1)
    x_paths_new = generate_hull_white_heston(n_paths, new_timeline, x0_new, lam, v_paths_new)
    
    # k_paths_new will have shape (old_n_paths, n_paths, n_steps + 1)
    k_paths_new = generate_hull_white(n_paths, new_timeline, k0_new, gamma, xi)
    
    # Compute full rate paths: r_t = f_curve_t + x_t, s_t = s_curve_t + k_t
    # new_f_curve: (n_steps + 1,)
    # new_s_curve: (n_steps + 1,)
    # x_paths_new: (old_n_paths, n_paths, n_steps + 1)
    # k_paths_new: (old_n_paths, n_paths, n_steps + 1)
    # Broadcast curves: (1, 1, n_steps + 1)
    f_curve_expanded = new_f_curve.view(1, 1, -1)
    s_curve_expanded = new_s_curve.view(1, 1, -1)
    r_paths_new = f_curve_expanded + x_paths_new  # shape: (old_n_paths, n_paths, n_steps + 1)
    s_paths_new = s_curve_expanded + k_paths_new  # shape: (old_n_paths, n_paths, n_steps + 1)
    
    # Compute cumulative sum for discounting
    # dt_new: (n_steps,) -> (1, 1, n_steps)
    dt_expanded = dt_new.view(1, 1, -1)
    sum_r_dt_new = (dt_expanded * r_paths_new[:, :, :-1]).cumsum(-1)  # shape: (old_n_paths, n_paths, n_steps)
    sum_s_dt_new = (dt_expanded * s_paths_new[:, :, :-1]).cumsum(-1)  # shape: (old_n_paths, n_paths, n_steps)
    
    # Compute OIS zero-coupon bonds
    # Average over n_paths dimension (which is at -2)
    ois_zcbs_new = torch.exp(-sum_r_dt_new).mean(-2)  # shape: (old_n_paths, n_steps)
    
    # Compute key rate zero-coupon bonds (r + s)
    key_zcbs_new = torch.exp(-(sum_r_dt_new + sum_s_dt_new)).mean(-2)  # shape: (old_n_paths, n_steps)
    
    return KeyRateModel(
        timeline=new_timeline,
        params=model.params,
        x_paths=x_paths_new,
        v_paths=v_paths_new,
        k_paths=k_paths_new,
        f_curve=new_f_curve,
        s_curve=new_s_curve,
        r_paths=r_paths_new,
        sum_r_dt=sum_r_dt_new,
        s_paths=s_paths_new,
        sum_s_dt=sum_s_dt_new,
        ois_zcbs=ois_zcbs_new,
        key_zcbs=key_zcbs_new
    )


def price_now_starting_avg_caplet(K, T , model):
    assert T.dim() == 0
    assert K.dim() == 0 
    T_id = id_from_years(T, model.timeline)
    A_T = model.sum_r_dt[:, T_id - 1]
    inv_B_T = torch.exp(-A_T)
    payoff = torch.clamp(A_T - T * K, min=1e-6) * inv_B_T
    return payoff.mean()


def price_key_caplet_surface(model: KeyRateModel, vol_key_rate, fwd_key_rate):
    vol_ids = id_from_years(torch.tensor(vol_key_rate.time_to_maturity.values), model.timeline)-1 
    tau_strikes = model.timeline[vol_ids] * torch.tensor(vol_key_rate.strike.values) 
    market_pvs = torch.tensor(vol_key_rate.pv.values)
    model_pvs = torch.zeros_like(market_pvs)

    for i in range(model_pvs.numel()):
        T = vol_ids[i]
        tauK = tau_strikes[i]
        inv_B_T = torch.exp(-model.sum_r_dt[:, T]) 
        payoff = torch.clamp(model.sum_r_dt[:, T] + model.sum_s_dt[:, T] - tauK, min=1e-6) 
        model_pvs[i] = torch.mean(payoff * inv_B_T)

    vol_key_rate['pv_model_key'] = model_pvs.detach().numpy()

    key_fwd_ids = id_from_years(torch.tensor(fwd_key_rate.time_to_maturity.values), model.timeline)-1
    market_key_fwd = torch.tensor(fwd_key_rate.forward_rate.values)
    model_key_fwd = torch.zeros_like(market_key_fwd)
    
    for i in range(model_key_fwd.numel()):
        T = key_fwd_ids[i]
        if T <= 0:
            T = 1
        tau = model.timeline[T]
        inv_B_T = torch.exp(-model.sum_r_dt[:, T]) 
        A_T = (model.sum_r_dt[:, T] + model.sum_s_dt[:, T]) / tau
        model_key_fwd[i] = torch.mean(A_T * inv_B_T) / torch.mean(inv_B_T)

    fwd_key_rate['fwd_model_key'] = model_key_fwd.detach().numpy()

    return torch.sum((model_pvs - market_pvs) ** 2), torch.sum((market_key_fwd - model_key_fwd) ** 2)
 

def calibrate_caplet_key_surface(data_dir: str, n_paths: int):
    fwd_ois = pd.read_csv(os.path.join(data_dir, 'forward_ois.csv'))
    fwd_key_rate = pd.read_csv(os.path.join(data_dir, 'forward_key_rate.csv'))
    vol_key_rate = pd.read_csv(os.path.join(data_dir, 'volatility_key_rate.csv'))


    ois_yield_curve = build_ois_yield_curve_from_now_starting(
        torch.tensor(fwd_ois.forward_rate.values),
        torch.tensor(fwd_ois.tenor.values)
    )
    ois_ifwd_curve = build_ifwd_curve_from_now_starting(
        torch.tensor(fwd_ois.forward_rate.values),
        torch.tensor(fwd_ois.tenor.values)
    )

    r0 = torch.tensor(fwd_ois.forward_rate.values[0])
    timeline = torch.linspace(0, 10., 3651) # 10 years Actual/365 day

    key_rate_fwd_curve = build_fwd_curve(
        torch.tensor(fwd_key_rate.forward_rate.values),
        torch.tensor(fwd_key_rate.time_to_maturity.values))

    key_ifwd_values = torch.tensor(fwd_key_rate.forward_rate.values, requires_grad=True)
    key_ifwd_curve = build_ifwd_key_curve_from_now_starting(
        key_ifwd_values, torch.tensor(fwd_key_rate.forward_rate.values), torch.tensor(fwd_key_rate.tenor.values))

    vol_key_rate['pv'] = caplet_premium_from_now_starting(vol_key_rate, key_rate_fwd_curve, ois_yield_curve).numpy()

    model_params = torch.tensor(
        [
            0.3, #v0 - 0
            0.01, #kappa - 1
            0.3, #theta - 2
            0.1, #epsilon - 3
            0., #x0 - 4
            2., #lam - 5
            0., #k0 - 6
            1., #gamma - 7
            0.01, #xi - 8  
        ],
        requires_grad=True
    )

    # Initial model creation
    model = create_key_rate_model(timeline, n_paths, key_ifwd_curve, ois_ifwd_curve, r0, model_params)
    loss_vol, loss_fwd = price_key_caplet_surface(model, vol_key_rate, fwd_key_rate)

    prev_loss = -1
    for i in range(2, 12):
        learning_rate = 1 / i

        loss = loss_vol + 0.001 * loss_fwd
        print(f'loss = {loss.item()}')
        if prev_loss > 0 and not (loss <= prev_loss):
            break
            
        grad_error = torch.autograd.grad(loss, [model_params, key_ifwd_values])
        prev_loss = loss.detach()

        key_ifwd_values = key_ifwd_values.detach() + learning_rate * grad_error[1]
        key_ifwd_values.requires_grad_()
        key_ifwd_curve = build_ifwd_key_curve_from_now_starting(
            key_ifwd_values, torch.tensor(fwd_key_rate.forward_rate.values), torch.tensor(fwd_key_rate.tenor.values))

        model_params = model_params.detach() + learning_rate * grad_error[0]
        model_params.requires_grad_()
        model = create_key_rate_model(timeline, n_paths, key_ifwd_curve, ois_ifwd_curve, r0, model_params)

        loss_vol, loss_fwd = price_key_caplet_surface(model, vol_key_rate, fwd_key_rate)


@click.command()
@click.argument('data_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--n_paths', type=int, default=1000, help='Number of simulation paths')
def main(data_dir: str, n_paths: int):
    """Calibrate caplet key surface from data directory.
    
    DATA_DIR: Path to directory containing forward_ois.csv, forward_key_rate.csv, and volatility_key_rate.csv
    """
    calibrate_caplet_key_surface(data_dir, n_paths)


if __name__ == '__main__':
    main()

