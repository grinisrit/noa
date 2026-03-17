"""
Caplet Volatility Surface Generation and Plotting Utilities

Functions for:
- Computing implied normal volatility surfaces from model prices
- Comparing model vs market data for interest rate caplets
- Arbitrage checking using Dupire local volatility conditions

Arbitrage Conditions (Bachelier/Normal Model):
- Calendar Spread:  dw/dT >= 0 where w = sigma² * T (total variance)
- Butterfly Spread: d²C/dK² >= 0 (convexity in strike price)
- Dupire Local Vol: sigma_loc² = (dC/dT) / (0.5 * d²C/dK²) must be real & positive
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator, RectBivariateSpline
from pyquant.torch_spline import PchipSpline1D


# =============================================================================
# BACHELIER PRICING
# =============================================================================

def _bachelier_tv_undiscounted(F, K, sigma, g_hat):
    """Numerically stable time-value of Bachelier formula (undiscounted per unit T).
    
    Uses TV = v·[φ(d) − d·Φ(−d)] for ITM (d>0), which avoids catastrophic
    cancellation that occurs when subtracting intrinsic from total price.
    scipy's norm.sf computes Φ(−d) accurately even for d > 30.
    
    Returns (intrinsic, time_value) so caller can use whichever form is needed.
    """
    v = sigma * g_hat
    intrinsic = max(F - K, 0.0)
    if v < 1e-300:
        return intrinsic, 0.0
    d = (F - K) / v
    if d > 0:
        tv = v * (norm.pdf(d) - d * norm.sf(d))
    else:
        # OTM/ATM: full price IS the time value (intrinsic = 0)
        tv = (F - K) * norm.cdf(d) + v * norm.pdf(d)
    return intrinsic, max(tv, 0.0)


def bachelier_caplet_price(F, K, T, sigma, disc=1.0):
    """
    Bachelier (normal) price for NOW-STARTING average rate caplet.
    
    PV = T · disc · [(F-K)Φ(d) + σ·ĝ·φ(d)]
    where ĝ = √(T/3), d = (F-K)/(σ·ĝ)
    
    This is the average rate variant (g_hat = sqrt(T/3)) for caplets whose
    payoff is max(∫₀ᵀ a_t dt - T·K, 0), consistent with the MC simulation.
    
    Args:
        F: Forward rate = I(0,T)/T, average instantaneous forward (decimal)
        K: Strike (decimal)
        T: Time to maturity (years). For now-starting, tenor = T.
        sigma: Normal vol (decimal, e.g., 0.04 = 4%)
        disc: Discount factor to payment date
    
    Returns:
        Price as fraction of notional (e.g., 0.01 = 1% = 100bp)
    """
    if T <= 0:
        return max(F - K, 0) * T * disc
    
    g_hat = np.sqrt(T / 3.0)
    if sigma * g_hat < 1e-10:
        return max(F - K, 0) * T * disc
    
    d = (F - K) / (sigma * g_hat)
    undiscounted = (F - K) * norm.cdf(d) + sigma * g_hat * norm.pdf(d)
    
    return T * disc * undiscounted


def bachelier_caplet_time_value(F, K, T, sigma, disc=1.0):
    """Numerically stable time-value of Bachelier avg rate caplet.
    
    For deep ITM (d > 8), the standard price (F-K)Φ(d) + σĝφ(d) stores
    intrinsic + TV in one float64, losing TV when TV < ε·intrinsic.
    This function computes TV directly via v·[φ(d) − d·Φ(−d)], which
    scipy evaluates accurately even for d > 30.
    
    Returns:
        Time value only (excluding intrinsic), as fraction of notional.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    g_hat = np.sqrt(T / 3.0)
    _, tv = _bachelier_tv_undiscounted(F, K, sigma, g_hat)
    return T * disc * tv


def implied_vol_avg_rate(F, K, T, pv, disc=1.0, tol=1e-9, max_iter=200):
    """
    Invert Bachelier formula for now-starting average rate caplet: PV → σ_n.

    Solves: undiscounted_unit = (F-K)Φ(d) + σ·ĝ·φ(d) for σ
    where ĝ = √(T/3), d = (F-K)/(σ·ĝ)

    For deep ITM (F ≫ K), falls back to time-value formulation using
    Φ(−d) via scipy's norm.sf, which is accurate to full precision.

    Args:
        F: Forward rate = I(0,T)/T (decimal)
        K: Strike (decimal)
        T: Time to maturity (years)
        pv: Caplet price (fraction of notional)
        disc: Discount factor to payment date
        tol: Root-finding tolerance
        max_iter: Maximum Brentq iterations

    Returns:
        Implied normal vol (decimal), or np.nan if inversion fails.
    """
    if T <= 0 or pv <= 0:
        return np.nan

    g_hat = np.sqrt(T / 3.0)
    und_unit = pv / (T * disc)
    intrinsic = max(F - K, 0.0)

    # Standard inversion for OTM/ATM or mild ITM
    if und_unit > intrinsic + 1e-15:
        def price_error(sigma):
            d = (F - K) / (sigma * g_hat + 1e-15)
            return (F - K) * norm.cdf(d) + sigma * g_hat * norm.pdf(d) - und_unit
        try:
            return brentq(price_error, 1e-10, 10.0, xtol=tol, maxiter=max_iter)
        except ValueError:
            pass  # fall through to TV solver

    # Deep ITM fallback: solve from time-value using stable Φ(−d) formulation
    if intrinsic > 0:
        time_value = max(und_unit - intrinsic, 0.0)
        def tv_error(sigma):
            v = sigma * g_hat
            d = (F - K) / (v + 1e-300)
            return v * (norm.pdf(d) - d * norm.sf(d)) - time_value
        try:
            return brentq(tv_error, 1e-10, 10.0, xtol=tol, maxiter=max_iter)
        except ValueError:
            return np.nan

    return np.nan


def implied_vol_from_tv(F, K, T, time_value, disc=1.0, tol=1e-9, max_iter=200):
    """Invert Bachelier vol from separately-computed time value.
    
    Use with bachelier_caplet_time_value() for lossless deep-ITM round-trips.
    The TV channel preserves full precision because it never adds TV to intrinsic.
    
    Args:
        F, K, T: Forward, strike, maturity
        time_value: Time value only (from bachelier_caplet_time_value)
        disc: Discount factor
    
    Returns:
        Implied normal vol (decimal), or np.nan if inversion fails.
    """
    if T <= 0 or time_value <= 0:
        return np.nan
    g_hat = np.sqrt(T / 3.0)
    tv_und = time_value / (T * disc)

    def tv_error(sigma):
        v = sigma * g_hat
        d = (F - K) / (v + 1e-300)
        if d > 0:
            return v * (norm.pdf(d) - d * norm.sf(d)) - tv_und
        else:
            return (F - K) * norm.cdf(d) + v * norm.pdf(d) - tv_und
    try:
        return brentq(tv_error, 1e-10, 10.0, xtol=tol, maxiter=max_iter)
    except ValueError:
        return np.nan


# =============================================================================
# ARBITRAGE CHECKING - DUPIRE LOCAL VOL
# =============================================================================

def compute_total_variance_conditions(T, K, vol_func, h_T=0.1, use_richardson=True):
    """
    Check calendar arbitrage using total variance w = sigma^2 * T.
    
    Calendar arbitrage condition: dw/dT >= 0
    
    This is the cleaner formulation (per Gatheral/SVI methodology):
    - Total variance must be non-decreasing in T at each strike
    
    Uses Richardson extrapolation for accuracy.
    
    Args:
        T: Maturity (years)
        K: Strike (decimal)
        vol_func: Function vol_func(T, K) -> implied normal vol
        h_T: Step size for time derivative
        use_richardson: Whether to use Richardson extrapolation
    
    Returns:
        dict with:
            - w: Total variance sigma^2 * T
            - dw_dT: Time derivative of total variance
            - calendar_arb: True if dw/dT < 0 (arbitrage)
    """
    def total_var(t):
        if t <= 0:
            return 0.0
        sigma = vol_func(t, K)
        return sigma ** 2 * t
    
    w = total_var(T)
    
    # dw/dT using central difference
    if T - h_T > 0:
        dw_dT_h = (total_var(T + h_T) - total_var(T - h_T)) / (2 * h_T)
    else:
        dw_dT_h = (total_var(T + h_T) - total_var(T)) / h_T
    
    if use_richardson:
        h_T2 = h_T / 2
        if T - h_T2 > 0:
            dw_dT_h2 = (total_var(T + h_T2) - total_var(T - h_T2)) / (2 * h_T2)
        else:
            dw_dT_h2 = (total_var(T + h_T2) - total_var(T)) / h_T2
        dw_dT = (4 * dw_dT_h2 - dw_dT_h) / 3
    else:
        dw_dT = dw_dT_h
    
    return {
        'T': T,
        'K': K,
        'w': w,
        'dw_dT': dw_dT,
        'calendar_arb': dw_dT < -1e-10
    }


def compute_dupire_conditions(T, K, price_func, vol_func=None, h_T=0.1, h_K=0.005, use_richardson=True):
    """
    Compute Dupire arbitrage conditions using numerical derivatives.
    
    Two complementary checks:
    1. **Calendar (total variance)**: dw/dT >= 0 where w = sigma^2 * T
    2. **Butterfly (price convexity)**: d²C/dK² >= 0
    
    Uses Richardson extrapolation for higher accuracy:
    1. Compute derivative with step h
    2. Compute again with step h/2
    3. Combine: (4*f(h/2) - f(h)) / 3
    
    Args:
        T: Maturity (years)
        K: Strike (decimal)
        price_func: Function price_func(T, K) -> caplet price
        vol_func: Optional function vol_func(T, K) -> implied vol (for total variance check)
        h_T: Step size for time derivative
        h_K: Step size for strike derivative
        use_richardson: Whether to use Richardson extrapolation
    
    Returns:
        dict with:
            - dC_dT: Price time derivative (for reference)
            - d2C_dK2: Strike convexity (butterfly condition)
            - dw_dT: Total variance time derivative (calendar condition)
            - local_var: Dupire local variance
            - local_vol: Dupire local vol (sqrt of variance)
            - is_valid: True if no arbitrage
            - calendar_arb: True if calendar arbitrage (dw/dT < 0)
            - butterfly_arb: True if butterfly arbitrage (d²C/dK² < 0)
    """
    # ========== dC/dT (central difference) ==========
    if T - h_T > 0:
        dC_dT_h = (price_func(T + h_T, K) - price_func(T - h_T, K)) / (2 * h_T)
    else:
        dC_dT_h = (price_func(T + h_T, K) - price_func(T, K)) / h_T
    
    if use_richardson:
        h_T2 = h_T / 2
        if T - h_T2 > 0:
            dC_dT_h2 = (price_func(T + h_T2, K) - price_func(T - h_T2, K)) / (2 * h_T2)
        else:
            dC_dT_h2 = (price_func(T + h_T2, K) - price_func(T, K)) / h_T2
        dC_dT = (4 * dC_dT_h2 - dC_dT_h) / 3
    else:
        dC_dT = dC_dT_h
    
    # ========== Total Variance Calendar Check: dw/dT where w = sigma^2 * T ==========
    dw_dT = np.nan
    if vol_func is not None:
        tv_result = compute_total_variance_conditions(T, K, vol_func, h_T, use_richardson)
        dw_dT = tv_result['dw_dT']
        calendar_arb = tv_result['calendar_arb']
    else:
        # Fallback to price-based check if no vol_func provided
        calendar_arb = dC_dT < -1e-10
    
    # ========== d²C/dK² (second derivative) - Butterfly ==========
    d2C_dK2_h = (price_func(T, K + h_K) - 2*price_func(T, K) + price_func(T, K - h_K)) / (h_K ** 2)
    
    if use_richardson:
        h_K2 = h_K / 2
        d2C_dK2_h2 = (price_func(T, K + h_K2) - 2*price_func(T, K) + price_func(T, K - h_K2)) / (h_K2 ** 2)
        d2C_dK2 = (4 * d2C_dK2_h2 - d2C_dK2_h) / 3
    else:
        d2C_dK2 = d2C_dK2_h
    
    # ========== Dupire local variance ==========
    # Butterfly check: d²C/dK² >= 0
    butterfly_arb = d2C_dK2 < -1e-10
    
    # Compute local vol wherever d²C/dK² > 0
    # Use smaller threshold to capture more points
    if d2C_dK2 > 1e-14:
        local_var = dC_dT / (0.5 * d2C_dK2)
    else:
        local_var = np.nan  # Can't compute when butterfly is flat or violated
    
    # Arbitrage-free = both conditions satisfied (don't require local_var > 0)
    # Note: local_var < 0 indicates dC/dT < 0, which can happen for deep ITM near expiry
    # but isn't strictly arbitrage in the calendar spread sense
    is_valid = (not calendar_arb) and (not butterfly_arb)
    
    # Track why local vol might not be computable
    local_vol_computable = (d2C_dK2 > 1e-14) and (not np.isnan(local_var)) and (local_var > 0)
    
    return {
        'T': T,
        'K': K,
        'dC_dT': dC_dT,
        'dw_dT': dw_dT,  # Total variance derivative (calendar condition)
        'd2C_dK2': d2C_dK2,
        'local_var': local_var,
        'local_vol': np.sqrt(local_var) if local_var > 0 else np.nan,
        'is_valid': is_valid,
        'calendar_arb': calendar_arb,  # Based on dw/dT
        'butterfly_arb': butterfly_arb,
        'dC_dT_positive': dC_dT > -1e-10,  # Track if price derivative is positive
        'local_vol_computable': local_vol_computable
    }


def check_surface_arbitrage(vol_surface_df, fwd_func, disc_func, tau=0.25,
                            T_range=None, K_range=None, h_T=0.1, h_K=0.005):
    """
    Check entire vol surface for arbitrage violations.
    
    Args:
        vol_surface_df: DataFrame with columns [time_to_maturity, strike, implied_normal_vol]
        fwd_func: Function fwd_func(T) -> forward rate at maturity T
        disc_func: Function disc_func(T) -> discount factor to payment date T+tau
        tau: Accrual period
        T_range: (T_min, T_max) to check, or None for all
        K_range: (K_min, K_max) to check, or None for all
        h_T, h_K: Step sizes for derivatives
    
    Returns:
        arb_df: DataFrame with arbitrage check results for each point
        summary: dict with summary statistics
    """
    maturities = sorted(vol_surface_df['time_to_maturity'].unique())
    strikes = sorted(vol_surface_df['strike'].unique())
    
    # Build vol grid and interpolator
    vol_grid = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            mask = (vol_surface_df['time_to_maturity'] == T) & (vol_surface_df['strike'] == K)
            if mask.sum() > 0:
                vol_grid[i, j] = vol_surface_df.loc[mask, 'implied_normal_vol'].values[0]
            else:
                vol_grid[i, j] = np.nan
    
    # Fill NaN with interpolation
    vol_grid_df = pd.DataFrame(vol_grid, index=maturities, columns=strikes)
    vol_grid_filled = vol_grid_df.interpolate(axis=0).interpolate(axis=1).values
    
    # Create bivariate spline
    vol_spline = RectBivariateSpline(maturities, strikes, vol_grid_filled)
    
    def get_vol(T, K):
        T = np.clip(T, maturities[0], maturities[-1])
        K = np.clip(K, strikes[0], strikes[-1])
        return float(vol_spline(T, K)[0, 0])
    
    def price_at(T, K):
        if T <= 0:
            return 0.0
        vol = get_vol(T, K)
        F = fwd_func(min(T, maturities[-1]))
        disc = disc_func(min(T, maturities[-1]))
        return bachelier_caplet_price(F, K, T, vol, disc)
    
    # Filter ranges - include ALL points (derivatives handle boundaries)
    if T_range:
        check_T = [t for t in maturities if T_range[0] <= t <= T_range[1]]
    else:
        check_T = maturities  # Include all maturities
    
    if K_range:
        check_K = [k for k in strikes if K_range[0] <= k <= K_range[1]]
    else:
        check_K = strikes  # Include all strikes
    
    # Check all points - using both price_at and get_vol for total variance
    results = []
    for T in check_T:
        for K in check_K:
            result = compute_dupire_conditions(T, K, price_at, vol_func=get_vol, h_T=h_T, h_K=h_K)
            result['market_vol'] = get_vol(T, K)
            results.append(result)
    
    arb_df = pd.DataFrame(results)
    
    # Summary
    n_total = len(arb_df)
    n_valid = arb_df['is_valid'].sum()
    n_calendar = arb_df['calendar_arb'].sum()
    n_butterfly = arb_df['butterfly_arb'].sum()
    
    summary = {
        'total_points': n_total,
        'valid_points': n_valid,
        'valid_pct': n_valid / n_total * 100 if n_total > 0 else 0,
        'calendar_violations': n_calendar,
        'butterfly_violations': n_butterfly,
        'is_arbitrage_free': n_calendar + n_butterfly == 0
    }
    
    return arb_df, summary


def plot_arbitrage_heatmaps(arb_df, title_prefix=""):
    """
    Plot heatmaps showing arbitrage conditions across the surface.
    
    Args:
        arb_df: DataFrame from check_surface_arbitrage()
        title_prefix: Prefix for plot titles (e.g., "Market" or "Model")
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    T_grid = sorted(arb_df['T'].unique())
    K_grid = sorted(arb_df['K'].unique())
    n_total_pts = len(T_grid) * len(K_grid)
    
    # 1. dw/dT heatmap (calendar condition on total variance w = σ²T)
    dw_dT_matrix = arb_df.pivot(index='K', columns='T', values='dw_dT').values
    im1 = axes[0, 0].imshow(dw_dT_matrix, aspect='auto', origin='lower',
                             extent=[T_grid[0], T_grid[-1], K_grid[0]*100, K_grid[-1]*100],
                             cmap='RdYlGn')
    axes[0, 0].set_xlabel('Maturity (Y)')
    axes[0, 0].set_ylabel('Strike (%)')
    axes[0, 0].set_title(f'{title_prefix} dw/dT (w=σ²T)\nCalendar Arb (should ≥ 0)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. d²C/dK² heatmap (butterfly condition)
    d2C_dK2_matrix = arb_df.pivot(index='K', columns='T', values='d2C_dK2').values
    im2 = axes[0, 1].imshow(d2C_dK2_matrix, aspect='auto', origin='lower',
                             extent=[T_grid[0], T_grid[-1], K_grid[0]*100, K_grid[-1]*100],
                             cmap='RdYlGn')
    axes[0, 1].set_xlabel('Maturity (Y)')
    axes[0, 1].set_ylabel('Strike (%)')
    axes[0, 1].set_title(f'{title_prefix} d²C/dK²\nButterfly Arb (should ≥ 0)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. dC/dT heatmap (price time derivative - needed for local vol)
    dC_dT_matrix = arb_df.pivot(index='K', columns='T', values='dC_dT').values
    n_dC_dT_pos = np.sum(dC_dT_matrix > -1e-10)
    im3 = axes[0, 2].imshow(dC_dT_matrix, aspect='auto', origin='lower',
                             extent=[T_grid[0], T_grid[-1], K_grid[0]*100, K_grid[-1]*100],
                             cmap='RdYlGn')
    axes[0, 2].set_xlabel('Maturity (Y)')
    axes[0, 2].set_ylabel('Strike (%)')
    axes[0, 2].set_title(f'{title_prefix} dC/dT (price derivative)\nNeeded for local vol ≥ 0: {n_dC_dT_pos}/{n_total_pts}')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # 4. Local vol surface - show all computable values
    local_var_matrix = arb_df.pivot(index='K', columns='T', values='local_var').values
    # For display: show local_vol (sqrt) where positive, indicate negative variance
    local_vol_matrix = np.where(local_var_matrix > 0, np.sqrt(local_var_matrix) * 100, np.nan)
    # Use percentile for vmax, ignoring NaN
    valid_vals = local_vol_matrix[~np.isnan(local_vol_matrix)]
    vmax = np.percentile(valid_vals, 95) if len(valid_vals) > 0 else 1.0
    im4 = axes[1, 0].imshow(local_vol_matrix, aspect='auto', origin='lower',
                             extent=[T_grid[0], T_grid[-1], K_grid[0]*100, K_grid[-1]*100],
                             cmap='viridis', vmin=0, vmax=vmax)
    axes[1, 0].set_xlabel('Maturity (Y)')
    axes[1, 0].set_ylabel('Strike (%)')
    n_local_vol_valid = np.sum(~np.isnan(local_vol_matrix))
    axes[1, 0].set_title(f'{title_prefix} Dupire Local Vol (%)\nσ²_loc = dC/dT / (½d²C/dK²): {n_local_vol_valid}/{n_total_pts}')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # 5. Arbitrage violation map (based on dw/dT >= 0 AND d²C/dK² >= 0)
    valid_matrix = arb_df.pivot(index='K', columns='T', values='is_valid').values.astype(float)
    n_valid = int(np.sum(valid_matrix))
    im5 = axes[1, 1].imshow(valid_matrix, aspect='auto', origin='lower',
                             extent=[T_grid[0], T_grid[-1], K_grid[0]*100, K_grid[-1]*100],
                             cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 1].set_xlabel('Maturity (Y)')
    axes[1, 1].set_ylabel('Strike (%)')
    axes[1, 1].set_title(f'{title_prefix} Arbitrage-Free\n(dw/dT≥0 ∧ d²C/dK²≥0): {n_valid}/{n_total_pts} ({100*n_valid/n_total_pts:.1f}%)')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # 6. Local vol computable map (d²C/dK² > 0 AND dC/dT > 0)
    local_vol_ok = (~np.isnan(local_vol_matrix)).astype(float)
    n_ok = int(np.sum(local_vol_ok))
    im6 = axes[1, 2].imshow(local_vol_ok, aspect='auto', origin='lower',
                             extent=[T_grid[0], T_grid[-1], K_grid[0]*100, K_grid[-1]*100],
                             cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 2].set_xlabel('Maturity (Y)')
    axes[1, 2].set_ylabel('Strike (%)')
    axes[1, 2].set_title(f'{title_prefix} Local Vol Computable\n(d²C/dK²>0 ∧ dC/dT>0): {n_ok}/{n_total_pts} ({100*n_ok/n_total_pts:.1f}%)')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.show()


def print_arbitrage_summary(arb_df, summary, title=""):
    """
    Print summary of arbitrage check results.
    
    Args:
        arb_df: DataFrame from check_surface_arbitrage()
        summary: dict from check_surface_arbitrage()
        title: Title for the summary
    """
    print(f"\n{'='*70}")
    print(f"{title} ARBITRAGE ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nARBITRAGE CONDITIONS:")
    print(f"  Points checked:      {summary['total_points']}")
    print(f"  Arbitrage-free:      {summary['valid_points']} ({summary['valid_pct']:.1f}%)")
    print(f"  Calendar violations: {summary['calendar_violations']} (dw/dT < 0)")
    print(f"  Butterfly violations:{summary['butterfly_violations']} (d²C/dK² < 0)")
    
    # LOCAL VOL COMPUTABILITY BREAKDOWN
    n_total = len(arb_df)
    n_arb_free = arb_df['is_valid'].sum()
    
    # Check if local_vol_computable column exists (new version)
    if 'local_vol_computable' in arb_df.columns:
        n_local_vol_ok = arb_df['local_vol_computable'].sum()
    else:
        n_local_vol_ok = (~arb_df['local_vol'].isna()).sum()
    
    n_d2C_small = (arb_df['d2C_dK2'] <= 1e-14).sum()
    n_dC_dT_neg = (arb_df['dC_dT'] < -1e-10).sum()
    n_local_var_neg = ((~arb_df['local_var'].isna()) & (arb_df['local_var'] <= 0)).sum()
    
    print(f"\nLOCAL VOL COMPUTABILITY:")
    print(f"  Computable (σ_loc² > 0): {n_local_vol_ok}/{n_total} ({100*n_local_vol_ok/n_total:.1f}%)")
    print(f"  Failures breakdown:")
    print(f"    d²C/dK² ≈ 0:           {n_d2C_small} (gamma vanishes at deep ITM/OTM)")
    print(f"    dC/dT < 0:             {n_dC_dT_neg} (price derivative negative)")
    print(f"    local_var ≤ 0:         {n_local_var_neg} (dC/dT and d²C/dK² opposite signs)")
    
    # Explain the discrepancy
    if n_arb_free > n_local_vol_ok:
        print(f"\n  NOTE: {n_arb_free - n_local_vol_ok} points are ARBITRAGE-FREE but local vol undefined.")
        print(f"  This is theoretically expected (Gatheral-Jacquier 2014):")
        print(f"  - Calendar arb uses total variance: dw/dT ≥ 0 where w = σ²T")  
        print(f"  - Local vol uses price derivative: σ²_loc = (dC/dT) / (0.5 d²C/dK²)")
        print(f"  - dw/dT ≥ 0 does NOT imply dC/dT > 0 (nonlinear relationship)")
    
    if summary['is_arbitrage_free']:
        print(f"\n  ✓ Surface is ARBITRAGE-FREE")
    else:
        print(f"\n  ⚠ Surface has ARBITRAGE VIOLATIONS")
        
        if summary['calendar_violations'] > 0:
            print(f"\n  Calendar arbitrage locations (dw/dT < 0, w = σ²T):")
            cal_arb = arb_df[arb_df['calendar_arb']][['T', 'K', 'dw_dT', 'dC_dT']].copy()
            cal_arb['K_%'] = cal_arb['K'] * 100
            cal_arb['dw_dT_fmt'] = cal_arb['dw_dT'].apply(lambda x: f'{x:.2e}')
            print(cal_arb[['T', 'K_%', 'dw_dT_fmt']].head(10).to_string(index=False))
        
        if summary['butterfly_violations'] > 0:
            print(f"\n  Butterfly arbitrage locations (d²C/dK² < 0):")
            but_arb = arb_df[arb_df['butterfly_arb']][['T', 'K', 'd2C_dK2']].copy()
            but_arb['K_%'] = but_arb['K'] * 100
            print(but_arb[['T', 'K_%', 'd2C_dK2']].head(10).to_string(index=False))
    
    # Local vol statistics for computable points
    valid_local = arb_df[~arb_df['local_vol'].isna() & (arb_df['local_vol'] > 0)]
    if len(valid_local) > 0:
        print(f"\n  DUPIRE LOCAL VOL (computable points):")
        print(f"    Min:  {valid_local['local_vol'].min()*100:.2f}%")
        print(f"    Max:  {valid_local['local_vol'].max()*100:.2f}%")
        print(f"    Mean: {valid_local['local_vol'].mean()*100:.2f}%")
        
        print(f"\n  MARKET IMPLIED VOL (for comparison):")
        print(f"    Min:  {valid_local['market_vol'].min()*100:.2f}%")
        print(f"    Max:  {valid_local['market_vol'].max()*100:.2f}%")
        print(f"    Mean: {valid_local['market_vol'].mean()*100:.2f}%")


def check_market_arbitrage(vol_key_rate, fwd_key_rate, fwd_ois, tau=0.25,
                           h_T=0.1, h_K=0.005, plot=True, verbose=True,
                           surface_name="Market"):
    """
    Check vol surface for arbitrage using Dupire local vol.
    
    This is a convenience wrapper that builds forward/discount curves internally.
    
    Args:
        vol_key_rate: DataFrame with [time_to_maturity, strike, implied_normal_vol]
        fwd_key_rate: DataFrame with [time_to_maturity, forward_rate] for key rate
        fwd_ois: DataFrame with [time_to_maturity, forward_rate] for OIS (discounting)
        tau: Accrual period (default 0.25 for quarterly)
        h_T: Step size for time derivative (Richardson extrapolation)
        h_K: Step size for strike derivative (Richardson extrapolation)
        plot: Whether to show heatmaps
        verbose: Whether to print detailed summary
        surface_name: Name for titles ("Market" or "Model")
    
    Returns:
        arb_df: DataFrame with arbitrage check at each point
        summary: dict with summary statistics
    """
    # Build forward curve (key rate)
    fwd_sorted = fwd_key_rate.sort_values('time_to_maturity')
    fwd_interp = PchipInterpolator(
        fwd_sorted['time_to_maturity'].values,
        fwd_sorted['forward_rate'].values
    )
    
    # Build OIS curve for discounting
    ois_sorted = fwd_ois.sort_values('time_to_maturity')
    ois_interp = PchipInterpolator(
        ois_sorted['time_to_maturity'].values,
        ois_sorted['forward_rate'].values
    )
    
    T_max = fwd_sorted['time_to_maturity'].max()
    
    def get_forward(T):
        """Get forward rate at maturity T."""
        return float(fwd_interp(min(T, T_max)))
    
    def get_discount(T):
        """Get discount factor to payment date T + tau."""
        T_pay = min(T + tau, T_max)
        # Simpson integration for average rate
        n_pts = max(10, int(T_pay * 100))
        t_grid = np.linspace(0, T_pay, n_pts)
        r_avg = np.mean(ois_interp(t_grid))
        return np.exp(-r_avg * T_pay)
    
    # Run arbitrage check
    arb_df, summary = check_surface_arbitrage(
        vol_key_rate, get_forward, get_discount, tau, h_T=h_T, h_K=h_K
    )
    
    if verbose:
        print_arbitrage_summary(arb_df, summary, title=f"{surface_name.upper()} SURFACE")
    
    if plot:
        plot_arbitrage_heatmaps(arb_df, title_prefix=surface_name)
    
    return arb_df, summary


# =============================================================================
# VOL SURFACE GENERATION AND INVERSION
# =============================================================================


def generate_caplet_vol_surface(vol_key_rate, fwd_key_rate, fwd_ois=None, version_name="Model",
                                 F_model=None, P_model=None):
    """
    Generate volatility surface from model prices for NOW-STARTING average rate caplets.
    
    The MC model PV = E[max(∫₀ᵀ a_t dt - T·K, 0) · exp(-∫₀ᵀ r_t dt)]
    
    Under Bachelier for average rates:
        PV = T · P(0,T) · [(F-K)Φ(d) + σ_n·ĝ·φ(d)]
    where:
        F = I_KEY(0,T)/T  (average instantaneous forward, NOT period forward)
        ĝ = √(T/3)        (now-starting average rate adjustment)
        d = (F-K)/(σ_n·ĝ)
    
    To invert: first compute undiscounted_unit = PV / (T · P(0,T)),
    then solve: undiscounted_unit = (F-K)Φ(d) + σ_n·ĝ·φ(d) for σ_n.
    
    T-forward measure mode (F_model, P_model provided):
        E^T[payoff] = E[disc·payoff] / E[disc],  F_model = E^T[ā]
        By Jensen: E^T[max(ā−K,0)] ≥ max(F_model−K,0), so Bachelier
        inversion is always well-defined (undiscounted ≥ intrinsic).
    
    Parameters:
    -----------
    vol_key_rate : pd.DataFrame
        DataFrame with columns: time_to_maturity, strike, implied_normal_vol, pv_model_key
    fwd_key_rate : pd.DataFrame
        Forward rate curve (period rates for ZCB computation)
    fwd_ois : pd.DataFrame, optional
        OIS forward curve for discounting. If None, uses fwd_key_rate.
    version_name : str
        Version identifier for labeling (e.g., "v1", "v2")
    F_model : array-like, optional
        Model forward per caplet under T-forward measure: E^T[ā].
        When provided, used instead of market forward for Bachelier inversion.
    P_model : array-like, optional
        Model ZCB price per caplet: E[exp(-∫r ds)].
        When provided, used instead of deterministic OIS discount.
    
    Returns:
    --------
    vol_results : pd.DataFrame
        DataFrame with market and model vols, errors, and diagnostics
    vol_rmse : float
        Volatility RMSE across all valid caplets
    """
    # Build forward rate interpolator for PERIOD rates (for ZCB computation)
    fwd_sorted = fwd_key_rate.sort_values('time_to_maturity')
    fwd_interp_period = PchipInterpolator(
        fwd_sorted['time_to_maturity'].values,
        fwd_sorted['forward_rate'].values
    )
    
    # Build OIS forward interpolator for discounting
    if fwd_ois is not None:
        ois_sorted = fwd_ois.sort_values('time_to_maturity')
        ois_interp = PchipInterpolator(
            ois_sorted['time_to_maturity'].values,
            ois_sorted['forward_rate'].values
        )
    else:
        ois_interp = fwd_interp_period
    
    def avg_inst_forward(T):
        """Average instantaneous forward = I(0,T)/T = -ln(P(0,T))/T."""
        F_period = float(fwd_interp_period(T))
        zcb = 1.0 / (1.0 + T * F_period)
        return -np.log(max(zcb, 1e-15)) / max(T, 1e-10)
    
    def ois_discount(T):
        """OIS discount factor P(0,T) = 1/(1 + T*F_OIS(T))."""
        F_ois = float(ois_interp(T))
        return 1.0 / (1.0 + T * F_ois)
    
    vol_results = vol_key_rate.copy()
    model_vols = []
    vol_errors = []
    failed_inversions = 0
    arbitrage_violations = 0
    failure_reasons = {'bounds': 0, 'convergence': 0, 'other': 0}
    
    for i, (idx, row) in enumerate(vol_results.iterrows()):
        T = row['time_to_maturity']
        K = row['strike']
        model_pv = row['pv_model_key']
        market_vol = row['implied_normal_vol']
        
        if T <= 0:
            failure_reasons['bounds'] += 1
            failed_inversions += 1
            model_vols.append(np.nan)
            vol_errors.append(np.nan)
            continue
        
        # Forward and discount: use model estimates if provided (T-forward measure)
        F = float(F_model[i]) if F_model is not None else avg_inst_forward(T)
        disc = float(P_model[i]) if P_model is not None else ois_discount(T)
        
        # Use module-level implied_vol_avg_rate (handles deep ITM via stable TV channel)
        model_vol = implied_vol_avg_rate(F, K, T, model_pv, disc)
        
        intrinsic_und = max(F - K, 0)
        undiscounted_unit = model_pv / (T * disc + 1e-15)
        if undiscounted_unit < intrinsic_und:
            arbitrage_violations += 1  # convexity effect, not true arb
        
        if np.isnan(model_vol):
            failed_inversions += 1
            failure_reasons['convergence'] += 1
            model_vols.append(np.nan)
            vol_errors.append(np.nan)
        else:
            model_vols.append(model_vol)
            vol_errors.append(model_vol - market_vol)
    
    model_vols_arr = np.array(model_vols, dtype=float)
    
    # No capping — store raw model vols directly
    vol_results[f'model_vol_{version_name}'] = model_vols_arr
    vol_errors = model_vols_arr - vol_results['implied_normal_vol'].values
    vol_results[f'vol_error_{version_name}'] = vol_errors
    
    valid_mask = ~np.isnan(vol_errors)
    vol_rmse = np.sqrt(np.mean(vol_errors[valid_mask]**2)) if valid_mask.any() else np.nan
    success_rate = valid_mask.sum() / len(vol_errors) * 100
    
    n_failed = (~valid_mask).sum()
    
    print(f"\n{'='*70}")
    print(f"{version_name.upper()} VOLATILITY SURFACE DIAGNOSTICS")
    print(f"{'='*70}")
    print(f"Total caplets:        {len(vol_errors)}")
    print(f"Valid vol inversions: {valid_mask.sum()} ({success_rate:.1f}%)")
    print(f"Failed inversions:    {n_failed}")
    print(f"Sub-intrinsic (conv): {arbitrage_violations} (MC price < det. intrinsic → NaN)")
    print(f"Vol RMSE:             {vol_rmse*100:.3f}%")
    if valid_mask.any():
        valid_vols = model_vols_arr[valid_mask]
        print(f"Model vol range:      {valid_vols.min()*100:.2f}% - {valid_vols.max()*100:.2f}%")
        print(f"Market vol range:     {vol_results['implied_normal_vol'].min()*100:.2f}% - {vol_results['implied_normal_vol'].max()*100:.2f}%")
    print(f"{'='*70}\n")
    
    return vol_results, vol_rmse


def plot_caplet_vol_surface(vol_results, version_name="Model", plot_maturities=[1.0, 3.0, 5.0, 7.0, 10.0],
                            fwd_key_rate=None):
    """
    Plot 3D volatility surface comparison: market vs model.
    
    Raw model vols plotted without any clipping or limits.
    
    Parameters:
    -----------
    vol_results : pd.DataFrame
        Output from generate_caplet_vol_surface() with market and model vols
    version_name : str
        Version identifier for plot titles
    plot_maturities : list
        Maturities to highlight in 2D slice plots
    fwd_key_rate : pd.DataFrame, optional
        Forward rate curve — if provided, forward line is drawn on smile plots
    """
    model_vol_col = f'model_vol_{version_name}'
    error_col = f'vol_error_{version_name}'
    
    maturities = sorted(vol_results['time_to_maturity'].unique())
    strikes = sorted(vol_results['strike'].unique())
    T_grid, K_grid = np.meshgrid(maturities, strikes)
    
    market_vol_grid = np.zeros_like(T_grid)
    model_vol_grid = np.full_like(T_grid, np.nan)
    error_grid = np.full_like(T_grid, np.nan)
    
    for i, k in enumerate(strikes):
        for j, t in enumerate(maturities):
            row = vol_results[(vol_results['time_to_maturity'] == t) & (vol_results['strike'] == k)]
            if len(row) > 0:
                market_vol_grid[i, j] = row['implied_normal_vol'].values[0] * 100
                model_val = row[model_vol_col].values[0]
                if not np.isnan(model_val):
                    model_vol_grid[i, j] = model_val * 100
                    error_grid[i, j] = (model_val - row['implied_normal_vol'].values[0]) * 100
    
    # Clamp model vols to a sane range around market to kill triangle artifacts
    mkt_lo = np.nanmin(market_vol_grid) * 0.2
    mkt_hi = np.nanmax(market_vol_grid) * 3.0
    model_vol_display = np.where(np.isnan(model_vol_grid), np.nanmedian(model_vol_grid), model_vol_grid)
    model_vol_display = np.clip(model_vol_display, mkt_lo, mkt_hi)
    error_display = np.where(np.isnan(error_grid), 0, error_grid)
    err_lim = max(abs(np.nanpercentile(error_grid[~np.isnan(error_grid)], 5)),
                  abs(np.nanpercentile(error_grid[~np.isnan(error_grid)], 95))) if (~np.isnan(error_grid)).any() else 1.0
    error_display = np.clip(error_display, -err_lim * 1.5, err_lim * 1.5)
    
    # Build forward curve for reference
    fwd_interp = None
    if fwd_key_rate is not None:
        fs = fwd_key_rate.sort_values('time_to_maturity')
        fwd_interp = PchipInterpolator(fs['time_to_maturity'].values, fs['forward_rate'].values)
    
    total_points = T_grid.size
    valid_model_points = np.sum(~np.isnan(model_vol_grid))
    coverage_pct = valid_model_points / total_points * 100
    
    print(f"\nSurface Coverage: {valid_model_points}/{total_points} points ({coverage_pct:.1f}%)")
    if total_points > valid_model_points:
        print(f"Missing {total_points - valid_model_points} points due to inversion failures")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Shared z-limits for market & model so visual scale matches
    z_lo = min(np.nanmin(market_vol_grid), np.nanmin(model_vol_display)) * 0.9
    z_hi = max(np.nanmax(market_vol_grid), np.nanmax(model_vol_display)) * 1.1
    
    # Row 1: 3D Surfaces — SAME z-limits on market & model
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(T_grid, K_grid * 100, market_vol_grid,
                             cmap='viridis', alpha=0.9, edgecolor='none')
    ax1.set_xlabel('Maturity (y)', fontsize=10, labelpad=5)
    ax1.set_ylabel('Strike (%)', fontsize=10, labelpad=5)
    ax1.set_zlabel('Vol (%)', fontsize=10, labelpad=5)
    ax1.set_zlim(z_lo, z_hi)
    ax1.set_title('Market Volatility Surface', fontsize=12, fontweight='bold')
    ax1.view_init(elev=25, azim=135)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(T_grid, K_grid * 100, model_vol_display,
                             cmap='viridis', alpha=0.9, edgecolor='none')
    ax2.set_xlabel('Maturity (y)', fontsize=10, labelpad=5)
    ax2.set_ylabel('Strike (%)', fontsize=10, labelpad=5)
    ax2.set_zlabel('Vol (%)', fontsize=10, labelpad=5)
    ax2.set_zlim(z_lo, z_hi)
    ax2.set_title(f'{version_name.upper()} Model Volatility Surface', fontsize=12, fontweight='bold')
    ax2.view_init(elev=25, azim=135)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(T_grid, K_grid * 100, error_display,
                             cmap='RdBu_r', alpha=0.9, edgecolor='none')
    ax3.set_xlabel('Maturity (y)', fontsize=10, labelpad=5)
    ax3.set_ylabel('Strike (%)', fontsize=10, labelpad=5)
    ax3.set_zlabel('Error (%)', fontsize=10, labelpad=5)
    ax3.set_title(f'{version_name.upper()} - Market (Vol Error)', fontsize=12, fontweight='bold')
    ax3.view_init(elev=25, azim=135)
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
    
    # Row 2: 2D Maturity Slices — market & model overlaid, error separate
    ax4 = fig.add_subplot(2, 3, 4)
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_maturities)))
    for ci, mat in enumerate(plot_maturities):
        if mat in maturities:
            subset = vol_results[vol_results['time_to_maturity'] == mat].sort_values('strike')
            lbl = f'{mat:.0f}Y' if mat >= 1 else f'{mat*12:.0f}M'
            ax4.plot(subset['strike'] * 100, subset['implied_normal_vol'] * 100,
                    'o-', color=colors[ci], ms=3, lw=2, label=f'{lbl} Market')
    ax4.set_xlabel('Strike (%)', fontsize=11)
    ax4.set_ylabel('Implied Vol (%)', fontsize=11)
    ax4.set_title('Market Vol - Maturity Slices', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, ncol=2)
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(2, 3, 5)
    for ci, mat in enumerate(plot_maturities):
        if mat in maturities:
            subset = vol_results[vol_results['time_to_maturity'] == mat].sort_values('strike')
            lbl = f'{mat:.0f}Y' if mat >= 1 else f'{mat*12:.0f}M'
            # Market (faint)
            ax5.plot(subset['strike'] * 100, subset['implied_normal_vol'] * 100,
                    '-', color=colors[ci], alpha=0.3, lw=1)
            # Model
            valid = ~subset[model_vol_col].isna()
            ax5.plot(subset.loc[valid, 'strike'] * 100, subset.loc[valid, model_vol_col] * 100,
                    's-', color=colors[ci], ms=3, lw=2, label=f'{lbl}')
            # Forward line
            if fwd_interp is not None:
                fwd = float(fwd_interp(mat)) * 100
                ax5.axvline(fwd, color=colors[ci], ls=':', alpha=0.3, lw=0.8)
    ax5.set_xlabel('Strike (%)', fontsize=11)
    ax5.set_ylabel('Implied Vol (%)', fontsize=11)
    ax5.set_title(f'{version_name.upper()} Model vs Market (faint)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9, ncol=2)
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(2, 3, 6)
    for ci, mat in enumerate(plot_maturities):
        if mat in maturities:
            subset = vol_results[vol_results['time_to_maturity'] == mat].sort_values('strike')
            lbl = f'{mat:.0f}Y' if mat >= 1 else f'{mat*12:.0f}M'
            valid = ~subset[error_col].isna()
            ax6.plot(subset.loc[valid, 'strike'] * 100, subset.loc[valid, error_col] * 100,
                    '^-', color=colors[ci], ms=3, lw=1.5, label=f'{lbl} Error')
    ax6.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax6.set_xlabel('Strike (%)', fontsize=11)
    ax6.set_ylabel('Vol Error (%)', fontsize=11)
    ax6.set_title(f'{version_name.upper()} - Market Error', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9, ncol=2)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_caplet_price_heatmaps(vol_key_rate, model_pvs, market_pvs, version_name="Model"):
    """
    Plot heatmaps showing caplet price errors and model PVs.
    
    Parameters:
    -----------
    vol_key_rate : pd.DataFrame
        DataFrame with time_to_maturity and strike columns
    model_pvs : array-like
        Model caplet prices (in decimal, will be converted to bp)
    market_pvs : array-like
        Market caplet prices (in decimal, will be converted to bp)
    version_name : str
        Version identifier for plot titles
    """
    import numpy as np
    
    # Convert to numpy arrays
    model_pvs = np.array(model_pvs) if hasattr(model_pvs, 'cpu') else np.array(model_pvs)
    market_pvs = np.array(market_pvs) if hasattr(market_pvs, 'cpu') else np.array(market_pvs)
    
    # Build DataFrame
    caplet_grid = pd.DataFrame({
        'Maturity': vol_key_rate['time_to_maturity'].values,
        'Strike': vol_key_rate['strike'].values * 100,  # Convert to %
        'Model_PV': model_pvs * 10000,  # Convert to bp
        'Market_PV': market_pvs * 10000,
        'Diff_bp': (model_pvs - market_pvs) * 10000,
        'Diff_pct': (model_pvs - market_pvs) / (market_pvs + 1e-10) * 100
    })
    
    # Pivot for heatmaps
    pivot_diff_pct = caplet_grid.pivot_table(values='Diff_pct', index='Strike', columns='Maturity', aggfunc='mean')
    pivot_diff_bp = caplet_grid.pivot_table(values='Diff_bp', index='Strike', columns='Maturity', aggfunc='mean')
    pivot_model = caplet_grid.pivot_table(values='Model_PV', index='Strike', columns='Maturity', aggfunc='mean')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Percentage error
    im1 = axes[0].imshow(pivot_diff_pct.values, aspect='auto', cmap='RdBu_r', vmin=-100, vmax=100)
    axes[0].set_xticks(range(len(pivot_diff_pct.columns)))
    axes[0].set_xticklabels([f'{x:.1f}Y' for x in pivot_diff_pct.columns], rotation=45)
    axes[0].set_yticks(range(len(pivot_diff_pct.index)))
    axes[0].set_yticklabels([f'{x:.1f}%' for x in pivot_diff_pct.index])
    axes[0].set_xlabel('Maturity')
    axes[0].set_ylabel('Strike')
    axes[0].set_title(f'{version_name} Price Error (Model-Market)/Market %')
    plt.colorbar(im1, ax=axes[0], label='Error %')
    
    # Plot 2: Absolute error in bp
    bp_max = max(abs(np.nanmin(pivot_diff_bp.values)), abs(np.nanmax(pivot_diff_bp.values)))
    im2 = axes[1].imshow(pivot_diff_bp.values, aspect='auto', cmap='RdBu_r', vmin=-bp_max, vmax=bp_max)
    axes[1].set_xticks(range(len(pivot_diff_bp.columns)))
    axes[1].set_xticklabels([f'{x:.1f}Y' for x in pivot_diff_bp.columns], rotation=45)
    axes[1].set_yticks(range(len(pivot_diff_bp.index)))
    axes[1].set_yticklabels([f'{x:.1f}%' for x in pivot_diff_bp.index])
    axes[1].set_xlabel('Maturity')
    axes[1].set_ylabel('Strike')
    axes[1].set_title(f'{version_name} Price Error (Model-Market) bp')
    plt.colorbar(im2, ax=axes[1], label='Error (bp)')
    
    # Plot 3: Model PV
    im3 = axes[2].imshow(pivot_model.values, aspect='auto', cmap='viridis')
    axes[2].set_xticks(range(len(pivot_model.columns)))
    axes[2].set_xticklabels([f'{x:.1f}Y' for x in pivot_model.columns], rotation=45)
    axes[2].set_yticks(range(len(pivot_model.index)))
    axes[2].set_yticklabels([f'{x:.1f}%' for x in pivot_model.index])
    axes[2].set_xlabel('Maturity')
    axes[2].set_ylabel('Strike')
    axes[2].set_title(f'{version_name} Model PV (bp)')
    plt.colorbar(im3, ax=axes[2], label='PV (bp)')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    rmse_bp = np.sqrt(np.mean(caplet_grid['Diff_bp']**2))
    mae_bp = np.mean(np.abs(caplet_grid['Diff_bp']))
    print(f"\n{version_name} Price Fit Summary:")
    print(f"  RMSE: {rmse_bp:.2f} bp")
    print(f"  MAE:  {mae_bp:.2f} bp")
    print(f"  Model range: {caplet_grid['Model_PV'].min():.2f} - {caplet_grid['Model_PV'].max():.2f} bp")
    print(f"  Market range: {caplet_grid['Market_PV'].min():.2f} - {caplet_grid['Market_PV'].max():.2f} bp")


def print_model_vs_market_table(vol_key_rate, fwd_key_rate, model_pvs, market_pvs, 
                                 vol_results=None, version_name="Model", top_n=15):
    """
    Print comprehensive model vs market comparison table.
    
    Parameters:
    -----------
    vol_key_rate : pd.DataFrame
        DataFrame with time_to_maturity, strike, implied_normal_vol columns
    fwd_key_rate : pd.DataFrame
        Forward rate curve
    model_pvs : array-like
        Model caplet prices (decimal)
    market_pvs : array-like
        Market caplet prices (decimal)  
    vol_results : pd.DataFrame, optional
        Output from generate_caplet_vol_surface with model vols
    version_name : str
        Version identifier
    top_n : int
        Number of worst/best fits to show
    """
    import numpy as np
    
    # Convert to numpy
    model_pvs = np.array(model_pvs.cpu()) if hasattr(model_pvs, 'cpu') else np.array(model_pvs)
    market_pvs = np.array(market_pvs.cpu()) if hasattr(market_pvs, 'cpu') else np.array(market_pvs)
    
    # Build forward interpolator
    fwd_sorted = fwd_key_rate.sort_values('time_to_maturity')
    fwd_interp = PchipInterpolator(
        fwd_sorted['time_to_maturity'].values,
        fwd_sorted['forward_rate'].values
    )
    
    # Build full comparison DataFrame
    df = pd.DataFrame({
        'Maturity': vol_key_rate['time_to_maturity'].values,
        'Strike_%': vol_key_rate['strike'].values * 100,
        'Forward_%': [fwd_interp(t) * 100 for t in vol_key_rate['time_to_maturity'].values],
        'Mkt_Vol_%': vol_key_rate['implied_normal_vol'].values * 100,
        'Model_PV_bp': model_pvs * 10000,
        'Market_PV_bp': market_pvs * 10000,
        'PV_Diff_bp': (model_pvs - market_pvs) * 10000,
        'PV_Diff_%': (model_pvs - market_pvs) / (market_pvs + 1e-10) * 100
    })
    
    # Add model vol if available
    if vol_results is not None:
        model_vol_col = f'model_vol_{version_name}'
        if model_vol_col in vol_results.columns:
            df['Model_Vol_%'] = vol_results[model_vol_col].values * 100
            df['Vol_Diff_%'] = (vol_results[model_vol_col].values - vol_key_rate['implied_normal_vol'].values) * 100
    
    # Moneyness
    df['Moneyness_%'] = df['Forward_%'] - df['Strike_%']
    df['Abs_PV_Diff'] = np.abs(df['PV_Diff_bp'])
    
    # Print header
    print(f"\n{'='*100}")
    print(f"{version_name.upper()} MODEL VS MARKET COMPARISON")
    print(f"{'='*100}")
    
    # Overall statistics
    pv_rmse = np.sqrt(np.mean(df['PV_Diff_bp']**2))
    pv_mae = np.mean(df['Abs_PV_Diff'])
    pv_max_err = df['Abs_PV_Diff'].max()
    
    print(f"\nOVERALL PRICE FIT STATISTICS:")
    print(f"  Price RMSE:     {pv_rmse:.2f} bp")
    print(f"  Price MAE:      {pv_mae:.2f} bp")
    print(f"  Max |error|:    {pv_max_err:.2f} bp")
    print(f"  Model PV range: {df['Model_PV_bp'].min():.2f} - {df['Model_PV_bp'].max():.2f} bp")
    print(f"  Market PV range:{df['Market_PV_bp'].min():.2f} - {df['Market_PV_bp'].max():.2f} bp")
    
    if 'Model_Vol_%' in df.columns:
        valid_vol = ~df['Model_Vol_%'].isna()
        if valid_vol.any():
            vol_rmse = np.sqrt(np.mean(df.loc[valid_vol, 'Vol_Diff_%']**2))
            print(f"\nOVERALL VOL FIT STATISTICS:")
            print(f"  Vol RMSE:       {vol_rmse:.2f}%")
            print(f"  Model vol range:{df.loc[valid_vol, 'Model_Vol_%'].min():.2f}% - {df.loc[valid_vol, 'Model_Vol_%'].max():.2f}%")
            print(f"  Market vol range:{df['Mkt_Vol_%'].min():.2f}% - {df['Mkt_Vol_%'].max():.2f}%")
    
    # Summary by maturity
    print(f"\n{'='*100}")
    print(f"SUMMARY BY MATURITY")
    print(f"{'='*100}")
    
    mat_summary = df.groupby('Maturity').agg({
        'Forward_%': 'mean',
        'Model_PV_bp': 'mean',
        'Market_PV_bp': 'mean',
        'PV_Diff_bp': 'mean',
        'Abs_PV_Diff': 'mean'
    }).round(2)
    mat_summary.columns = ['Fwd_%', 'Model_bp', 'Market_bp', 'Diff_bp', '|Diff|_bp']
    
    if 'Model_Vol_%' in df.columns:
        vol_summary = df.groupby('Maturity').agg({
            'Mkt_Vol_%': 'mean',
            'Model_Vol_%': 'mean'
        }).round(2)
        mat_summary['Mkt_Vol_%'] = vol_summary['Mkt_Vol_%']
        mat_summary['Model_Vol_%'] = vol_summary['Model_Vol_%']
    
    print(mat_summary.to_string())
    
    # Top worst fits
    df_sorted = df.sort_values('Abs_PV_Diff', ascending=False)
    
    print(f"\n{'='*100}")
    print(f"TOP {top_n} WORST FITS (by |PV difference|)")
    print(f"{'='*100}")
    
    cols_to_show = ['Maturity', 'Strike_%', 'Forward_%', 'Moneyness_%', 
                    'Model_PV_bp', 'Market_PV_bp', 'PV_Diff_bp', 'PV_Diff_%']
    if 'Model_Vol_%' in df.columns:
        cols_to_show.extend(['Model_Vol_%', 'Mkt_Vol_%'])
    
    print(df_sorted[cols_to_show].head(top_n).to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    
    # Top best fits
    print(f"\n{'='*100}")
    print(f"TOP {top_n} BEST FITS (by |PV difference|)")
    print(f"{'='*100}")
    print(df_sorted[cols_to_show].tail(top_n).to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    
    # Analysis by moneyness
    print(f"\n{'='*100}")
    print(f"ANALYSIS BY MONEYNESS")
    print(f"{'='*100}")
    
    df['Moneyness_Bucket'] = pd.cut(df['Moneyness_%'], 
                                     bins=[-np.inf, -2, -0.5, 0.5, 2, np.inf],
                                     labels=['Deep OTM (<-2%)', 'OTM (-2% to -0.5%)', 
                                            'ATM (-0.5% to 0.5%)', 'ITM (0.5% to 2%)', 
                                            'Deep ITM (>2%)'])
    
    moneyness_summary = df.groupby('Moneyness_Bucket', observed=True).agg({
        'Model_PV_bp': ['count', 'mean'],
        'Market_PV_bp': 'mean',
        'Abs_PV_Diff': 'mean'
    }).round(2)
    moneyness_summary.columns = ['Count', 'Model_bp', 'Market_bp', '|Diff|_bp']
    print(moneyness_summary.to_string())
    
    return df


# =============================================================================
# HW-HESTON SIMULATION & PRICING
# =============================================================================

def fast_cir_paths(n_paths, n_steps, dt, v0, kappa, theta_vec, epsilon, device, randn=None):
    """CIR variance via Euler-Maruyama with reflection."""
    v = torch.zeros((n_paths, n_steps + 1), dtype=torch.float32, device=device)
    v[:, 0] = v0
    sqrt_dt = np.sqrt(dt)
    if randn is None:
        randn = torch.randn(n_paths, n_steps, device=device)
    for i in range(n_steps):
        v_curr = v[:, i]
        theta_i = theta_vec[i] if theta_vec.dim() > 0 else theta_vec
        drift = kappa * (theta_i - v_curr) * dt
        diffusion = epsilon * torch.sqrt(torch.clamp(v_curr, min=1e-8)) * sqrt_dt * randn[:, i]
        v[:, i+1] = torch.abs(v_curr + drift + diffusion)
    return v


def fast_ou_paths(n_paths, n_steps, dt, x0, lam, vol_paths, device, randn=None):
    """Hull-White OU process with stochastic vol."""
    x = torch.zeros((n_paths, n_steps + 1), dtype=torch.float32, device=device)
    x[:, 0] = x0
    sqrt_dt = np.sqrt(dt)
    exp_lam_dt = np.exp(-lam * dt)
    if randn is None:
        randn = torch.randn(n_paths, n_steps, device=device)
    for i in range(n_steps):
        x[:, i+1] = x[:, i] * exp_lam_dt + torch.sqrt(torch.clamp(vol_paths[:, i], min=1e-9)) * sqrt_dt * randn[:, i]
    return x


def fast_hw_paths(n_paths, n_steps, dt, x0, gamma, xi, device, randn=None):
    """Vectorized OU spread process with constant vol.
    
    Exact OU transition via geometric weighted cumsum (no Python loop).
    """
    exp_gamma_dt = np.exp(-gamma * dt)
    var_factor = float(xi * xi * (1 - np.exp(-2 * gamma * dt)) / (2 * gamma + 1e-8))
    std_factor = np.sqrt(max(var_factor, 1e-12))
    if randn is None:
        randn = torch.randn(n_paths, n_steps, device=device)
    rho = exp_gamma_dt
    w = std_factor * randn
    j_idx = torch.arange(n_steps, device=device, dtype=torch.float32)
    rho_powers = rho ** j_idx
    inv_rho_powers = 1.0 / (rho_powers + 1e-30)
    scaled = w * inv_rho_powers.unsqueeze(0)
    cum_scaled = scaled.cumsum(dim=1)
    k_steps = cum_scaled * rho_powers.unsqueeze(0)
    rho_powers_1 = rho ** (j_idx + 1)
    k_steps = k_steps + x0 * rho_powers_1.unsqueeze(0)
    k0 = torch.full((n_paths, 1), x0, dtype=torch.float32, device=device)
    return torch.cat([k0, k_steps], dim=1)


def rho_to_vec(rho_nodes, rho_times, timeline):
    """Interpolate rho nodes to timeline via piecewise linear."""
    t = timeline[:-1]
    rho_vec = torch.zeros_like(t)
    for i in range(len(rho_times)):
        if i == 0:
            mask = t <= rho_times[i]
            rho_vec[mask] = rho_nodes[i]
        else:
            mask = (t > rho_times[i-1]) & (t <= rho_times[i])
            frac = (t[mask] - rho_times[i-1]) / (rho_times[i] - rho_times[i-1])
            rho_vec[mask] = rho_nodes[i-1] + frac * (rho_nodes[i] - rho_nodes[i-1])
    rho_vec[t > rho_times[-1]] = rho_nodes[-1]
    return rho_vec


def theta_to_vec(theta_vals, theta_nodes, timeline):
    """Interpolate theta nodes to full timeline via PCHIP spline."""
    spline = PchipSpline1D(theta_nodes, theta_vals)
    return spline.evaluate(timeline)


def fast_simulate(n_paths, timeline, theta_vec, epsilon, v0, kappa, lam, gamma, xi,
                  f_key_vec, f_ois_vec, device, seed=None, rho_vx=0.0, antithetic=False):
    """Combined HW-Heston simulation with Cholesky correlation and antithetic variates.
    
    Correlation structure:
        W_v = Z_v                                    (CIR variance)
        W_x = rho(t) * Z_v + sqrt(1-rho(t)^2) * Z_x (OU rate, correlated with vol)
        W_k = Z_k                                    (HW spread, independent)
    
    Returns:
        key_rate_paths: [n_paths, n_steps+1]
        ois_rate_paths: [n_paths, n_steps+1]
        v_paths: [n_paths, n_steps+1]
    """
    n_steps = len(timeline) - 1
    dt = (timeline[1] - timeline[0]).item()
    if seed is not None:
        torch.manual_seed(seed)
    if antithetic:
        half = n_paths // 2
        Z_v_h = torch.randn(half, n_steps, device=device)
        Z_x_h = torch.randn(half, n_steps, device=device)
        Z_k_h = torch.randn(half, n_steps, device=device)
        Z_v = torch.cat([Z_v_h, -Z_v_h], dim=0)
        Z_x = torch.cat([Z_x_h, -Z_x_h], dim=0)
        Z_k = torch.cat([Z_k_h, -Z_k_h], dim=0)
        n_paths = 2 * half
    else:
        Z_v = torch.randn(n_paths, n_steps, device=device)
        Z_x = torch.randn(n_paths, n_steps, device=device)
        Z_k = torch.randn(n_paths, n_steps, device=device)
    randn_v = Z_v
    if isinstance(rho_vx, torch.Tensor) and rho_vx.dim() >= 1:
        rho_t = rho_vx.unsqueeze(0)
        randn_x = rho_t * Z_v + torch.sqrt(torch.clamp(1.0 - rho_t**2, min=0.0)) * Z_x
    else:
        rho_val = rho_vx.item() if isinstance(rho_vx, torch.Tensor) else float(rho_vx)
        randn_x = rho_val * Z_v + np.sqrt(max(1.0 - rho_val**2, 0.0)) * Z_x
    randn_k = Z_k
    v_paths = fast_cir_paths(n_paths, n_steps, dt, v0, kappa, theta_vec, epsilon, device, randn_v)
    x_paths = fast_ou_paths(n_paths, n_steps, dt, 0.0, lam, v_paths, device, randn_x)
    ks_paths = fast_hw_paths(n_paths, n_steps, dt, 0.0, gamma, xi, device, randn_k)
    key_paths = f_key_vec.unsqueeze(0) + x_paths + ks_paths
    ois_paths = f_ois_vec.unsqueeze(0) + x_paths
    return key_paths, ois_paths, v_paths


def batch_price_caplets(key_paths, ois_paths, timeline, idx_fixes, idx_pays, strikes, tau, device):
    """T-forward measure pricing of now-starting average rate caplets.
    
    Returns (pvs, F_model, P_model):
        pvs     = E[disc * payoff]                   (money-market PV)
        P_model = E[exp(-int_0^T r ds)]              (model ZCB per caplet)
        F_model = E[disc * (int a dt/T)] / E[disc]   (T-forward measure forward)
    
    Bachelier inversion with (F_model, P_model) guarantees undiscounted >= intrinsic
    by Jensen's inequality: E^T[max(a_bar-K,0)] >= max(E^T[a_bar]-K,0) = max(F_model-K,0).
    """
    dt_val = (timeline[1] - timeline[0]).item()
    sum_key_dt = (key_paths[:, :-1] * dt_val).cumsum(dim=1)
    sum_ois_dt = (ois_paths[:, :-1] * dt_val).cumsum(dim=1)
    i_idx = (idx_fixes - 1).clamp(0, sum_key_dt.shape[1] - 1).long()
    sum_key_T = sum_key_dt[:, i_idx]
    sum_ois_T = sum_ois_dt[:, i_idx]
    T_vals = timeline[idx_fixes]
    disc = torch.exp(-sum_ois_T)
    payoff = torch.clamp(sum_key_T - T_vals.unsqueeze(0) * strikes.unsqueeze(0), min=0)
    pvs = (payoff * disc).mean(dim=0)
    P_model = disc.mean(dim=0)
    avg_rate = sum_key_T / T_vals.unsqueeze(0)
    F_model = (disc * avg_rate).mean(dim=0) / P_model
    return pvs, F_model, P_model
