"""
Caplet Volatility Surface Generation and Plotting Utilities

Functions for computing implied normal volatility surfaces from model prices
and comparing against market data for interest rate caplets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator


def generate_caplet_vol_surface(vol_key_rate, fwd_key_rate, version_name="Model"):
    """
    Generate volatility surface from model prices.
    
    Parameters:
    -----------
    vol_key_rate : pd.DataFrame
        DataFrame with columns: time_to_maturity, strike, implied_normal_vol, pv_model_key
    fwd_key_rate : pd.DataFrame
        Forward rate curve for strike mapping
    version_name : str
        Version identifier for labeling (e.g., "v1", "v2")
    
    Returns:
    --------
    vol_results : pd.DataFrame
        DataFrame with market and model vols, errors, and diagnostics
    vol_rmse : float
        Volatility RMSE across all valid caplets
    """
    
    def implied_vol_from_price(F, K, price, T, tol=1e-9, max_iter=100):
        """Invert Bachelier formula to get implied vol."""
        if price <= 0 or T <= 0:
            return np.nan
        
        intrinsic = max(F - K, 0)
        max_reasonable_price = max(intrinsic * 1000, 100 * abs(F), 10.0)
        if price > max_reasonable_price:
            return np.nan
        
        sqrt_T = np.sqrt(T)
        
        # Initial guess using ATM approximation: price ≈ 0.4 * sigma * sqrt(T)
        sigma_init = price / (0.3989 * sqrt_T + 1e-10)
        
        def price_error(sigma):
            if sigma <= 0:
                return 1e10
            d = (F - K) / (sigma * sqrt_T + 1e-10)
            model_price = (F - K) * norm.cdf(d) + sigma * sqrt_T * norm.pdf(d)
            return model_price - price
        
        try:
            # Try narrow range first centered on initial guess
            lo = max(sigma_init * 0.01, 1e-8)
            hi = min(sigma_init * 100, 5.0)
            vol = brentq(price_error, lo, hi, xtol=tol, maxiter=max_iter)
            return vol if 1e-6 < vol < 5.0 else np.nan
        except ValueError:
            try:
                # Try wider range
                vol = brentq(price_error, 1e-10, 10.0, xtol=tol, maxiter=max_iter)
                return vol if vol > 0 else np.nan
            except:
                return np.nan
    
    # Build forward rate interpolator (NOT dict lookup - that has fallback bug!)
    # Sort by maturity to ensure monotonic x for interpolation
    fwd_sorted = fwd_key_rate.sort_values('time_to_maturity')
    fwd_interp = PchipInterpolator(
        fwd_sorted['time_to_maturity'].values,
        fwd_sorted['forward_rate'].values
    )
    
    vol_results = vol_key_rate.copy()
    model_vols = []
    vol_errors = []
    failed_inversions = 0
    arbitrage_violations = 0
    failure_reasons = {'bounds': 0, 'convergence': 0, 'other': 0}
    
    for idx, row in vol_results.iterrows():
        # Use interpolation instead of dict lookup (fixes the bug!)
        F = float(fwd_interp(row['time_to_maturity']))
        K = row['strike']
        T = row['time_to_maturity']
        model_price = row['pv_model_key']
        market_vol = row['implied_normal_vol']
        
        intrinsic = max(F - K, 0)
        
        if T <= 0:
            failure_reasons['bounds'] += 1
            failed_inversions += 1
            model_vols.append(np.nan)
            vol_errors.append(np.nan)
            continue
        
        if model_price < intrinsic * 0.99:
            arbitrage_violations += 1
        
        effective_price = max(model_price, 1e-10)
        model_vol = implied_vol_from_price(F, K, effective_price, T)
        
        if np.isnan(model_vol):
            failed_inversions += 1
            failure_reasons['convergence'] += 1
            fallback_vol = effective_price / (0.3989 * np.sqrt(T) + 1e-10)
            if 0 < fallback_vol < 5.0:
                model_vols.append(fallback_vol)
                vol_errors.append(fallback_vol - market_vol)
            else:
                model_vols.append(np.nan)
                vol_errors.append(np.nan)
        else:
            model_vols.append(model_vol)
            vol_errors.append(model_vol - market_vol)
    
    vol_results[f'model_vol_{version_name}'] = model_vols
    vol_results[f'vol_error_{version_name}'] = vol_errors
    
    valid_mask = ~np.isnan(vol_errors)
    vol_rmse = np.sqrt(np.mean(np.array(vol_errors)[valid_mask]**2)) if valid_mask.any() else np.nan
    success_rate = valid_mask.sum() / len(vol_errors) * 100
    
    # Count fallback uses vs true failures
    fallback_used = failure_reasons['convergence'] - (len(vol_errors) - valid_mask.sum())
    true_failures = len(vol_errors) - valid_mask.sum()
    
    print(f"\n{'='*70}")
    print(f"{version_name.upper()} VOLATILITY SURFACE DIAGNOSTICS")
    print(f"{'='*70}")
    print(f"Total caplets:        {len(vol_errors)}")
    print(f"Valid vol inversions: {valid_mask.sum()} ({success_rate:.1f}%)")
    print(f"  - Direct Brentq:    {valid_mask.sum() - fallback_used}")
    print(f"  - Fallback ATM:     {fallback_used} (Brentq failed, used ATM approx)")
    print(f"True failures:        {true_failures} (no valid vol computed)")
    print(f"Arbitrage violations: {arbitrage_violations} (model PV < intrinsic)")
    print(f"Vol RMSE:             {vol_rmse*100:.3f}%")
    if valid_mask.any():
        valid_vols = np.array(model_vols)[valid_mask]
        print(f"Model vol range:      {valid_vols.min()*100:.2f}% - {valid_vols.max()*100:.2f}%")
        print(f"Market vol range:     {vol_results['implied_normal_vol'].min()*100:.2f}% - {vol_results['implied_normal_vol'].max()*100:.2f}%")
    print(f"{'='*70}\n")
    
    return vol_results, vol_rmse


def plot_caplet_vol_surface(vol_results, version_name="Model", plot_maturities=[1.0, 3.0, 5.0, 7.0, 10.0]):
    """
    Plot 3D volatility surface comparison: market vs model.
    
    Parameters:
    -----------
    vol_results : pd.DataFrame
        Output from generate_caplet_vol_surface() with market and model vols
    version_name : str
        Version identifier for plot titles
    plot_maturities : list
        Maturities to highlight in 2D slice plots
    """
    model_vol_col = f'model_vol_{version_name}'
    
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
    
    total_points = T_grid.size
    valid_model_points = np.sum(~np.isnan(model_vol_grid))
    coverage_pct = valid_model_points / total_points * 100
    
    print(f"\nSurface Coverage: {valid_model_points}/{total_points} points ({coverage_pct:.1f}%)")
    print(f"Missing {total_points - valid_model_points} points due to inversion failures\n")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Row 1: 3D Surfaces
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(T_grid, K_grid * 100, market_vol_grid, cmap='viridis', alpha=0.9, edgecolor='none')
    ax1.set_xlabel('Maturity (y)', fontsize=10, labelpad=5)
    ax1.set_ylabel('Strike (%)', fontsize=10, labelpad=5)
    ax1.set_zlabel('Vol (%)', fontsize=10, labelpad=5)
    ax1.set_title('Market Volatility Surface', fontsize=12, fontweight='bold')
    ax1.view_init(elev=25, azim=135)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(T_grid, K_grid * 100, model_vol_grid, cmap='viridis', alpha=0.9, edgecolor='none')
    ax2.set_xlabel('Maturity (y)', fontsize=10, labelpad=5)
    ax2.set_ylabel('Strike (%)', fontsize=10, labelpad=5)
    ax2.set_zlabel('Vol (%)', fontsize=10, labelpad=5)
    ax2.set_title(f'{version_name.upper()} Model Volatility Surface', fontsize=12, fontweight='bold')
    ax2.view_init(elev=25, azim=135)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(T_grid, K_grid * 100, error_grid, cmap='RdBu_r', alpha=0.9, edgecolor='none')
    ax3.set_xlabel('Maturity (y)', fontsize=10, labelpad=5)
    ax3.set_ylabel('Strike (%)', fontsize=10, labelpad=5)
    ax3.set_zlabel('Error (%)', fontsize=10, labelpad=5)
    ax3.set_title(f'{version_name.upper()} - Market (Vol Error)', fontsize=12, fontweight='bold')
    ax3.view_init(elev=25, azim=135)
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
    
    # Row 2: 2D Maturity Slices
    ax4 = fig.add_subplot(2, 3, 4)
    for mat in plot_maturities:
        if mat in maturities:
            subset = vol_results[vol_results['time_to_maturity'] == mat].sort_values('strike')
            ax4.plot(subset['strike'] * 100, subset['implied_normal_vol'] * 100, 
                    'o-', label=f'{mat}Y Market', alpha=0.7, linewidth=2)
    ax4.set_xlabel('Strike (%)', fontsize=11)
    ax4.set_ylabel('Implied Vol (%)', fontsize=11)
    ax4.set_title('Market Vol - Maturity Slices', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, ncol=2)
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(2, 3, 5)
    for mat in plot_maturities:
        if mat in maturities:
            subset = vol_results[vol_results['time_to_maturity'] == mat].sort_values('strike')
            valid = ~subset[model_vol_col].isna()
            ax5.plot(subset.loc[valid, 'strike'] * 100, subset.loc[valid, model_vol_col] * 100, 
                    's-', label=f'{mat}Y {version_name.upper()}', alpha=0.7, linewidth=2)
    ax5.set_xlabel('Strike (%)', fontsize=11)
    ax5.set_ylabel('Implied Vol (%)', fontsize=11)
    ax5.set_title(f'{version_name.upper()} Model Vol - Maturity Slices', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9, ncol=2)
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(2, 3, 6)
    for mat in plot_maturities:
        if mat in maturities:
            subset = vol_results[vol_results['time_to_maturity'] == mat].sort_values('strike')
            error_col = f'vol_error_{version_name}'
            valid = ~subset[error_col].isna()
            ax6.plot(subset.loc[valid, 'strike'] * 100, subset.loc[valid, error_col] * 100, 
                    '^-', label=f'{mat}Y Error', alpha=0.7, linewidth=2)
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
