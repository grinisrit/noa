# HW-Heston Caplet Calibration — Version History

## Project Overview

Hull-White/Heston stochastic interest rate model for calibrating normal (Bachelier)
implied volatility surfaces of average-rate caplets. The model runs on GPU via
PyTorch CUDA tensors and uses SPSA (Simultaneous Perturbation Stochastic
Approximation) for derivative-free calibration.

### Model Components

The instantaneous key rate decomposes as:

$$a(t) = f_{\text{key}}(t) + x(t) + k(t)$$

| Process | SDE | Role |
|---------|-----|------|
| **CIR variance** | $dv = \kappa(\theta(t) - v)\,dt + \varepsilon\sqrt{v}\,dZ_v$ | Stochastic volatility — controls skew |
| **OU rate innovation** | $dx = -\lambda x\,dt + \sqrt{v}\,dZ_x$ | Mean-reverting rate perturbation, vol from CIR |
| **HW spread** | $dk = -\gamma k\,dt + \xi\,dZ_k$ | Constant-vol spread, independent of vol |

Correlation structure (Cholesky):
- $W_v = Z_v$
- $W_x = \rho(t) Z_v + \sqrt{1-\rho(t)^2} Z_x$ — rate innovation correlated with vol
- $W_k = Z_k$ — spread independent

### Caplet Pricing

Average-rate caplet payoff at maturity $T$:

$$\text{payoff} = \max\!\left(\int_0^T a(t)\,dt - T K,\; 0\right)$$

Priced in the money-market measure via:

$$PV = E\!\left[e^{-\int_0^T r^{\text{ois}}(t)\,dt} \cdot \text{payoff}\right]$$

Bachelier implied vol inversion uses a **stable time-value (TV) channel**:

$$TV = \sigma\hat{g}\left[\varphi(d) - d\,\Phi(-d)\right], \quad \hat{g} = \sqrt{T/3}, \quad d = \frac{F - K}{\sigma\hat{g}}$$

where $\Phi(-d)$ is computed via `scipy.stats.norm.sf(d)` for deep-ITM stability.

---

## Version Timeline

### v6 — Baseline (Reference in `vol_surface_review.ipynb`)

The first stable calibrated parameter set, used as the reference baseline for
all subsequent verification and round-trip testing.

**Parameters (scalar, fixed):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\kappa$ | 3.25 | CIR mean-reversion speed |
| $\varepsilon$ | 0.274 | CIR vol-of-vol |
| $\lambda$ | 0.219 | OU mean-reversion speed |
| $\gamma$ | 0.5 | Spread OU mean-reversion |
| $\xi$ | 0.01 | Spread constant vol |
| $\rho$ | 0.267 | Vol-rate correlation (positive: rates rise when vol rises) |
| $v_0$ | $\theta(0)$ | Initial CIR variance set at first $\theta$ node |

**$\theta(t)$ construction:**
- 7 nodes at $t \in \{0, 0.25, 0.5, 1.0, 3.0, 5.0, 10.0\}$ years
- Anchored at $t=0$: $\theta(0) = \theta(0.25)$ to prevent PCHIP extrapolation artefacts
- Analytically solved: $\theta(T) = \sigma^2_{\text{ATM}} / \text{corr}(\lambda, T)$
- OU correction factor:

$$\text{corr}(\lambda, T) = \frac{3}{T^3}\left[\frac{T}{\lambda^2} - \frac{2(1-e^{-\lambda T})}{\lambda^3} + \frac{1-e^{-2\lambda T}}{2\lambda^3}\right]$$

This converts CIR variance $\theta$ to Bachelier normal vol $\sigma_B$ for average-rate caplets:
$\sigma^2_B = \theta \cdot \text{corr}(\lambda, T)$.

**Monte Carlo:** 100,000 paths with antithetic variates (200k effective).

**Key characteristics:**
- $\kappa = 3.25$ gives CIR half-life $\ln 2 / 3.25 = 0.21$ years
- Stochastic vol essentially dies after ~1 year
- Scalar $\rho$ — same vol-rate correlation at all maturities
- Round-trip verification: all caplets recovered via stable TV channel (zero failures)

---

### v7 — Intermediate (Referenced only; warm-start source for v8)

v7 is not saved as a standalone parameter set but is referenced as the warm-start
source in the SPSA v8 calibration cell.

**Known changes from v6:**
- $\kappa = 3.2$ (slightly lower, half-life $\approx 0.22$ years)
- Still suffered from "stochastic vol dead beyond 1Y"
- No long-end skew: CIR variance mean-reverts too quickly

**Role:** v7's calibrated `best_params` dictionary is loaded as the warm-start
initialization for v8, with parameters clamped to v8's tighter bounds.

---

### v8 — Current Production (SPSA calibration in `fast_calibration.ipynb`)

Major overhaul targeting persistent stochastic volatility and term-structure of skew.

**KEY FIX:** Force persistent stochastic volatility via tight $\kappa$ bounds.

| Problem (v7) | Solution (v8) |
|---------------|---------------|
| $\kappa = 3.2 \Rightarrow$ CIR half-life 0.21Y | $\kappa \in [0.3, 1.5] \Rightarrow$ half-life 0.46–2.3Y |
| Stoch vol dead beyond 1Y | Stoch vol persists to 5–10Y |
| No long-end skew | Long-dated skew preserved |
| Scalar $\rho = 0.267$ | Time-dependent $\rho(t)$ with 3 nodes |

**Parameter bounds (15 total):**

| # | Parameter | Bounds | Count | Description |
|---|-----------|--------|-------|-------------|
| 1–6 | $\theta(T_i)$ | [0.0002, 0.04] | 6 | CIR long-run variance at nodes 0.25, 0.5, 1, 3, 5, 10Y |
| 7 | $v_0$ | [1e-5, 0.04] | 1 | Initial CIR variance |
| 8 | $\kappa$ | [0.3, 1.5] | 1 | CIR mean-reversion (tightened) |
| 9 | $\varepsilon$ | [0.15, 1.0] | 1 | Vol-of-vol |
| 10 | $\lambda$ | [0.01, 1.0] | 1 | OU mean-reversion |
| 11 | $\gamma$ | [0.01, 3.0] | 1 | Spread mean-reversion |
| 12 | $\xi$ | [1e-5, 0.01] | 1 | Spread vol |
| 13–15 | $\rho(T_j)$ | [-0.95, 0.95] | 3 | Vol-rate correlation at 1, 5, 10Y |

**SPSA configuration (Spall 1992, 1998):**

| Setting | Value |
|---------|-------|
| Iterations | 3,000 |
| Gradient paths | 6,000 (per forward/backward) |
| Evaluation paths | 12,000 |
| $a$ (step scale) | 0.06 |
| $c$ (perturbation scale) | 0.03 |
| $\alpha$ (step decay) | 0.602 |
| $\gamma_{\text{SPSA}}$ (pert. decay) | 0.101 |
| $A$ (stability constant) | $0.1 \times$ max_iter = 300 |
| Total MC paths | ~36 million over full calibration |

**Efficiency:** SPSA requires only 2 MC evaluations per iteration (forward +
backward perturbation) vs. $2p$ for central finite-differences ($p = 15$
parameters). This allows 7.5× more paths per evaluation at the same total cost.

**Loss function (vega-weighted):**

$$L = \sqrt{\frac{1}{N}\sum_{i \in \text{calib}} \left(\frac{PV^{\text{model}}_i - PV^{\text{market}}_i}{\text{vega}^{\text{market}}_i}\right)^2} + 0.1\sum_j (\theta_{j+1} - \theta_j)^2$$

- Vega weighting normalizes PV differences to implied-vol space
- Vega floor at 5% of median prevents far-OTM blowup
- Smoothness penalty on adjacent $\theta$ nodes
- **Moneyness filter:** Exclude deep ITM caplets with $K < F - 3\%$

**Warm start from v7:**
- Load v7's `best_params` dictionary
- Clamp each parameter to v8's (possibly tighter) bounds
- Provides faster convergence than random initialization

**Correlation evolution:**

| Aspect | v6 | v8 |
|--------|----|----|
| Form | Scalar $\rho = 0.267$ | Piecewise linear $\rho(t)$ |
| Nodes | — | 3 nodes at 1Y, 5Y, 10Y |
| Interpolation | — | Linear between nodes, flat extrapolation |
| Effect | Uniform skew | Term-structure of skew: different vol-rate coupling at each maturity |

---

## T-Forward Measure Pricing

Introduced to guarantee that Bachelier inversion is well-posed.

**Definitions:**
- $P^{\text{model}} = E\!\left[e^{-\int_0^T r\,dt}\right]$ — model zero-coupon bond
- $F^{\text{model}} = \frac{E\!\left[e^{-\int_0^T r\,dt} \cdot \bar{a}\right]}{P^{\text{model}}}$ — T-forward measure forward rate
- where $\bar{a} = \frac{1}{T}\int_0^T a(t)\,dt$ is the realized average rate

**Jensen inequality guarantee:**

$$E^T\!\left[\max(\bar{a} - K, 0)\right] \geq \max(F^{\text{model}} - K, 0)$$

This ensures the undiscounted model price is always at least the intrinsic value
when evaluated at $F^{\text{model}}$, making Bachelier inversion from time-value
numerically stable even for deep ITM caplets.

**Forward rate distinction:**
- $F_{\text{period}}$ — simple-compounding market quote: $P = 1/(1 + T F_{\text{period}})$
- $F_{\text{avg}} = I(0,T)/T = -\ln P(0,T)/T$ — average instantaneous forward
- They differ by Jensen's inequality: $-\ln(1+Tx)/(Tx) < 1$ for $x > 0$
- **Both calibration and MC use $F_{\text{avg}}$** because the caplet payoff
  involves $\int_0^T a(t)\,dt$ and $E[\int a(t)\,dt] = I(0,T) = T \cdot F_{\text{avg}}$

---

## GPU Acceleration

All tensors reside on GPU (PyTorch CUDA). The speedup comes from **path-level
parallelism**: each SPSA evaluation launches 6,000–12,000 MC paths that execute
simultaneously, while time steps remain sequential due to SDE causality.

- **CIR / OU paths** — Python loop over time steps; each step is a vectorized
  tensor op across all paths on GPU
- **HW spread** — fully vectorized via `cumsum` (constant vol allows analytical
  unrolling), parallel in both time and path dimensions
- **Batch pricing** — all caplets priced simultaneously via cumulative integrals
  and tensor indexing
- **$\theta(t)$ interpolation** — GPU-native `PchipSpline1D`, avoiding CPU↔GPU
  round-trips

---

## Bachelier Implied Vol Inversion

### The Deep-ITM Problem

For deep in-the-money caplets (strike far below forward), the Bachelier price is
almost entirely intrinsic value:

$$PV \approx T \cdot P(0,T) \cdot (F - K)$$

Standard root-finding on total price suffers from catastrophic cancellation:
the solver tries to match a tiny time-value residual on top of a large intrinsic.

### Stable Time-Value Channel

**Solution:** Decompose price as $PV = \text{intrinsic} + \text{time\_value}$ and
invert only the time-value component:

$$TV = \sigma\hat{g}\left[\varphi(d) - d\,\Phi(-d)\right]$$

- $\varphi(d)$ — standard normal PDF
- $\Phi(-d)$ computed via `scipy.stats.norm.sf(d)` — accurate even for $d > 30$
- The function $h(d) = \varphi(d) - d\,\Phi(-d)$ is smooth, positive, and monotonically
  decreasing — ideal for root-finding

**Inversion pipeline:**
1. Compute $TV$ from model price (strip intrinsic and discounting)
2. Solve $TV = \sigma\hat{g} \cdot h\!\left(\frac{F-K}{\sigma\hat{g}}\right)$ for $\sigma$ via Brentq
3. Falls back to standard PV-based inversion if TV method returns NaN

**Round-trip verification:** Market $\sigma \to PV \to \sigma_{\text{out}}$ achieves near-zero
error (~$10^{-8}\%$ RMSE) across all caplets including deep ITM.

---

## Arbitrage Checks

Market vol surface is validated for no-arbitrage using Dupire local vol conditions:

- **Calendar spread:** $\frac{\partial w}{\partial T} \geq 0$ where $w = \sigma^2 T$ (total variance non-decreasing)
- **Butterfly spread:** $\frac{\partial^2 C}{\partial K^2} \geq 0$ (price convexity in strike)
- **Local vol:** $\sigma^2_{\text{loc}} = \frac{\partial C / \partial T}{0.5 \cdot \partial^2 C / \partial K^2} \geq 0$

Numerical derivatives use **Richardson extrapolation** (steps $h$ and $h/2$ averaged for
4th-order accuracy).

---

## Codebase Architecture

### Files

| File | Purpose |
|------|---------|
| `caplet_vol_surface.py` | Shared module: all pricing, vol inversion, surface generation, simulation, and plotting |
| `fast_calibration.ipynb` | SPSA v8 calibration notebook |
| `calibrated_surface_review.ipynb` | Post-calibration review: loads checkpoint, simulates, compares model vs market vol surface, arbitrage checks |
| `vol_surface_review.ipynb` | Formula verification: round-trips, forward rate checks, v6 baseline reference |
| `single_caplet_testing.ipynb` | Single caplet pricing tests |
| `multi_theta_base_testing.ipynb` | Multi-theta base testing |

### Module Organization (`caplet_vol_surface.py`)

| Section | Functions | Lines |
|---------|-----------|-------|
| Bachelier pricing | `bachelier_caplet_price`, `bachelier_caplet_time_value`, `_bachelier_tv_undiscounted` | ~26–118 |
| Vol inversion | `implied_vol_avg_rate`, `implied_vol_from_tv` | ~168–241 |
| Arbitrage | `compute_total_variance_conditions`, `compute_dupire_conditions`, `check_surface_arbitrage`, `check_market_arbitrage` | ~262–619 |
| Plotting (arbitrage) | `plot_arbitrage_heatmaps`, `print_arbitrage_summary` | ~427–564 |
| Surface generation | `generate_caplet_vol_surface` (accepts optional F_model/P_model) | ~644–760 |
| Plotting (vol surface) | `plot_caplet_vol_surface` | ~763–893 |
| Diagnostics | `plot_caplet_price_heatmaps`, `print_model_vs_market_table` | ~896–1131 |
| **Simulation** | `fast_cir_paths`, `fast_ou_paths`, `fast_hw_paths` | ~1335–1383 |
| **Interpolation** | `rho_to_vec` (piecewise linear), `theta_to_vec` (PCHIP) | ~1345–1359 |
| **Combined sim** | `fast_simulate` (Cholesky correlation, antithetic) | ~1386–1448 |
| **Batch pricing** | `batch_price_caplets` (T-forward measure) | ~1451–1473 |

### Dependencies

- **PyTorch** (CUDA) — all tensor operations on GPU
- **NumPy** — analytics and parameter initialization
- **SciPy** — `norm.sf` for deep-ITM inversion, `brentq` root-finding, `PchipInterpolator`
- **PchipSpline1D** (custom) — GPU-native monotone cubic spline for $\theta(t)$
- **Pandas** — data loading and result tables
- **Matplotlib** — 3D surfaces and smile slices

---

## Known Issues

### Deep ITM Short-Maturity Volatility Explosion

The calibrated model produces extremely high implied volatility for short-maturity
($< 1$ year) deep-ITM strikes. Root causes under investigation:

- CIR variance + OU process may generate excessive variance at short maturities
  for far-below-ATM strikes
- SPSA's moneyness filter (3% ITM threshold) explicitly excludes deep ITM from
  calibration, so model behavior in that region is unconstrained
- The round-trip $\sigma \to PV \to \sigma$ works correctly (pricing/inversion
  machinery is sound) — the issue is the model dynamics themselves


