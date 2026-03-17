# Affine Processes for Interest Rate Modeling

**Thesis Topic:** Modelling of Multi-Curve Term Structures and RFR Volatility Surfaces Using Affine Dynamics  
**Last Updated:** February 2026

---

## 📁 Files Overview

| File | Purpose | Status |
|------|---------|--------|
| `interest_rates_ap.ipynb` | Main experimental notebook (scalar parameters) | ✅ Active |
| `multi_theta_model.ipynb` | Time-varying θ(t) extension for volatility term structure | ✅ Active |
| `hwh_model_guide.ipynb` | Theory guide: HWH model, CIR process, measure changes | 📚 Reference |
| `archive/multi_curve_pricing.ipynb` | Original reference copy (pre-optimization) | 📦 Archived |

---

## 🔑 Model Hierarchy

```
Hull-White (constant σ)
     │
     │ Add stochastic variance (CIR process)
     ▼
Hull-White-Heston (HWH)
     │
     │ Add time-varying θ(t), ε(t)
     ▼
Multi-Theta Model (this folder)
     │
     │ Add Key Rate spread process
     ▼
Multi-Curve HWH (OIS + Key Rate)
```

---

## 📐 Model Specification

### Core Dynamics (Risk-Neutral Measure Q)

**OIS Short Rate:**
$$r_t = f(t) + x_t, \quad dx_t = -\lambda x_t \, dt + \sqrt{v_t} \, dW_t^x$$

**Stochastic Variance (CIR):**
$$dv_t = \kappa(\theta(t) - v_t) \, dt + \varepsilon \sqrt{v_t} \, dW_t^v$$

**Key Rate Spread:**
$$s_t = s^a(t) + k_t, \quad dk_t = -\gamma k_t \, dt + \xi \, dW_t^k$$

All Brownian motions are **independent** ($dW^x \cdot dW^v = dW^x \cdot dW^k = 0$).

### Parameters

| Symbol | Name | Process | Calibrated? |
|--------|------|---------|-------------|
| $v_0$ | Initial variance | CIR | ✅ |
| $\kappa$ | Variance mean reversion | CIR | ✅ |
| $\theta(t)$ | Time-varying mean level | CIR | ✅ (at market maturities) |
| $\varepsilon$ | Vol-of-vol | CIR | ✅ |
| $\lambda$ | Rate mean reversion | HW | ✅ |
| $\gamma$ | Spread mean reversion | Key Rate | ✅ |
| $\xi$ | Spread volatility | Key Rate | ✅ |
| $f(t)$ | OIS instantaneous forward | Deterministic | Fixed (HJM) |
| $s^a(t)$ | Spread curve | Deterministic | Fixed (market) |

---

## 🚀 Quick Start

### 1. Theory Overview
Open `hwh_model_guide.ipynb` for:
- CIR process properties (Feller condition, χ² distribution)
- Hull-White-Heston dynamics
- Measure changes (Q → Q^T forward measure)
- Simulation code examples

### 2. Experimental Work
Open `interest_rates_ap.ipynb` for:
- Scalar parameter calibration
- Vasicek/HWH Monte Carlo illustrations
- Backward-looking rate calculations (R, A)
- Caplet pricing

### 3. Multi-Theta Extension
Open `multi_theta_model.ipynb` for:
- Time-varying θ(t) at market maturities
- PCHIP interpolation for smooth curves
- Volatility surface calibration
- Comparison: Linear vs PCHIP interpolation

---

## 📊 Key Results

### Volatility Surface Fitting

The multi-theta model with PCHIP interpolation achieves:
- **Fit to market:** Vol RMSE ~ 0.5-2% (depending on calibration settings)
- **Parameters:** 13 θ values + 1 ε (constant) + 4 fixed parameters
- **Computation:** ~3-5 seconds per calibration (LBFGS optimizer)

### Approximate Implied Vol Mapping

$$\sigma_N(T) \propto \sqrt{\theta(T) \cdot \varepsilon}$$

- $\theta(T)$: Controls volatility **level** at maturity T
- $\varepsilon$: Controls **curvature** (fat tails via vol-of-vol)
- Product captures combined uncertainty

---

## 📝 Data Requirements

All notebooks use data from `../../../data/`:
- `forward_ois.csv` - OIS forward rates
- `forward_key_rate.csv` - Key rate forwards  
- `volatility_key_rate.csv` - Implied normal volatility surface

---

## 🔧 Technical Notes

### CIR Positivity (Feller Condition)
$$2\kappa\theta > \varepsilon^2$$
Must hold for all $t$ to ensure $v_t > 0$.

### Timeline Requirements
- **Daily (3651 steps)**: Required for accurate average rate calculation
- **Monthly (121 steps)**: Sufficient for quick calibration testing

### Simulation Methods
- **CIR:** QE (Quadratic Exponential) scheme for accurate non-central χ² sampling
- **HWH:** Exact OU transition with stochastic variance input
- **Spread:** Standard OU exact transition

---

## 📚 References

1. **Hull-White (1990)**: Pricing Interest-Rate-Derivative Securities
2. **Cox-Ingersoll-Ross (1985)**: A Theory of the Term Structure of Interest Rates
3. **Heston (1993)**: A Closed-Form Solution for Options with Stochastic Volatility
4. **Lyashenko-Mercurio (2019)**: Looking Forward to Backward-Looking Rates

---

## 📋 Historical Notes

### Resolved Issues (Dec 2025)
- ✅ Autodiff pipeline fixed (vectorized caplet pricing)
- ✅ ZCB computation corrected (proper discount factor signs)
- ✅ Circular calibration broken (market curves fixed, not recalibrated)
- ✅ Performance optimized (9000× speedup via coarse grids + reduced paths)

### Current Development (Feb 2026)
- ✅ Added HWH theory guide notebook
- ✅ Updated multi_theta_model with comprehensive documentation
- ✅ PCHIP interpolation for time-varying parameters
- 🔄 Investigating time-varying ε(t) in addition to θ(t)
