# Interest Rate Model Calibration - Issues & Fixes

## Date: December 4, 2025
## Branch: ap_calib
## Status: 🔴 CRITICAL ISSUES IDENTIFIED - CALIBRATION NON-FUNCTIONAL

---

## Executive Summary

The interest rate model calibration has **three interconnected fundamental failures**:

1. **Autograd Pipeline Blocked**: `.detach()` calls and loop-based tensor indexing break gradient flow
2. **Martingale Property Violated**: Zero-coupon bonds computed incorrectly (wrong sign, missing deterministic components)
3. **Circular Calibration Logic**: Model curves fitted from simulations that depend on parameters being calibrated

**Impact**: Loss returns NaN, calibration fails immediately, gradient descent cannot proceed.

---

## Root Cause Analysis

### Issue 1: Broken Autodifferentiation

**Problem**: The pricing function uses operations that break PyTorch's computation graph:
- Loop-based indexed assignment: `caplet_pvs[i] = ...` 
- Premature `.detach()` calls before loss computation
- Tensor indexing in loops prevents vectorization

**Evidence**: 
```python
RuntimeError: One of the differentiated Tensors appears to not have been used in the graph
```

**Impact**: Gradients cannot flow from loss back to `model_params` and `key_ifwd_values`

### Issue 2: Incorrect Zero-Coupon Bond Computation

**Problem**: Multiple mathematical errors in ZCB calculation:

1. **Wrong discount sign**: `B_T = exp(+∫r dt)` should be `exp(-∫r dt)`
   - Bank account numeraire grows: `B_t = exp(∫r)`
   - Discount factor decays: `1/B_t = exp(-∫r)`

2. **Missing deterministic component**: ZCBs computed from pure simulation without market anchor
   ```python
   # WRONG: Pure stochastic
   zcbs = mean(exp(-sum_r_dt))
   
   # CORRECT: Market base × stochastic factor  
   zcbs = exp(-∫f_curve dt) × E[exp(-∫x_paths dt)]
   ```

3. **Martingale violation**: `P_{t,T} × B_t` must be a martingale under risk-neutral measure Q
   - Current implementation doesn't guarantee `E^Q[P_{t+dt,T}/P_{t,T}] = E^Q[B_t/B_{t+dt}]`

**Evidence**: 
- Loss values → NaN after first iteration
- Parameters explode to invalid ranges: `[-166.6, 380.4]`
- CIR variance process fails with negative values (cannot sqrt negative variance)

### Issue 3: Circular Calibration

**Problem**: The calibration loop has circular dependencies:

```
model_params → simulate → f_curve (fitted from sim) → pricing → loss → grad → model_params
                ↑______________________________________________|
```

`f_curve` and `s_curve` should be **market-fixed**, not recalibrated from simulations.

**Current (WRONG)**:
```python
# Inside simulate_model - curves depend on simulation output
f_curve = fit_from_simulation_paths(...)  # CIRCULAR!
```

**Correct**:
```python
# f_curve is market data, NOT model output
f_curve = ois_ifwd_curve.derivative(timeline)  # Market-fixed
r_paths = f_curve + x_paths  # Market base + stochastic noise
```

---

## Detailed Fix Plan

### Phase 1: Fix Autodiff Pipeline ✅ ATTEMPTED

**Target**: Enable gradient flow from loss to parameters

**Changes Made**:
1. ✅ Vectorized caplet pricing (removed loop-based indexing)
2. ✅ Delayed `.detach()` until after gradient computation
3. ✅ Changed gradient ascent (`+`) to descent (`-`)
4. ✅ Added `allow_unused=True` for unused tensors
5. ⚠️ Learning rate scaled down 1e-6× (from 0.5 to 5e-7)

**Result**: Still fails - need Phase 2 fixes for ZCB computation

### Phase 2: Correct Zero-Coupon Bond Mathematics ❌ NOT STARTED

**Target**: Implement proper martingale dynamics

**Required Changes**:

1. **Fix discount factor sign**:
   ```python
   # In price_key_caplet_surface()
   B_T = torch.exp(-sim_paths.sum_r_dt[:, vol_ids])  # NEGATIVE exponent
   payoff = torch.maximum(accrual - tau_strikes, 0.)
   model_pvs = torch.mean(payoff * B_T, dim=0)  # Discount with 1/B_T
   ```

2. **Split ZCB into market base × stochastic**:
   ```python
   # In simulate_model()
   f_curve = ois_ifwd_curve.derivative(timeline).T  # Market deterministic
   dt = timeline.diff()
   
   # Market ZCB (deterministic)
   market_zcb_log = -(f_curve[:-1] * dt).cumsum(1)
   market_zcb = torch.exp(market_zcb_log)
   
   # Stochastic factor
   stochastic_zcb_log = -(x_paths[:,:-1] * dt).cumsum(1)
   stochastic_factor = torch.exp(stochastic_zcb_log)
   
   # Combined ZCB
   zcbs = market_zcb * torch.mean(stochastic_factor, dim=0)
   ```

3. **Verify martingale property**:
   ```python
   # Add diagnostic cell
   dt = timeline.diff()
   P_t = zcbs[:, :-1]
   P_t1 = zcbs[:, 1:]
   B_t = torch.exp((r_paths[:,:-1] * dt).cumsum(1))
   B_t1 = torch.exp((r_paths[:,1:] * dt).cumsum(1))
   
   martingale = P_t1 * B_t1
   is_martingale = torch.allclose(
       torch.mean(martingale, dim=0),
       P_t[0] * B_t[0],
       rtol=0.05
   )
   print(f"Martingale check: {is_martingale}")
   ```

### Phase 3: Break Circular Calibration ❌ NOT STARTED

**Target**: Anchor base curves to market data

**Required Changes**:

1. **Fix f_curve to market OIS**:
   ```python
   # In simulate_model() - NEVER recalibrate this
   f_curve = ois_ifwd_curve.derivative(timeline).T  # Fixed to market
   ```

2. **Fix s_curve to market spread**:
   ```python
   # Spread between Key Rate and OIS
   s_curve = key_ifwd_curve.derivative(timeline).T - f_curve  # Fixed
   ```

3. **Model parameters affect ONLY stochastic components**:
   ```python
   # These are calibrated:
   v_paths = generate_cir(..., model_params[0:4])  # Variance process
   x_paths = generate_hwh(..., model_params[4], v_paths)  # OIS noise
   ks_paths = generate_hw(..., model_params[5:7])  # Key spread noise
   
   # These are market-fixed:
   r_paths = f_curve + x_paths  # OIS rate = market + noise
   s_paths = s_curve + ks_paths  # Spread = market + noise
   ```

### Phase 4: Implement T-Forward Measure Change ❌ NOT STARTED

**Target**: Correct forward rate expectations

**Current (WRONG)**:
```python
# Naive Q-measure expectation
model_fwd = torch.mean(A_T, dim=0)
```

**Correct (T-forward measure)**:
```python
# Proper measure change using Radon-Nikodym derivative
numerator = torch.mean(A_T * B_T, dim=0)
denominator = torch.mean(B_T, dim=0)
model_fwd = numerator / denominator  # E^{Q^T}[A_T] = E^Q[A_T·B_T]/E^Q[B_T]
```

**Verification**:
- Under T-forward measure, `P_{t,T}` is numeraire
- Forward rates `A_{t,T}` are martingales under Q^T
- Check: `E^{Q^T}_t[A_{t+dt,T}] = A_{t,T}`

---

## Summary of Changes Made (Current State)

### Cell-by-Cell Changes

**Successfully Applied:**

## 1. Python Path Configuration (NEW CELL) ✅

**Cell 3 (lines 17-19)**
```python
# Add parent directory to Python path for pyquant imports
import sys
sys.path.insert(0, '..')
```

**Status:** ✅ Working correctly

---

## 2. Data Loading Path Fix ✅

**Cell 5 (lines 25-35)**

**CHANGED FROM:**
```python
#data
import os
current_dir = os.getcwd()
os.chdir('../../')
fwd_ois = pd.read_csv('data/forward_ois.csv')
fwd_key_rate = pd.read_csv('data/forward_key_rate.csv')
vol_key_rate = pd.read_csv('data/volatility_key_rate.csv')
os.chdir(current_dir)
```

**CHANGED TO:**
```python
#data
import os
from pathlib import Path

# Use absolute path to project root
project_root = Path(r'c:\Personal\Education\MSc - PHDs\MIPT\Final paper\noa')
data_dir = project_root / 'data'

fwd_ois = pd.read_csv(data_dir / 'forward_ois.csv')
fwd_key_rate = pd.read_csv(data_dir / 'forward_key_rate.csv')
vol_key_rate = pd.read_csv(data_dir / 'volatility_key_rate.csv')
```

**Status:** ✅ Working correctly

---

## 3. Forward OIS Rate Plot Fix ✅

**Cell 9 (lines 145-148)**

**CHANGED FROM:**
```python
fwd_ois.set_index('tenor').sort_index().forward_rate.plot(title='Forward OIS Rate', grid=True, figsize=(20,10));
```

**CHANGED TO:**
```python
import matplotlib.pyplot as plt
fwd_ois.set_index('tenor').sort_index().forward_rate.plot(title='Forward OIS Rate', grid=True, figsize=(20,10))
plt.grid(True, alpha=0.3)
plt.show()
```

**Status:** ✅ Plot now displays correctly

---

## 4. Spread Curve Plot Fix ✅

**Cell 57 (lines 702-707)**

**CHANGED FROM:**
```python
plt.figure(figsize=(11, 6))
plt.plot(timeline, s_curve.detach().T)
plt.ylabel('Instantaneous Forward')
plt.xlabel('Time')
plt.grid()
plt.show()
```

**CHANGED TO:**
```python
plt.figure(figsize=(11, 6))
plt.plot(timeline, sim_paths.s_curve.detach().T)
plt.ylabel('Instantaneous Forward')
plt.xlabel('Time')
plt.grid()
plt.show()
```

**Status:** ✅ Working correctly

---

## 5. Price Key Caplet Surface - Vectorization ⚠️ PARTIAL

**Cell 60 (lines 728-766)**

**Cell 60 (lines 758-824)**

**CHANGED FROM:** Loop-based caplet pricing with indexed assignment
```python
for i, vol_id in enumerate(vol_ids):
    caplet_pvs[i] = ...  # Breaks autograd
```

**CHANGED TO:** Vectorized computation
```python
B_T = torch.exp(-sim_paths.sum_r_dt[:, vol_ids])  # All caplets at once
payoff = torch.maximum(accrual - tau_strikes.unsqueeze(0), 0.)
model_pvs = torch.mean(payoff * B_T, dim=0)
```

**Purpose:** 
- Remove loop-based indexing that breaks gradient flow
- Vectorize over all caplets simultaneously
- Preserve autograd computation graph

**Status:** ⚠️ Code updated but **STILL PRODUCES NaN** - needs Phase 2 fixes (ZCB mathematics)

**Remaining Issues:**
1. `sum_r_dt` and `sum_s_dt` contain NaN from simulation
2. Discount factor sign may be wrong
3. Missing market anchor in ZCB computation

---

## 6. Validation Cell Fix ✅

**Cell 62 (lines 772-790)**

**CHANGED FROM:**
```python
print(f"Combined loss: {loss.item():.6e}")
```

**CHANGED TO:**
```python
loss = loss_vol + 0.001 * loss_fwd
print(f"Volatility loss (caplet pricing):  {loss_vol.item():.6e}")
print(f"Forward rate loss:                  {loss_fwd.item():.6e}")
print(f"Combined loss:                      {loss.item():.6e}")
```

**Status:** ✅ Validation displays when data is valid

**Note:** Currently shows NaN because simulation produces NaN values

---

## 7. Calibration Loop Fix ❌ FAILED

**Cell 70 (lines 961-995)**

**Issues Identified:**
1. ❌ **Gradient direction**: Original used `+` (ascent), needs `-` (descent)
2. ❌ **Learning rate too high**: `1/i` starts at 0.5, causes parameter explosion
3. ❌ **Missing graph retention**: Needs proper iteration structure
4. ❌ **Circular dependencies**: Loss computed from old graph in second iteration

**Attempted Fixes:**
```python
# 1. Changed gradient direction
model_params = model_params.detach() - learning_rate * grad_error[0]  # Was +

# 2. Scaled learning rate
learning_rate = 1e-6 * (1/i)  # Was just 1/i (too large)

# 3. Added allow_unused for disconnected tensors
grad_error = torch.autograd.grad(loss, [...], allow_unused=True)

# 4. Restructured to recompute loss each iteration
sim_paths = simulate_model(...)  # Fresh computation graph
loss_vol, loss_fwd = price_key_caplet_surface(...)
```

**Status:** ❌ **STILL FAILS**

**Current Error:**
```
RuntimeError: Trying to backward through the graph a second time
```

**Root Cause:** `key_ifwd_values` and `model_params` retain connections to previous computation graphs. Need complete graph isolation between iterations.

**Blocker:** Cannot proceed until Phase 2 (ZCB fixes) implemented - simulation produces NaN, making calibration impossible.

---

## Debugging Additions

Added diagnostic output to track NaN propagation:

### In `simulate_model()`:
```python
if torch.any(torch.isnan(v_paths)):
    print(f"❌ ERROR: CIR paths contain NaN!")
if torch.any(torch.isnan(x_paths)):
    print(f"❌ ERROR: Hull-White-Heston paths contain NaN!")
```

### In `price_key_caplet_surface()`:
```python
if torch.any(torch.isnan(sim_paths.sum_r_dt)):
    print(f"❌ ERROR: sum_r_dt contains NaN values")
    print(f"   NaN count: {torch.isnan(sim_paths.sum_r_dt).sum().item()}")
```

**Findings:**
- NaN originates in CIR process when `v0 < 0` or `theta < 0`
- Parameters become negative after first gradient step
- Indicates learning rate still too high OR gradient direction issue OR need constraints

---

## Execution Status (Current)

**Before calibration loop:**
- ✅ Cells 1-61: All execute successfully
- ✅ Data loaded correctly
- ✅ Curves built from market data
- ✅ Initial simulation runs
- ⚠️ Initial losses: **volatility=2.04, forward=0.0315** (valid when params are positive)

**Calibration attempts:**
- ❌ Iteration 0: Loss = NaN (simulation failed)
- ❌ Parameters corrupted from previous failed run
- ❌ Need to reset and re-run setup cells before calibration

---

## Critical Path Forward

### Immediate Next Steps:

1. **Reset notebook state** ✅
   - Re-run cells 52-61 (model params through initial losses)
   - Verify initial loss is valid (~2.04)

2. **Implement Phase 2: Fix ZCB computation** ❌ NOT STARTED
   - Correct discount sign: `exp(-sum)` not `exp(+sum)`
   - Add market base: `market_zcb * stochastic_factor`
   - Verify martingale property

3. **Add parameter constraints** ❌ NOT STARTED
   - Option A: Reparameterize using `softplus` or `exp` to enforce positivity
   - Option B: Project gradients when parameters hit boundaries
   - Option C: Use constrained optimization (L-BFGS-B)

4. **Fix calibration loop graph management** ❌ NOT STARTED
   - Ensure clean detachment between iterations
   - Consider using `.clone().detach().requires_grad_()` pattern
   - May need to rebuild curves from scratch each iteration

### Long-term Fixes:

5. **Implement proper T-forward measure** (Phase 4)
6. **Add martingale verification diagnostics** (Phase 2)
7. **Optimize Monte Carlo variance** (antithetic variates, more paths)
8. **Verify CubicSpline1D differentiability**

---

## Files Modified

1. **`docs/quant/ap/interest_rates_modelling.ipynb`** - Working copy with attempted fixes
2. **`docs/quant/ap/CHANGES.md`** - This documentation file

---

## Known Blockers

1. **🔴 CRITICAL**: Simulation produces NaN when model parameters go negative
2. **🔴 CRITICAL**: Calibration loop cannot proceed - gradient graph issues
3. **🟡 HIGH**: Zero-coupon bonds not computed according to martingale dynamics
4. **🟡 HIGH**: Circular calibration - curves depend on simulation output
5. **🟡 MEDIUM**: Learning rate tuning - need adaptive or line search
6. **🟢 LOW**: Monte Carlo noise (~3% with 1000 paths)

---

## Next Action

**DO NOT** attempt calibration until:
1. ✅ Notebook state reset (cells 52-61 re-run)
2. ❌ Phase 2 ZCB fixes implemented
3. ❌ Parameter constraints added
4. ❌ Calibration loop graph isolation fixed

**Current state**: Notebook is in broken state from failed calibration attempts. Must reset before proceeding.

---

## References

**Mathematical Background:**
- T-forward measure: Brigo & Mercurio, "Interest Rate Models - Theory and Practice"
- Martingale pricing: Shreve, "Stochastic Calculus for Finance II"
- Autograd best practices: PyTorch documentation on "Autograd mechanics"

**Code Patterns:**
- Gradient descent with constraints: `torch.optim.LBFGS` documentation
- Vectorized Monte Carlo: "Differentiable Monte Carlo" papers
- Spline differentiability: Check `CubicSpline1D` source in `pyquant`

---

## Appendix: Original vs Fixed Code

### Calibration Loop Structure

**Original (BROKEN):**
```python
for i in range(2,12):
    learning_rate = 1/i  # Too high!
    loss = loss_vol + 0.001*loss_fwd  # From PREVIOUS iteration
    grad_error = torch.autograd.grad(loss, [...])  # Breaks on iteration 2
    
    # WRONG: Gradient ASCENT (minimizing should use -)
    model_params = model_params.detach() + learning_rate * grad_error[0]
    
    # Recompute for NEXT iteration (circular dependency)
    sim_paths = simulate_model(...)
    loss_vol, loss_fwd = price_key_caplet_surface(...)
```

**Attempted Fix (STILL BROKEN):**
```python
# Initialize clean state
model_params = model_params.detach().requires_grad_()
key_ifwd_values = key_ifwd_values.detach().requires_grad_()

for i in range(2,12):
    learning_rate = 1e-6 * (1/i)  # Scaled down
    
    # Recompute loss at START with fresh graph
    sim_paths = simulate_model(...)
    loss_vol, loss_fwd = price_key_caplet_surface(...)
    loss = loss_vol + 0.001*loss_fwd
    
    grad_error = torch.autograd.grad(loss, [...], allow_unused=True)
    
    # FIXED: Gradient DESCENT
    if grad_error[0] is not None:
        model_params = model_params.detach() - learning_rate * grad_error[0]
        model_params.requires_grad_()
```

**Still fails because:** 
- Simulation produces NaN (need Phase 2 fixes)
- Graph retention issues with `key_ifwd_values`

---

## End of Document
