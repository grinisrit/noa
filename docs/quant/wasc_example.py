import numpy as np
from pyquant.black_scholes import *
from pyquant.common import *
from pyquant.vol_surface import *
from pyquant.wasc import WASC

strikes = np.array([1300.0, 1400, 1500, 1600, 1700, 1800.0, 1900, 2000, 2100, 2200])
tau = 0.01
F = 1723.75
forward = Forward(Spot(F), ForwardYield(0.01), TimeToMaturity(tau))

pvs = np.array(
    [
        1.72375,
        1.72375,
        3.4475,
        6.895,
        26.718125,
        11.204375,
        4.309375,
        1.72375,
        0.861875,
        0.861875,
    ]
)
bs_calc = BSCalc()
implied_vols = bs_calc.implied_vols(forward, Strikes(strikes), Premiums(pvs)).data
# need to sqare, as we calibrate to sigma**2
implied_vols = np.array([x**2 for x in implied_vols])

vol_smile_chain = VolSmileChainSpace(
    forward, Strikes(strikes), ImpliedVols(implied_vols)
)

weights = CalibrationWeights(np.ones_like(vol_smile_chain.Ks))

for params_dim in [1, 2, 3, 4, 5, 6]:
    print(f"===== Calibration for matrix dim {params_dim} =====")
    wasc = WASC(params_dim=params_dim, is_log=False, params_init_type="normal_diag")
    clip_params, calib_error = wasc.calibrate(vol_smile_chain, weights)
    print("=== R matrix ===")
    print(clip_params.R)
    print("=== Q matrix ===")
    print(clip_params.Q)
    print("=== Sigma matrix ===")
    print(clip_params.sigma)
    print("=== Calibration normalized MSE ===")
    print(calib_error.v)
    print("Market IV", implied_vols)
    print("Got IV   ", wasc._vol_wasc(F, strikes, clip_params))
