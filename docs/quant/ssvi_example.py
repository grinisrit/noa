import numpy as np
from pyquant.black_scholes import *
from pyquant.common import *
from pyquant.ssvi import SSVI
from pyquant.svi import SVICalc
from pyquant.vol_surface import *

strikes = np.array(
    [1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0, 2100.0, 2200.0]
)
F = 1723.75
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

vol_smile_chains = []
for tau in [0.005, 0.01, 0.1, 1]:
    forward = Forward(Spot(F), ForwardYield(0.01), DiscountYield(0.0) ,TimeToMaturity(tau))
    bs_calc = BSCalc()
    implied_vols = bs_calc.implied_vols(forward, Strikes(strikes), Premiums(pvs)).data

    vol_smile_chain = VolSmileChainSpace(
        forward, Strikes(strikes), ImpliedVols(implied_vols)
    )
    vol_smile_chains.append(vol_smile_chain)

ssvi = SSVI(vol_smile_chains, is_log=True)
ssvi.calibrate()
print([x.array()  for x in ssvi.delta_space_raw_params_list])
print([x.array()  for x in ssvi.delta_space_natural_params_list])
