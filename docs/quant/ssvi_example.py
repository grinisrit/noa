import numpy as np
from pyquant.black_scholes import *
from pyquant.common import *
from pyquant.ssvi import SSVICalc
from pyquant.svi import SVICalc
from pyquant.vol_surface import *

strikes = np.array(
    [1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0, 2100.0, 2200.0]
)
premiums = np.array(
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
spot = 1723.75
ttms = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])

forward_rates = np.array(
    [1700.0, 1702.0, 1703.0, 1704.0, 1705.0, 1736.0, 1737.0, 1738.0, 1739.0, 1740.0]
)

forward_curve: ForwardCurve = forward_curve_from_forward_rates(
    Spot(spot), ForwardRates(forward_rates), TimesToMaturity(ttms)
)


vol_surface_chain_space = VolSurfaceChainSpace(
    forward_curve=forward_curve,
    times_to_maturity=TimesToMaturity(ttms),
    strikes=Strikes(strikes),
    option_types=OptionTypes(np.array([True if x > spot else False for x in strikes])),
    premiums=Premiums(premiums),
    compute_implied_vol=True,
)
# convert to delta-space
vol_surface_delta_space: VolSurfaceDeltaSpace = SVICalc().surface_to_delta_space(
    vol_surface_chain_space
)

ssvi = SSVICalc(is_log=True)
ssvi.calibrate(vol_surface_delta_space)
