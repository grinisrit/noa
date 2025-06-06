import numpy as np
from deribit_vol_surface import get_vol_surface
from pyquant.black_scholes import *
from pyquant.common import *
from pyquant.sabr import Backbone, SABRCalc
from pyquant.ssvi import SSVICalc
from pyquant.svi import SVICalc
from pyquant.vol_surface import *

# from .black_scholes import *
# from .common import *
# from .ssvi import SSVICalc
# from .svi import SVICalc
# from .vol_surface import *

vol_surface_chain_space = get_vol_surface("deribit_vol_surface.csv")
# convert to delta-space
vol_surface_delta_space: VolSurfaceDeltaSpace = SABRCalc().surface_to_delta_space(
    vol_surface_chain_space, Backbone(1.0)
)

ssvi = SSVICalc()
print(ssvi.calibrate(vol_surface_delta_space, 4))
