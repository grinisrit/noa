import numpy as np
import numba as nb

from .common import *
from .black_scholes import *
from .vol_surface import *


@nb.experimental.jitclass([
    ("beta", nb.float64)
])
class Backbone:
    def __init__(self, beta: nb.float64):
        assert beta <= 1
        assert beta >= 0
        self.beta = beta


@nb.experimental.jitclass([
    ("alpha", nb.float64)
])
class Volatility:
    def __init__(self, alpha: nb.float64):
        self.alpha = alpha 


@nb.experimental.jitclass([
    ("rho", nb.float64)
])
class Correlation:
    def __init__(self, rho: nb.float64):
        self.rho = rho 


@nb.experimental.jitclass([
    ("v", nb.float64)
])
class VolOfVol:
    def __init__(self, v: nb.float64):
        self.v = v 


@nb.experimental.jitclass([
    ("alpha", nb.float64),
    ("rho", nb.float64),
    ("v", nb.float64)
])
class SABRParams:
    def __init__(
        self, volatility: Volatility, correlation: Correlation, volvol: VolOfVol
    ): 
        self.alpha = volatility.alpha
        self.rho = correlation.rho
        self.v = volvol.v

        
@nb.experimental.jitclass([
    ("beta", nb.float64)
])
class Backbone:
    def __init__(self, beta: nb.float64):
        assert beta <= 1
        assert beta >= 0
        self.beta = beta
     
        
@nb.experimental.jitclass([
    ("f", nb.float64),
    ("T", nb.float64),
    ("beta", nb.float64)
])        
class SABR:
    def __init__(self, forward: Forward, backbone: Backbone):
        self.f = forward.forward_rate().fv
        self.T = forward.T
        self.beta = backbone.beta

    def calibrate(self, chain: VolSmileChain) -> SABRParams:
        pass

    def delta_space(self, params: SABRParams) -> VolSmileDeltaSpace:
        pass

    def implied_vol_for_strike(self, params: SABRParams, strike: Strike) -> ImpliedVol:
        pass

    def implied_vol_for_delta(self, params: SABRParams, delta: Delta) -> ImpliedVol:
        pass
    
    
    

   