import numpy as np
import numba as nb

from .black_scholes import *

@nb.experimental.jitclass([
    ("alpha", nb.float64),
    ("rho", nb.float64),
    ("v", nb.float64)
])
class SABRParams:
    def __init__(
        self, alpha: nb.float64, rho: nb.float64, volvol: nb.float64, beta: nb.float64
    ):
        assert alpha > 0
        assert rho <= 1
        assert rho >= -1
        assert volvol >= 0
        assert beta <= 1
        assert beta >= 0
        
        self.alpha = alpha
        self.v = v
        self.beta = beta
        self.rho = rho

        
@nb.experimental.jitclass([
    ("beta", nb.float64)
])
class Backbone:
    def __init__(self, beta: nb.float64):
        assert beta <= 1
        assert beta >= 0
        self.beta = beta

        
@nb.experimental.jitclass([
    ("forward", nb.float64),
    ("strikes", nb.float64[:]),
    ("implied_vols", nb.float64[:]),
])
class VolSmileChain:
    def __init__(self, forward: Forward, strikes: nb.float64[:], implied_vols: nb.float64[:]):
        assert strikes.shape == implied_vols.shape
        self.forward = forward  
        self.strikes = strikes
        self.implied_vols = implied_vols    
        
        
@nb.experimental.jitclass([
    ("beta", nb.float64)
])        
class SABR:
    def __init__(self, backbone: Backbone):
        self.beta = backbone.beta
       
