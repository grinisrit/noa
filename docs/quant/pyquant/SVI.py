import numpy as np
import numba as nb

from .common import *
from .black_scholes import *
from .vol_surface import *

@nb.experimental.jitclass([
    ("a", nb.float64)
])
class A:
    def __init__(self, a: nb.float64):
        self.a = a

@nb.experimental.jitclass([
    ("b", nb.float64)
])
class B:
    def __init__(self, b: nb.float64):
        if not (b >= 0):
            raise ValueError('B not >= 0')
        self.b = b

@nb.experimental.jitclass([
    ("rho", nb.float64)
])
class Rho:
    def __init__(self, rho: nb.float64):
        if not (rho >= -1 and rho <= 1):
            raise ValueError('Rho not from [-1, 1]')
        self.rho = rho

@nb.experimental.jitclass([
    ("m", nb.float64)
])
class M:
    def __init__(self, m: nb.float64):
        self.m = m

@nb.experimental.jitclass([
    ("sigma", nb.float64)
])
class Sigma:
    def __init__(self, sigma: nb.float64):
        if not (sigma > 0):
            raise ValueError('Sigma not > 0')
        self.sigma = sigma

@nb.experimental.jitclass([
    ("a", nb.float64),
    ("b", nb.float64),
    ("rho", nb.float64),
    ("m", nb.float64),
    ("sigma", nb.float64)
])
class SABRParams:
    def __init__(
        self, a: A, b: B, rho: Rho, m: M, sigma: Sigma
    ): 
        self.a = a.a
        self.b = b.b
        self.rho = rho.rho
        self.m = m.m
        self.sigma = sigma.sigma
    
    def array(self) -> nb.float64[:]:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])
    
