import numpy as np
import numba as nb

@nb.njit()
def normal_cdf(x: nb.float64) -> nb.float64:
    t = 1 / (1 + 0.2316419 * abs(x))
    summ = (
        0.319381530 * t
        - 0.356563782 * t**2
        + 1.781477937 * t**3
        - 1.821255978 * t**4
        + 1.330274429 * t**5
    )
    if x >= 0:
        return 1 - summ * np.exp(-np.absolute(x) ** 2 / 2) / np.sqrt(2 * np.pi)
    else:
        return summ * np.exp(-np.absolute(x) ** 2 / 2) / np.sqrt(2 * np.pi)


@nb.njit()
def normal_pdf(x: nb.float64) -> nb.float64:
    probability = 1.0 / np.sqrt(2 * np.pi)
    probability *= np.exp(-0.5 * x**2)
    return probability

@nb.njit()
def np_clip(a, a_min, a_max):
    if a < a_min:
        out = a_min
    elif a > a_max:
        out = a_max
    else:
        out = a
    return out 

@nb.experimental.jitclass([
    ("sigma", nb.float64)
])
class ImpliedVol:
    def __init__(self, sigma: nb.float64):
        assert sigma > 0
        self.sigma = sigma
        
@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class ImpliedVols:
    def __init__(self, sigmas: nb.float64[:]):
        assert np.all(sigmas >= 0.)
        self.data = sigmas
        
@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Premium:
    def __init__(self, pv: nb.float64):
        self.pv = pv
        
@nb.experimental.jitclass([
    ("S", nb.float64),
    ("r", nb.float64),
    ("T", nb.float64)
])
class Forward:
    def __init__(self, spot: nb.float64, forward_yield: nb.float64, tenor: nb.float64):
        assert tenor > 0
        self.S = spot
        self.r = forward_yield
        self.T = tenor
        
    def fv(self) -> nb.float64:
        return self.S * np.exp(self.r * self.T)

@nb.experimental.jitclass([
    ("K", nb.float64)  
])
class Strike:
    def __init__(self, strike: nb.float64):
        self.K = strike
        
@nb.experimental.jitclass([
    ("data",  nb.float64[:])  
])
class Strikes:
    def __init__(self, strikes:  nb.float64[:]):
        self.data = nb.float64[:]
              

@nb.experimental.jitclass([
    ("is_call", nb.boolean)
])
class OptionType:
    def __init__(self, is_call: nb.boolean):
        self.is_call = is_call
        
@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Delta:
    def __init__(self, delta: nb.float64):
        self.pv = delta
        
@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Gamma:
    def __init__(self, gamma: nb.float64):
        self.pv = gamma

        
@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Vega:
    def __init__(self, vega: nb.float64):
        self.pv = vega
        
@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Vanna:
    def __init__(self, vanna: nb.float64):
        self.pv = vanna
        
@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Volga:
    def __init__(self, volga: nb.float64):
        self.pv = volga

     