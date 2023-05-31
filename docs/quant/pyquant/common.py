import numpy as np
import numba as nb

from .utils import *


@nb.experimental.jitclass([
    ("sigma", nb.float64)
])
class ImpliedVol:
    def __init__(self, sigma: nb.float64):
        if not sigma > 0:
            raise ValueError('Non-positive implied vol')
        self.sigma = sigma
                
@nb.experimental.jitclass([
    ("sigma", nb.float64)
])
class VolatilityQuote:
    def __init__(self, sigma: nb.float64):
        self.sigma = sigma


@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class ImpliedVols:
    def __init__(self, sigmas: nb.float64[:]):
        if not np.all(sigmas > 0.):
            raise ValueError('Not all implied vols are positive')
        self.data = sigmas
   

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class VolatilityQuotes:
    def __init__(self, sigmas: nb.float64):
        self.data = sigmas


@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Premium:
    def __init__(self, pv: nb.float64):
        self.pv = pv

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class Premiums:
    def __init__(self, pvs: nb.float64[:]):
        self.data = pvs


@nb.experimental.jitclass([
    ("S", nb.float64)
])
class Spot:
    def __init__(self, spot: nb.float64):
        self.S = spot


@nb.experimental.jitclass([
    ("r", nb.float64)
])
class ForwardYield:
    def __init__(self, rate: nb.float64):
        self.r = rate


@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Numeraire:
    def __init__(self, numeraire: nb.float64):
        if not numeraire > 0:
            raise ValueError('Non-positive numeraire')
        self.pv = numeraire


@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class ForwardYields:
    def __init__(self, rates: nb.float64[:]):
        self.data = rates


@nb.experimental.jitclass([
    ("T", nb.float64)
])
class TimeToMaturity:
    def __init__(self, time_to_maturity: nb.float64):
        if not time_to_maturity > 0:
            raise ValueError('Non-positive time to maturity')
        self.T = time_to_maturity


@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class TimesToMaturity:
    def __init__(self, T: nb.float64[:]):
        if not np.all(T >= 0):
            raise ValueError('Not all times to maturity are positive')
        self.data = T      


@nb.experimental.jitclass([
    ("r", nb.float64[:]),
    ("T", nb.float64[:])
])
class ForwardYieldCurve:
    def __init__(self, forward_yields: ForwardYields, times_to_maturity: TimesToMaturity):
        if not forward_yields.data.shape == times_to_maturity.data.shape:
            raise ValueError('Inconsistent data between yields and times to maturity')
        self.r = forward_yields.data
        self.T = times_to_maturity.data


@nb.experimental.jitclass([
    ("fv", nb.float64)
])
class ForwardRate:
    def __init__(self, forward: nb.float64):
        self.fv = forward


@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class ForwardRates:
    def __init__(self, forwards: nb.float64[:]):
        self.data = forwards


@nb.experimental.jitclass([
    ("S", nb.float64),
    ("r", nb.float64),
    ("T", nb.float64)
])
class Forward:
    def __init__(self, spot: Spot, forward_yield: ForwardYield, time_to_maturity: TimeToMaturity):
        self.S = spot.S
        self.r = forward_yield.r
        self.T = time_to_maturity.T
        
    def forward_rate(self) -> ForwardRate:
        return ForwardRate(self.S * np.exp(self.r * self.T))
    
    def numeraire(self) -> Numeraire:
        return Numeraire(np.exp(-self.r * self.T))
    
    @staticmethod
    def from_forward_rate(spot: Spot, forward_rate: ForwardRate, time_to_maturity: TimeToMaturity):
        return Forward(
            spot, ForwardYield(- np.log(spot.S / forward_rate.fv)/ time_to_maturity.T), time_to_maturity
            )

    
@nb.experimental.jitclass([
    ("S", nb.float64),
    ("r", nb.float64[:]),
    ("T", nb.float64[:])
])
class ForwardCurve:
    def __init__(self, spot: Spot, forward_yields: ForwardYieldCurve):
        if not is_sorted(forward_yields.T):
            raise ValueError('Tenors are not ordered')
        self.S = spot.S
        self.r = forward_yields.r
        self.T = forward_yields.T
        
    def forward_rates(self) -> ForwardRates:
        return ForwardRates(self.S * np.exp(self.r * self.T))
    
    @staticmethod
    def from_forward_rates(spot: Spot, forward_rates: ForwardRates, times_to_maturity: TimesToMaturity):
        return ForwardCurve(
            spot, ForwardYieldCurve( 
                    ForwardYields(- np.log(spot.S / forward_rates.data) / times_to_maturity.data),
                    times_to_maturity
                )   
            )


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
        self.data = strikes
              

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
    ("data", nb.float64[:])
])
class Deltas:
    def __init__(self, deltas: nb.float64[:]):
        self.data = deltas


@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Gamma:
    def __init__(self, gamma: nb.float64):
        self.pv = gamma

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class Gammas:
    def __init__(self, gammas: nb.float64[:]):
        self.data = gammas

        
@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Vega:
    def __init__(self, vega: nb.float64):
        self.pv = vega

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class Vegas:
    def __init__(self, vegas: nb.float64):
        self.data = vegas

@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Rega:
    def __init__(self, rega: nb.float64):
        self.pv = rega

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class Regas:
    def __init__(self, regas: nb.float64):
        self.data = regas


@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Sega:
    def __init__(self, sega: nb.float64):
        self.pv = sega

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class Segas:
    def __init__(self, segas: nb.float64):
        self.data = segas


@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Vanna:
    def __init__(self, vanna: nb.float64):
        self.pv = vanna

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class Vannas:
    def __init__(self, vannas: nb.float64[:]):
        self.data = vannas


@nb.experimental.jitclass([
    ("pv", nb.float64)
])
class Volga:
    def __init__(self, volga: nb.float64):
        self.pv = volga

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class Volgas:
    def __init__(self, volgas: nb.float64[:]):
        self.data = volgas
   