import numpy as np
import numba as nb

from .utils import *

@nb.experimental.jitclass([
    ("v", nb.float64)
])
class CalibrationError:
    def __init__(self, value: nb.float64):
        self.v = value


@nb.experimental.jitclass([
    ("w", nb.float64[:])
])
class CalibrationWeights:
    def __init__(self, w: nb.float64):
        if not np.all(w>=0):
            raise ValueError('Weights must be non-negative')
        if not w.sum() > 0:
            raise ValueError('At least one weight must be non-trivial')
        self.w = w


@nb.experimental.jitclass([
    ("v", nb.boolean)
])
class StickyStrike:
    def __init__(self, v: nb.boolean = False):
        self.v = v  


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
    def __init__(self, sigmas: nb.float64[:]):
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
    ("N", nb.float64)
])
class Notional:
    def __init__(self, N: nb.float64):
        self.N = N

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class Notionals:
    def __init__(self, notionals: nb.float64[:]):
        self.data = notionals
        

@nb.experimental.jitclass([
    ("S", nb.float64)
])
class Spot:
    def __init__(self, spot: nb.float64):
        self.S = spot


@nb.experimental.jitclass([
    ("r_d", nb.float64)
])
class DiscountYield:
    def __init__(self, rate: nb.float64):
        self.r_d = rate


@nb.experimental.jitclass([
    ("D", nb.float64)
])
class DiscountFactor:
    def __init__(self, D: nb.float64):
        self.D = D


@nb.experimental.jitclass([
    ("r", nb.float64)
])
class ForwardYield:
    def __init__(self, rate: nb.float64):
        self.r = rate


@nb.experimental.jitclass([
    ("D", nb.float64)
])
class ForwardDiscount:
    def __init__(self, d: nb.float64):
        self.D = d


@nb.experimental.jitclass([
    ("D", nb.float64)
])
class DiscountRatio:
    def __init__(self, d: nb.float64):
        self.D = d

@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class DiscountYields:
    def __init__(self, rates: nb.float64[:]):
        self.data = rates


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
        if not np.all(T >= 0) and is_sorted(T):
            raise ValueError('Not all times to maturity are positive and sorted')
        self.data = T      


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
    ("data", nb.float64[:])
])
class DiscountFactors:
    def __init__(self, data: nb.float64[:]):
        self.data = data


@nb.experimental.jitclass([
    ("r_d", nb.float64),
    ("T", nb.float64)
])
class Discount:
    def __init__(self, rate: DiscountYield, time_to_maturity: TimesToMaturity):
        self.r_d = rate.r_d
        self.T = time_to_maturity.T

    def discount_yield(self) -> DiscountYield:
        return DiscountYield(self.r_d)
    
    def time_to_maturity(self) -> TimeToMaturity:
        return TimeToMaturity(self.T)       
    
    def discount_factor(self) -> DiscountFactor:
        return DiscountFactor(np.exp(-self.r_d * self.T))


@nb.experimental.jitclass([
    ("S", nb.float64),
    ("r", nb.float64),
    ("r_d", nb.float64),
    ("T", nb.float64)
])
class Forward:
    def __init__(self, spot: Spot, forward_yield: ForwardYield, discount_yield: DiscountYield, time_to_maturity: TimeToMaturity):
        self.S = spot.S
        self.r = forward_yield.r
        self.r_d = discount_yield.r_d
        self.T = time_to_maturity.T
        
    def spot(self) -> Spot:
        return Spot(self.S)
    
    def forward_yield(self) -> ForwardYield:
        return ForwardYield(self.r)
    
    def discount_yield(self) -> DiscountYield:
        return DiscountYield(self.r_d)
    
    def discount_factor(self) -> DiscountFactor:
        return DiscountFactor(np.exp(-self.r_d * self.T))
    
    def time_to_maturity(self) -> TimeToMaturity:
        return TimeToMaturity(self.T)
        
    def forward_rate(self) -> ForwardRate:
        return ForwardRate(self.S * np.exp(self.r * self.T))
    
    def forward_discount(self) -> ForwardDiscount:
        return ForwardDiscount(np.exp(-self.r * self.T))
    
    def discount_ratio(self) -> DiscountRatio:
        return DiscountRatio(self.discount_factor().D / self.forward_discount().D)


@nb.njit
def forward_from_forward_rate(
    spot: Spot,
    forward_rate: ForwardRate,
    time_to_maturity: TimeToMaturity
) -> Forward:
    # assumes discount and forward yield are the same
    r = - np.log(spot.S / forward_rate.fv)/ time_to_maturity.T
    return Forward(
        spot, ForwardYield(r), DiscountYield(r), time_to_maturity
        )



@nb.experimental.jitclass()
class ForwardYieldCurve:
    _spline: PchipSpline1D

    def __init__(self, forward_yields: ForwardYields, times_to_maturity: TimesToMaturity):
        if not forward_yields.data.shape == times_to_maturity.data.shape:
            raise ValueError('Inconsistent data between yields and times to maturity')
        if not is_sorted(times_to_maturity.data) and np.all(times_to_maturity.data > 0):
            raise ValueError('Times to maturity are invalid')
    
        self._spline = PchipSpline1D(
           XAxis(np.append(np.array([0.]), times_to_maturity.data)),
           YAxis(np.append(np.array([0.]), times_to_maturity.data*forward_yields.data)) 
        )
    
    def forward_yield(self, time_to_maturity: TimeToMaturity) -> ForwardYield:
        assert time_to_maturity.T > 0.
        return ForwardYield(self._spline.apply(time_to_maturity.T) / time_to_maturity.T)
    
    def forward_yields(self, times_to_maturity: TimeToMaturity) -> ForwardYields:
        Ts = times_to_maturity.data
        res = np.zeros_like(Ts)
        for i in range(len(Ts)):
            assert Ts[i] > 0.
            res[i] = self._spline.apply(Ts[i]) / Ts[i]
        return ForwardYields(res)
    

@nb.experimental.jitclass()
class DiscountCurve:
    _spline: PchipSpline1D

    def __init__(self, discount_yields: DiscountYields, times_to_maturity: TimesToMaturity):
        if not discount_yields.data.shape == times_to_maturity.data.shape:
            raise ValueError('Inconsistent data between discount yields and times to maturity')
        if not is_sorted(times_to_maturity.data) and np.all(times_to_maturity.data > 0):
            raise ValueError('Times to maturity are invalid')
    
        self._spline = PchipSpline1D(
           XAxis(np.append(np.array([0.]), times_to_maturity.data)),
           YAxis(np.append(np.array([0.]), times_to_maturity.data*discount_yields.data)) 
        )
    
    def discount_yield(self, time_to_maturity: TimeToMaturity) -> DiscountYield:
        assert time_to_maturity.T > 0.
        return DiscountYield(self._spline.apply(time_to_maturity.T) / time_to_maturity.T)
    
    def discount_factor(self, time_to_maturity: TimeToMaturity) -> DiscountFactor:
        return DiscountFactor(np.exp(-self.discount_yield(time_to_maturity).r_d * time_to_maturity.T))
    
    def discount_yields(self, times_to_maturity: TimeToMaturity) -> DiscountYields:
        Ts = times_to_maturity.data
        res = np.zeros_like(Ts)
        for i in range(len(Ts)):
            assert Ts[i] > 0.
            res[i] = self._spline.apply(Ts[i]) / Ts[i]
        return DiscountYields(res)
    
    def discount_factors(self, times_to_maturity: TimeToMaturity) -> DiscountFactors:
        return DiscountFactors(
            np.exp(-self.discount_yields(times_to_maturity).data * times_to_maturity.data)
        )


@nb.experimental.jitclass([
    ("S", nb.float64)
])
class ForwardCurve:
    _curve: ForwardYieldCurve
    _curve_d: DiscountCurve

    def __init__(self, spot: Spot, forward_yield_curve: ForwardYieldCurve, discount_curve: DiscountCurve):
        self.S = spot.S
        self._curve = forward_yield_curve
        self._curve_d = discount_curve

    def spot(self) -> Spot:
        return Spot(self.S)
  
    def forward(self, time_to_maturity: TimeToMaturity) -> Forward:
        return Forward(
            Spot(self.S), 
            self._curve.forward_yield(time_to_maturity), 
            self._curve_d.discount_yield(time_to_maturity),
            time_to_maturity
        )
    
    def forward_rates(self, times_to_maturity: TimesToMaturity) -> ForwardRates:
        return ForwardRates(
            self.S * np.exp(times_to_maturity.data * self._curve.forward_yields(times_to_maturity).data)
        )
    
    def forward_yields(self, times_to_maturity: TimesToMaturity) -> ForwardYields:
        return self._curve.forward_yields(times_to_maturity)
    
    def discount_yields(self, times_to_maturity: TimesToMaturity) -> DiscountYields:
        return self._curve_d.discount_yields(times_to_maturity)
    
    def discount_factors(self, times_to_maturity: TimesToMaturity) -> DiscountFactors:
        return self._curve_d.discount_factors(times_to_maturity)


@nb.njit
def forward_curve_from_forward_rates(
    spot: Spot,
    forward_rates: ForwardRates,
    times_to_maturity: TimesToMaturity
) -> ForwardCurve:
    rs = - np.log(spot.S / forward_rates.data) / times_to_maturity.data
    return ForwardCurve(
        spot,
        ForwardYieldCurve( 
            ForwardYields(rs),
            times_to_maturity), 
        DiscountCurve(
            DiscountYields(rs),
            times_to_maturity)
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


@nb.njit()
def call_option(): 
    return OptionType(True)


@nb.njit()
def put_option(): 
    return OptionType(False)
        
        
@nb.experimental.jitclass([
    ("data", nb.boolean[:])
])
class OptionTypes:
    def __init__(self, is_call: nb.boolean[:]):
        self.data = is_call
      
    
@nb.experimental.jitclass([
    ("is_call", nb.boolean),
    ("K", nb.float64),
    ("N", nb.float64),
    ("T", nb.float64)
])
class Vanilla:
    def __init__(self, option_type: OptionType, strike: Strike, notional: Notional, time_to_maturity: TimeToMaturity):
        self.is_call = option_type.is_call
        self.K = strike.K
        self.N = notional.N
        self.T = time_to_maturity.T
        
    def option_type(self) -> OptionType:
        return OptionType(self.is_call)
        
    def strike(self) -> Strike:
        return Strike(self.K)
        
    def time_to_maturity(self) -> TimeToMaturity:
        return TimeToMaturity(self.T)
        
        
@nb.experimental.jitclass([
    ("is_call", nb.boolean[:]),
    ("Ks", nb.float64[:]),
    ("Ns", nb.float64[:]),
    ("T", nb.float64)
])
class SingleMaturityVanillas:
    def __init__(self, option_types: OptionTypes, strikes: Strikes, notionals: Notionals, time_to_maturity: TimeToMaturity):
        if not option_types.data.shape == strikes.data.shape:
            raise ValueError('Inconsistent data between strikes and option types')
        if not notionals.data.shape == strikes.data.shape:
            raise ValueError('Inconsistent data between strikes and notionals')
        self.is_call = option_types.data
        self.Ks = strikes.data
        self.Ns = notionals.data
        self.T = time_to_maturity.T
        
    def strikes(self) -> Strikes:
        return Strikes(self.Ks)

    def time_to_maturity(self):
        return TimeToMaturity(self.T)
        

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

@nb.experimental.jitclass([
    ("S", nb.float64),
    ("Ts", nb.float64[:]),
    ("Ks", nb.float64[:])
])
class StrikesMaturitiesGrid:
    def __init__(
        self,
        spot: Spot,
        times_to_maturity: TimesToMaturity,
        strikes: Strikes):

        assert times_to_maturity.data.shape == strikes.data.shape
        self.S = spot.S
        self.Ts = times_to_maturity.data
        self.Ks = strikes.data

    def spot(self) -> Spot:
        return Spot(self.S)

   