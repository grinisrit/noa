import numpy as np
import numba as nb
from typing import Tuple

from .common import *
from .black_scholes import *
from .utils import *


@nb.experimental.jitclass([
    ("sigma", nb.float64),
    ("T", nb.float64)
])
class Straddle:
    def __init__(self, implied_vol: ImpliedVol, time_to_maturity: TimeToMaturity):
        self.sigma = implied_vol.sigma 
        self.T = time_to_maturity.T

@nb.experimental.jitclass([
    ("delta", nb.float64),
    ("sigma", nb.float64),
    ("T", nb.float64)
    
])
class RiskReversal:
    def __init__(self, delta: Delta, vol_quote: VolatilityQuote, time_to_maturity: TimeToMaturity):
        if not (delta.pv <=1 and delta.pv >= 0):
            raise ValueError('Delta expected within [0,1]')
        self.delta = delta.pv
        self.sigma = vol_quote.sigma 
        self.T = time_to_maturity.T
        
@nb.experimental.jitclass([
    ("delta", nb.float64),
    ("sigma", nb.float64),
    ("T", nb.float64)
])
class Butterfly:
    def __init__(self, delta: Delta, vol_quote: VolatilityQuote, time_to_maturity: TimeToMaturity):
        if not (delta.pv <=1 and delta.pv >= 0):
            raise ValueError('Delta expected within [0,1]')
        self.delta = delta.pv
        self.sigma = vol_quote.sigma
        self.T = time_to_maturity.T


@nb.experimental.jitclass([
    ("T", nb.float64),
    ("S", nb.float64),
    ("r", nb.float64),  
    ("r_d", nb.float64),
    ("f", nb.float64),
    ("sigmas", nb.float64[:]),
    ("Ks", nb.float64[:]),
])
class VolSmileChainSpace:
    def __init__(self, forward: Forward, strikes: Strikes, implied_vols: ImpliedVols):
        if not strikes.data.shape == implied_vols.data.shape:
            raise ValueError('Inconsistent data between strikes and implied vols')
        if not is_sorted(strikes.data):
            raise ValueError('Strikes are not in order')

        self.T = forward.T
        self.S = forward.S
        self.r = forward.r
        self.r_d = forward.r_d
        self.f = forward.forward_rate().fv

        self.sigmas = implied_vols.data 
        self.Ks = strikes.data
        
    def strikes(self) -> Strikes:
        return Strikes(self.Ks)
    
    def implied_vols(self) -> ImpliedVols:
        return ImpliedVols(self.sigmas)

    def time_to_maturity(self) -> TimeToMaturity:
        return TimeToMaturity(self.T)

    def vanillas(self) -> SingleMaturityVanillas:
        return SingleMaturityVanillas(
            OptionTypes(self.Ks >= self.f),
            self.strikes(),
            Notionals(np.ones_like(self.Ks)),
            self.time_to_maturity())
    
    def forward(self) -> Forward:
        return Forward(Spot(self.S), ForwardYield(self.r), DiscountYield(self.r_d), TimeToMaturity(self.T))


@nb.experimental.jitclass([
    ("T", nb.float64),
    ("S", nb.float64),
    ("r", nb.float64),
    ("r_d", nb.float64),
    ("f", nb.float64),
    ("ATM", nb.float64),
    ("RR25", nb.float64),
    ("BF25", nb.float64),
    ("RR10", nb.float64),
    ("BF10", nb.float64),
    ("atm_blip", nb.float64),
    ("rr25_blip", nb.float64),
    ("bf25_blip", nb.float64),
    ("rr10_blip", nb.float64),
    ("bf10_blip", nb.float64),
    ("strike_lower", nb.float64),
    ("strike_upper", nb.float64),
    ("delta_tol", nb.float64),
    ("delta_grad_eps", nb.float64)
]) 
class VolSmileDeltaSpace:
    bs_calc: BSCalc
    def __init__(
        self, 
        forward: Forward,
        ATM: Straddle, 
        RR25: RiskReversal, 
        BF25: Butterfly, 
        RR10: RiskReversal, 
        BF10: Butterfly
    ):
        self.T = forward.T
        self.S = forward.S
        self.r = forward.r
        self.r_d = forward.r_d
        self.f = forward.forward_rate().fv

        if not ATM.T == self.T:
            raise ValueError('Inconsistent time_to_maturity for ATM')
        self.ATM = ATM.sigma

        if not RR25.delta == 0.25:
            raise ValueError('Inconsistent delta for 25RR')
        if not RR25.T == self.T:
            raise ValueError('Inconsistent time_to_maturity for 25RR')
        self.RR25 = RR25.sigma 

        if not BF25.delta == 0.25:
            raise ValueError('Inconsistent delta for 25BF')
        if not BF25.T == self.T:
            raise ValueError('Inconsistent time_to_maturity for 25BF')
        self.BF25 = BF25.sigma

        if not RR10.delta == 0.1:
            raise ValueError('Inconsistent delta for 10RR')
        if not RR10.T == self.T:
            raise ValueError('Inconsistent delta for 10RR')
        self.RR10 = RR10.sigma

        if not BF10.delta == 0.1:
            raise ValueError('Inconsistent delta for 10BF')
        if not BF10.T == self.T:
            raise ValueError('Inconsistent time to maturity for 10BF')
        self.BF10 = BF10.sigma

        self.atm_blip = 0.0025
        self.rr25_blip = 0.001
        self.bf25_blip = 0.001
        self.rr10_blip = 0.0016
        self.bf10_blip = 0.00256
        
        self.strike_lower = 0.1
        self.strike_upper = 10.
        self.delta_tol = 10**-12
        self.delta_grad_eps = 1e-4
        
        self.bs_calc = BSCalc()
        self.bs_calc.strike_lower = self.strike_lower
        self.bs_calc.strike_upper = self.strike_upper
        self.bs_calc.delta_tol = self.delta_tol
        self.bs_calc.delta_grad_eps = self.delta_grad_eps

    def _implied_vols(self, RR: nb.float64, BB: nb.float64) -> Tuple[nb.float64]:
        return -RR/2 + (BB + self.ATM), RR/2 + (BB + self.ATM)
    
    def _get_strike(self, sigma: nb.float64, delta: nb.float64) -> nb.float64:
        return self.bs_calc.strike_from_delta(
            Forward(Spot(self.S), ForwardYield(self.r), DiscountYield(self.r_d), TimeToMaturity(self.T)), 
            Delta(delta), 
            ImpliedVol(sigma)
        ).K

    def to_chain_space(self) -> VolSmileChainSpace:
        ivs = np.zeros(5, dtype=np.float64)
        strikes = np.zeros(5, dtype=np.float64)
           
        ivs[2] = self.ATM     
        ivs[1], ivs[3] = self._implied_vols(self.RR25, self.BF25)
        ivs[0], ivs[4] = self._implied_vols(self.RR10, self.BF10)

        strikes[0] = self._get_strike(ivs[0], -0.1)
        strikes[1] = self._get_strike(ivs[1], -0.25)
        strikes[2] = self.f
        strikes[3] = self._get_strike(ivs[3], 0.25)
        strikes[4] = self._get_strike(ivs[4], 0.1)
        
        return VolSmileChainSpace(
            Forward(Spot(self.S), ForwardYield(self.r), DiscountYield(self.r_d), TimeToMaturity(self.T)),
            Strikes(strikes),
            ImpliedVols(ivs)
        ) 
    
    def forward(self) -> Forward:
        return Forward(Spot(self.S), ForwardYield(self.r), DiscountYield(self.r_d), TimeToMaturity(self.T))
    
    def blip_ATM(self):
        self.ATM += self.atm_blip
        return self

    def blip_25RR(self):
        self.RR25 += self.rr25_blip
        return self

    def blip_25BF(self):
        self.BF25 += self.bf25_blip
        return self

    def blip_10RR(self):
        self.RR10 += self.rr10_blip
        return self

    def blip_10BF(self):
        self.BF10 += self.bf10_blip
        return self


@nb.experimental.jitclass([
    ("sigma", nb.float64[:]),
    ("T", nb.float64[:])
])
class Straddles:
    def __init__(self, implied_vols: ImpliedVols, times_to_maturity: TimesToMaturity):
        if not implied_vols.data.shape == times_to_maturity.data.shape:
            raise ValueError('Inconsistent data between implied vols and times to maturity')
        if not is_sorted(times_to_maturity.data):
            raise ValueError('Times to maturity are not in order')
        self.sigma = implied_vols.data 
        self.T = times_to_maturity.data


@nb.experimental.jitclass([
    ("delta", nb.float64),
    ("sigma", nb.float64[:]),
    ("T", nb.float64[:])
    
])
class RiskReversals:
    def __init__(self, delta: Delta, volatility_quotes: VolatilityQuotes, times_to_maturity: TimesToMaturity):
        if not (delta.pv <=1 and delta.pv >= 0):
            raise ValueError('Delta expected within [0,1]')
        if not volatility_quotes.data.shape == times_to_maturity.data.shape:
            raise ValueError('Inconsistent data between quotes and times to maturity')
        if not is_sorted(times_to_maturity.data):
            raise ValueError('Times to maturity are not in order')
        self.delta = delta.pv
        self.sigma = volatility_quotes.data 
        self.T = times_to_maturity.data  


@nb.experimental.jitclass([
    ("delta", nb.float64),
    ("sigma", nb.float64[:]),
    ("T", nb.float64[:])
])
class Butterflies:
    def __init__(self, delta: Delta, volatility_quotes: VolatilityQuotes, times_to_maturity: TimesToMaturity):
        if not (delta.pv <=1 and delta.pv >= 0):
            raise ValueError('Delta expected within [0,1]')
        if not volatility_quotes.data.shape == times_to_maturity.data.shape:
            raise ValueError('Inconsistent data between quotes and times to maturity')
        if not is_sorted(times_to_maturity.data):
            raise ValueError('Times to maturity are not in order')
        self.delta = delta.pv
        self.sigma = volatility_quotes.data 
        self.T = times_to_maturity.data 


@nb.experimental.jitclass([
    ("S", nb.float64),
    ("max_T", nb.float64),
    ("min_T", nb.float64)
])
class VolSurfaceDeltaSpace:
    FWD: ForwardCurve
    ATM: CubicSpline1D
    RR25: CubicSpline1D
    BF25: CubicSpline1D
    RR10: CubicSpline1D
    BF10: CubicSpline1D

    def __init__(
        self, 
        forward_curve: ForwardCurve, 
        straddles: Straddles,
        risk_reversals_25: RiskReversals,
        butterflies_25: Butterflies,
        risk_reversals_10: RiskReversals,
        butterflies_10: Butterflies
    ): 
        self.S = forward_curve.S

        self.FWD = forward_curve

        self.ATM = CubicSpline1D(
            XAxis(straddles.T),
            YAxis(straddles.T*straddles.sigma*straddles.sigma)
        )

        self.RR25 = CubicSpline1D(
            XAxis(risk_reversals_25.T),
            YAxis(straddles.T*risk_reversals_25.sigma)
        )

        self.BF25 = CubicSpline1D(
            XAxis(butterflies_25.T),
            YAxis(straddles.T*butterflies_25.sigma)
        )

        self.RR10 = CubicSpline1D(
            XAxis(risk_reversals_10.T),
            YAxis(straddles.T*risk_reversals_10.sigma)
        )

        self.BF10 = CubicSpline1D(
            XAxis(butterflies_10.T),
            YAxis(straddles.T*butterflies_10.sigma)
        )

        self.max_T = np.min(np.array([ 
            straddles.T[-1],
            risk_reversals_25.T[-1],
            butterflies_25.T[-1],
            risk_reversals_10.T[-1],
            butterflies_10.T[-1]
        ]))

        self.min_T = np.max(np.array([ 
            straddles.T[0],
            risk_reversals_25.T[0],
            butterflies_25.T[0],
            risk_reversals_10.T[0],
            butterflies_10.T[0]
        ]))
        assert self.max_T > self.min_T
        assert self.min_T > 0
        
    def get_vol_smile(self, time_to_maturity: TimeToMaturity) -> VolSmileDeltaSpace:
        T = time_to_maturity.T
        if not (T >= self.min_T and T <= self.max_T):
            print('TimeToMaturity outside available bounds, calendar arbitrage possible!')
    
        return VolSmileDeltaSpace(
            self.FWD.forward(time_to_maturity),
            Straddle(
                ImpliedVol( np.sqrt(self.ATM.apply(T) / T) ),
                time_to_maturity
            ),
            RiskReversal(
                Delta(.25),
                VolatilityQuote(self.RR25.apply(T) / T),
                time_to_maturity
            ),
            Butterfly(
                Delta(.25),
                VolatilityQuote(self.BF25.apply(T) / T),
                time_to_maturity
            ),
            RiskReversal(
                Delta(.1),
                VolatilityQuote(self.RR10.apply(T) / T),
                time_to_maturity
            ),
            Butterfly(
                Delta(.1),
                VolatilityQuote(self.BF10.apply(T) / T),
                time_to_maturity
            )
        )

    def forward_curve(self) -> ForwardCurve:
        return self.forward_curve
    
    def atm_quotes(self, times_to_maturity: TimesToMaturity) -> ImpliedVols:
        Ts = times_to_maturity.data
        atm = np.zeros_like(Ts)
        for i in range(len(Ts)):
            atm[i] = self.ATM.apply(Ts[i]) / Ts[i]
        return ImpliedVols(atm)
    
    def rr25_quotes(self, times_to_maturity: TimesToMaturity) -> VolatilityQuotes:
        Ts = times_to_maturity.data
        rr = np.zeros_like(Ts)
        for i in range(len(Ts)):
            rr[i] = self.RR25.apply(Ts[i]) / Ts[i]
        return VolatilityQuotes(rr)
    
    def bf25_quotes(self, times_to_maturity: TimesToMaturity) -> VolatilityQuotes:
        Ts = times_to_maturity.data
        bf = np.zeros_like(Ts)
        for i in range(len(Ts)):
            bf[i] = self.BF25.apply(Ts[i]) / Ts[i]
        return VolatilityQuotes(bf)
    
    def rr10_quotes(self, times_to_maturity: TimesToMaturity) -> VolatilityQuotes:
        Ts = times_to_maturity.data
        rr = np.zeros_like(Ts)
        for i in range(len(Ts)):
            rr[i] = self.RR10.apply(Ts[i]) / Ts[i]
        return VolatilityQuotes(rr)

    def bf10_quotes(self, times_to_maturity: TimesToMaturity) -> VolatilityQuotes:
        Ts = times_to_maturity.data
        bf = np.zeros_like(Ts)
        for i in range(len(Ts)):
            bf[i] = self.BF10.apply(Ts[i]) / Ts[i]
        return VolatilityQuotes(bf)


@nb.experimental.jitclass([
    ("S", nb.float64),
    ("Ts", nb.float64[:]),
    ("Ks", nb.float64[:]),
    ("pvs", nb.float64[:]),
    ("sigmas", nb.float64[:]),
    ("compute_implied_vol", nb.boolean)
])
class VolSurfaceChainSpace:
    bs_calc: BSCalc
    FWD: ForwardCurve
  
    def __init__(
        self,
        forward_curve: ForwardCurve, 
        times_to_maturity: TimesToMaturity,
        strikes: Strikes,
        option_types: OptionTypes,
        premiums: Premiums,
        compute_implied_vol: bool = True
    ):
        if not times_to_maturity.data.shape == strikes.data.shape == premiums.data.shape == option_types.data.shape:
            raise ValueError('Inconsistent data shape between times to maturity, strikes, premiums and option types')
        if not np.all(premiums.data > 0):
            raise ValueError('Invalid premiums data')
        if not is_sorted(times_to_maturity.data):
            raise ValueError('Invalid TTM data')
        
        self.bs_calc = BSCalc()
        
        self.S = forward_curve.S
        self.FWD = forward_curve

        self.compute_implied_vol = compute_implied_vol
        self._process(times_to_maturity.data.flatten(), strikes.data.flatten(), option_types.data.flatten(), premiums.data.flatten())

    def _process(self, Ts: nb.float64[:], Ks: nb.float64[:], Cs: nb.float64[:], PVs: nb.float64[:]):
        lTs = []
        lKs = []
        lPVs = []
        lIVs = []
        n = len(Ts)
        
        lT = Ts[0]
        assert lT > 0

        F = self.FWD.forward(TimeToMaturity(lT))
        f = F.forward_rate().fv

        for i in range(n):
            T = Ts[i]
            assert T >= lT
            if T > lT:
                F = self.FWD.forward(TimeToMaturity(T))
                f = F.forward_rate().fv
                lT = T

            K = Ks[i]
            is_call = Cs[i]

            if (f > K and is_call) or (K >= f and not is_call):
                continue
            else:
                pv = PVs[i]
                if self.compute_implied_vol:
                    iv = self.bs_calc.implied_vol(F, Strike(K), Premium(pv)).sigma
                else:
                    iv = -1

                lTs.append(T)
                lKs.append(K)
                lPVs.append(pv)
                lIVs.append(iv)
        
        self.Ts = np.array(lTs)
        self.Ks = np.array(lKs)
        self.pvs = np.array(lPVs)
        self.sigmas = np.array(lIVs)

    def times_to_maturities(self) -> TimesToMaturity:
        return TimesToMaturity(np.unique(self.Ts))

    def get_vol_smile(self, time_to_maturity: TimeToMaturity) -> VolSmileChainSpace:
        F = self.FWD.forward(time_to_maturity)
        T = time_to_maturity.T
        n = len(self.Ts)

        lKs = []
        lsigmas = []

        for i in range(n):
            cT = self.Ts[i]
            if cT < T:
                continue
            elif cT == T:
                lKs.append(self.Ks[i])
                lsigmas.append(self.sigmas[i])
            else:
                break

        Ks = np.array(lKs)
        sigmas = np.array(lsigmas)
        idx = np.argsort(Ks)

        return VolSmileChainSpace(F , Strikes(Ks[idx]), ImpliedVols(sigmas[idx]))
    
    def forward_curve(self) -> ForwardCurve:
        return self.FWD
    
    def strikes_maturities_grid(self) -> Tuple[StrikesMaturitiesGrid, OptionTypes]:
        forwards = self.FWD.forward_rates(TimesToMaturity(self.Ts))
        return StrikesMaturitiesGrid(
            self.FWD.spot(),
            TimesToMaturity(self.Ts),
            Strikes(self.Ks)
        ), OptionTypes(forwards.data <= self.Ks)
        

 