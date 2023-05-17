import numpy as np
import numba as nb
from scipy.interpolate import CubicSpline


from .common import *
from .black_scholes import *


@nb.experimental.jitclass([
    ("sigma", nb.float64),
    ("T", nb.float64)
])
class Straddle:
    def __init__(self, implied_vol: ImpliedVol, tenor: Tenor):
        self.sigma = implied_vol.sigma 
        self.T = tenor.T


@nb.experimental.jitclass([
    ("delta", nb.float64),
    ("sigma", nb.float64),
    ("T", nb.float64)
    
])
class RiskReversal:
    def __init__(self, delta: Delta, implied_vol: ImpliedVol, tenor: Tenor):
        assert delta.pv <=1
        assert delta.pv >= 0
        self.delta = delta.pv
        self.sigma = implied_vol.sigma 
        self.T = tenor.T
        
@nb.experimental.jitclass([
    ("delta", nb.float64),
    ("sigma", nb.float64),
    ("T", nb.float64)
])
class Butterfly:
    def __init__(self, delta: Delta, implied_vol: ImpliedVol, tenor: Tenor):
        assert delta.pv <=1
        assert delta.pv >= 0
        self.delta = delta.pv
        self.sigma = implied_vol.sigma
        self.T = tenor.T


@nb.experimental.jitclass([
    ("f", nb.float64),
    ("T", nb.float64),
    ("sigma", nb.float64[:]),
    ("K", nb.float64[:]),
])
class VolSmileChain:
    def __init__(self, forward: Forward, strikes: Strikes, implied_vols: ImpliedVols):
        assert strikes.data.shape == implied_vols.data.shape
        self.f = forward.forward_rate().fv
        self.T = forward.T
        self.sigma = implied_vols.data 
        self.K = strikes.data                    
    

@nb.experimental.jitclass([
    ("forward", Forward),
    ("ATM", Straddle),
    ("RR25", RiskReversal),
    ("BB25", Butterfly),
    ("RR10", RiskReversal),
    ("BB10", Butterfly),
    ("tol", nb.float64),
    ("upper_strike", nb.float64),
    ("lower_strike", nb.float64),
    
]) 
class VolSmileDeltaSpace:
    def __init__(
        self, 
        forward: Forward,
        ATM: Straddle, 
        RR25: RiskReversal, 
        BB25: Butterfly, 
        RR10: RiskReversal, 
        BB10: Butterfly
    ):
        self.forward = forward

        assert ATM.T == forward.T
        self.ATM = ATM

        assert RR25.delta == 0.25
        assert RR25.T == forward.T
        self.RR25 = RR25 

        assert BB25.delta == 0.25
        assert BB25.T == forward.T
        self.BB25 = BB25

        assert RR10.delta == 0.1
        assert RR10.T == forward.T
        self.RR10 = RR10

        assert BB10.delta == 0.1
        assert BB10.T == forward.T
        self.BB10 = BB10

        self.upper_strike = 100*self.forward.fv()
        self.lower_strike = 0.01*self.forward.fv()

    def _implied_vols(self, RR: RiskReversal, BB: Butterfly) -> tuple[nb.float64]:
        assert RR.delta.pv == BB.delta.pv
        return -RR.iv.sigma/2 + (BB.sigma + self.ATM.sigma), RR.sigma/2 + (BB.sigma + self.ATM.sigma)
    
    def _get_strike(self, sigma: nb.float64, delta: nb.float64) -> nb.float64:
        return BlackScholesCalc(OptionType(delta >= 0.)).strike_from_delta(
            self.forward, Delta(delta), ImpliedVol(sigma)).K

    def to_chain_space(self):
        ivs = np.zeros(5, dtype=np.float64)
        strikes = np.zeros(5, dtype=np.float64)
        
        ivs[2] = self.ATM.sigma     
        ivs[1], ivs[3] = self._implied_vols(self.RR25, self.BB25)
        ivs[0], ivs[4] = self._implied_vols(self.RR10, self.BB10)

        strikes[0] = self._get_strike(ivs[0], -self.BB10.delta)
        strikes[1] = self._get_strike(ivs[1], -self.BB25.delta)
        strikes[2] = self.forward.fv()
        strikes[0] = self._get_strike(ivs[0], self.RR25.delta)
        strikes[0] = self._get_strike(ivs[0], self.BB25.delta)
        
        return VolSmileChain(
            self.forward,
            Strikes(strikes),
            ImpliedVols(ivs)
        )   


@nb.experimental.jitclass([
    ("sigma", nb.float64[:]),
    ("T", nb.float64[:])
])
class Straddles:
    def __init__(self, implied_vols: ImpliedVols, tenors: Tenors):
        assert implied_vols.data.shape == tenors.data.shape
        assert is_sorted(tenors.data)
        self.sigma = implied_vols.sigma 
        self.T = tenors.T


@nb.experimental.jitclass([
    ("delta", nb.float64),
    ("sigma", nb.float64[:]),
    ("T", nb.float64[:])
    
])
class RiskReversals:
    def __init__(self, delta: Delta, implied_vols: ImpliedVols, tenors: Tenors):
        assert delta.pv <=1
        assert delta.pv >= 0
        assert implied_vols.data.shape == tenors.data.shape
        assert is_sorted(tenors.data)
        self.delta = delta.pv
        self.sigma = implied_vols.data 
        self.T = tenors.data  


@nb.experimental.jitclass([
    ("delta", nb.float64),
    ("sigma", nb.float64[:]),
    ("T", nb.float64[:])
])
class Butterflies:
    def __init__(self, delta: Delta, implied_vols: ImpliedVols, tenors: Tenors):
        assert delta.pv <=1
        assert delta.pv >= 0
        assert implied_vols.data.shape == tenors.data.shape
        assert is_sorted(tenors.data)
        self.delta = delta.pv
        self.sigma = implied_vols.data 
        self.T = tenors.data 


class VolSurface:

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

        self.f = CubicSpline(
            np.append(np.array([0.]), forward_curve.T),
            np.append(np.array([0.]), forward_curve.forward_rates())
        )

        self.ATM = CubicSpline(
            np.append(np.array([0.]), straddles.T),
            np.append(np.array([0.]), straddles.sigma)
        )

        self.RR25 = CubicSpline(
            np.append(np.array([0.]), risk_reversals_25.T),
            np.append(np.array([0.]), risk_reversals_25.sigma)
        )

        self.BB25 = CubicSpline(
            np.append(np.array([0.]), butterflies_25.T),
            np.append(np.array([0.]), butterflies_25.sigma)
        )

        self.RR10 = CubicSpline(
            np.append(np.array([0.]), risk_reversals_10.T),
            np.append(np.array([0.]), risk_reversals_10.sigma)
        )

        self.BB10 = CubicSpline(
            np.append(np.array([0.]), butterflies_10.T),
            np.append(np.array([0.]), butterflies_10.sigma)
        )

        self.max_T = np.min(np.array([
            forward_curve.T[-1], 
            straddles.T[-1],
            risk_reversals_25.T[-1],
            butterflies_25.T[-1],
            risk_reversals_10.T[-1],
            butterflies_10.T[-1]
        ]))
        
    def get_vol_smile(self, tenor: Tenor) -> VolSmileDeltaSpace:
        T = tenor.T
        assert T > 0 and T <= self.max_T
    
        return VolSmileDeltaSpace(
            Forward.from_forward_rate(
                Spot(self.S),
                ForwardRate(self.f(T)),
                tenor
            ),
            Straddle(
                ImpliedVol(self.ATM(T)),
                tenor
            ),
            RiskReversal(
                Delta(.25),
                ImpliedVol(self.RR25(T)),
                tenor
            ),
            Butterfly(
                Delta(.25),
                ImpliedVol(self.BB25(T)),
                tenor
            ),
            RiskReversal(
                Delta(.1),
                ImpliedVol(self.RR10(T)),
                tenor
            ),
            Butterfly(
                Delta(.1),
                ImpliedVol(self.BB10(T)),
                tenor
            )
        )