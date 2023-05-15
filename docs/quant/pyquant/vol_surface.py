import numpy as np
import numba as nb


from .common import *
from .black_scholes import *


@nb.experimental.jitclass([
    ("delta", nb.float64),
    ("sigma", nb.float64)
])
class RiskReversal:
    def __init__(self, delta: Delta, implied_vol: ImpliedVol):
        assert delta.pv <=1
        assert delta.pv >= 0
        self.delta = delta.pv
        self.sigma = implied_vol.sigma 
        
@nb.experimental.jitclass([
    ("delta", nb.float64),
    ("sigma", nb.float64)
])
class Butterfly:
    def __init__(self, delta: Delta, implied_vol: ImpliedVol):
        assert delta.pv <=1
        assert delta.pv >= 0
        self.delta = delta.pv
        self.sigma = implied_vol.sigma

@nb.experimental.jitclass([
    ("iv", nb.float64)
])
class ATM:
    def __init__(self, iv: ImpliedVol):
        self.sigma = iv.sigma 


@nb.experimental.jitclass([
    ("forward", Forward),
    ("strikes", Strikes),
    ("implied_vols", ImpliedVols),
])
class VolSmileChain:
    def __init__(self, forward: Forward, strikes: Strikes, implied_vols: ImpliedVols):
        assert strikes.data.shape == implied_vols.data.shape
        self.forward = forward  
        self.strikes = strikes
        self.implied_vols = implied_vols                     
    

@nb.experimental.jitclass([
    ("forward", Forward),
    ("ATM", ATM),
    ("RR25", RiskReversal),
    ("BB25", Butterfly),
    ("RR10", RiskReversal),
    ("BB10", Butterfly),
    ("tol", nb.float64),
    ("upper_strike", nb.float64),
    ("sigma_upper", nb.float64),
    
]) 
class VolSmileDeltaSpace:
    def __init__(
        self, 
        forward: Forward,
        ATM: ATM, 
        RR25: RiskReversal, 
        BB25: Butterfly, 
        RR10: RiskReversal, 
        BB10: Butterfly
    ):
        self.forward = forward
        self.ATM = ATM
        assert RR25.delta.pv == 0.25
        self.RR25 = RR25 
        assert BB25.delta.pv == 0.25
        self.BB25 = BB25 
        assert RR10.delta.pv == 0.1
        self.RR10 = RR10
        assert BB10.delta.pv == 0.1
        self.BB10 = BB10

        self.upper_strike = 100*self.forward.fv()
        self.upper_strike = 0.01*self.forward.fv()

    def _implied_vols(self, RR: RiskReversal, BB: Butterfly) -> tuple[nb.float64]:
        assert RR.delta.pv == BB.delta.pv
        return -RR.iv.sigma/2 + (BB.sigma + self.ATM.sigma), RR.sigma/2 + (BB.sigma + self.ATM.sigma)
    
    def _get_strike(self, sigma: nb.float64, delta: nb.float64) -> nb.float64:
        f = self.forward.fv()
        K_r = f if delta < 0 else 100*f
        K_l = f if delta > 0 else 0.01*f
        return BlackScholes.from_delta_space(
            self.forward, ImpliedVol(sigma), Delta(delta), Strike(K_l), Strike(K_r)
        ) 

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
    ("forward", Forward),
    ("strikes", Strikes),
    ("implied_vols", ImpliedVols),
])
class VolSmileChain:
    def __init__(self, forward: Forward, strikes: Strikes, implied_vols: ImpliedVols):
        assert strikes.data.shape == implied_vols.data.shape
        self.forward = forward  
        self.strikes = strikes
        self.implied_vols = implied_vols   
        

class VolSurface:

    def __init__(self, forwards: np.array, ATMs):
        