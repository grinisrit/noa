import numpy as np
import numba as nb


from .common import *
from .black_scholes import *


@nb.experimental.jitclass([
    ("iv", nb.float64)
])
class ATM:
    def __init__(self, iv: ImpliedVol):
        self.iv = iv 
        
    def implied_vols(self, RR: RiskReversal, BB: Butterfly) -> ImpliedVols:
        assert RR.delta.pv == BB.delta.pv
        res = np.zeros(3, dtype=np.float64)
        res[0] = -RR.iv.sigma/2 + (BB.sigma + self.iv.sigma)
        res[1] = self.iv.sigma
        res[2] = RR.iv.sigma/2 + (BB.sigma + self.iv.sigma)
        return ImpliedVols(res)
                      
        
@nb.experimental.jitclass([
    ("delta", Delta),
    ("iv", ImpliedVol)
])
class RiskReversal:
    def __init__(self, delta: Delta, implied_vol: ImpliedVol):
        self.delta = delta
        self.iv = iv 
        
@nb.experimental.jitclass([
    ("delta", Delta),
    ("iv", ImpliedVol)
])
class Butterfly:
    def __init__(self, delta: Delta, implied_vol: ImpliedVol):
        self.delta = delta
        self.iv = iv

@nb.experimental.jitclass([
    ("forward", Forward),
    ("ATM", ATM),
    ("RR25", RiskReversal),
    ("BB25", Butterfly),
    ("RR10", RiskReversal),
    ("BB10", Butterfly),
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
    
    @staticmethod
    def from_delta_space(vol: VolSmileDeltaSpace):
        ivs = np.zeros(5, dtype=np.float64)
        strikes = np.zeros(5, dtype=np.float64)
        
        ivs[2] = vol.ATM.sigma
        strikes[2] = vol.forward.fv()
        
        iv25 = vol.ATM.implied_vols(vol.RR25, vol.BB25)
        iv10 = vol.ATM.implied_vols(vol.RR10, vol.BB10)
        
        return VolSmileChain(
            vol.forward,
            Strikes(strikes),
            ImpliedVols(implied_vols)
        )
        