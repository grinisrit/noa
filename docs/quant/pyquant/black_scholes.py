import numpy as np
import numba as nb

from .common import *


@nb.experimental.jitclass([
    ("S", nb.float64),
    ("r", nb.float64),
    ("T", nb.float64),
    ("K", nb.float64),
    ("is_call", nb.boolean),
    ("tol", nb.float64),
    ("sigma_lower", nb.float64),
    ("sigma_upper", nb.float64),
    
])
class BlackScholes:
    def __init__(self, forward: Forward, strike: Strike, option_type: OptionType):
        self.S = forward.S
        self.r = forward.r
        self.T = forward.T
        self.K = strike.K
        self.is_call = option_type.is_call
        self.tol = 10**-6
        self.sigma_lower = 10**-4
        self.sigma_upper = 2.
        
    @staticmethod
    def from_delta_space(
        forward: Forward, 
        implied_vol: ImpliedVol,
        delta: Delta, 
        option_type: OptionType,
        lower_strike: Strike, 
        upper_strike: Strike,
        tol = 10**-12
    ):
        K_l = lower_strike
        K_r = upper_strike
        
        def g(K, delta):
            return BlackScholes(forward, Strike(K), option_type).delta(implied_vol).pv - delta.pv

        def g_prime(K):
            return BlackScholes(forward, Strike(K), option_type)._dDelta_dK(implied_vol.sigma)
        
        assert g(K_l, delta) * g(K_r, delta) <= 0
        
        res = BlackScholes(forward, 
                           Strike((K_l+K_r) / 2), 
                           option_type)
        
        epsilon = g(res.K, delta)
        grad = g_prime(res.K)
        i = 0
        while abs(epsilon) > tol and i < 10: 
            if abs(grad) > 1e-6:
                res.K -= epsilon / grad
                if res.K > K_r or res.K < K_l:
                    res.K = (K_l + K_r) / 2
                    if g(K_l, delta)*epsilon > 0:
                        K_l = res.K
                    else:
                        K_r = res.K
                    res.K = (K_l + K_r) / 2
            else:
                if g(K_l, delta)*epsilon > 0:
                    K_l = res.K
                else:
                    K_r = res.K
                res.K = (K_l + K_r) / 2
            
            epsilon = g(res.K, delta)
            grad = g_prime(res.K)
            i += 1
        return res
        
    def implied_vol(self, premium: Premium) -> ImpliedVol:
          
        def g(pv, sigma):
            return pv - self.premium(ImpliedVol(sigma)).pv

        def g_prime(sigma):
            return -self.vega(ImpliedVol(sigma)).pv 
        
        sigma_l = self.sigma_lower
        sigma_r = self.sigma_upper
        pv = premium.pv
        
        assert g(pv,sigma_l) * g(pv,sigma_r) <= 0
        
        sigma = (sigma_l + sigma_r) / 2
        epsilon = g(pv, sigma)
        grad = g_prime(sigma)
        while abs(epsilon) > self.tol:   
            if abs(grad) > 1e-6:
                sigma -= epsilon / grad
                if sigma > sigma_r or sigma < sigma_l:
                    sigma = (sigma_l + sigma_r) / 2
                    if g(pv, sigma_l)*epsilon > 0:
                        sigma_l = sigma
                    else:
                        sigma_r = sigma
                    sigma = (sigma_l + sigma_r) / 2
            else:
                if g(pv, sigma_l)*epsilon > 0:
                    sigma_l = sigma
                else:
                    sigma_r = sigma
                sigma = (sigma_l + sigma_r) / 2
            
            epsilon = g(pv, sigma)
            grad = g_prime(sigma) 
        return ImpliedVol(sigma)
       
    def premium(self, implied_vol: ImpliedVol) -> Premium:
        sigma = implied_vol.sigma
        pm = 1 if self.is_call else -1
        d1 = self._d1(sigma)
        d2 = self._d2(d1, sigma)
        return Premium(
            pm * self.S * normal_cdf(pm * d1) - pm * self.K * \
            np.exp(-self.r * self.T) * normal_cdf(pm * d2)
        )
    
    def delta(self, implied_vol: ImpliedVol) -> Delta:
        d1 = self._d1(implied_vol.sigma)
        return Delta(
            normal_cdf(d1) if self.is_call else normal_cdf(d1) - 1.0
        )
    
    def gamma(self, implied_vol: ImpliedVol) -> Gamma:
        sigma = implied_vol.sigma
        d1 = self._d1(sigma) 
        return Gamma(
            normal_pdf(d1) / (self.S * sigma * np.sqrt(self.T))
        )
    
    def vega(self, implied_vol: ImpliedVol) -> Vega:
        return Vega(
            self.S * np.sqrt(self.T) * normal_pdf(self._d1(implied_vol.sigma))
        )
    
    def vanna(self, implied_vol: ImpliedVol) -> Vanna:
        sigma = implied_vol.sigma
        d2 = self._d2(self._d1(sigma), sigma)
        return Vanna(
            self.vega(implied_vol).pv * d2 / (sigma * self.S)
        )
    
    def volga(self, implied_vol: ImpliedVol) -> Volga:
        sigma = implied_vol.sigma
        d1 = self._d1(sigma)
        d2 = self._d2(d1,sigma)
        return Volga(
            self.vega(implied_vol).pv * d1 * d2 / sigma
        )
    
    def _d1(self, sigma: nb.float64) -> nb.float64:
        d1 = (np.log(self.S / self.K) + (self.r + sigma**2 / 2) * self.T) / (sigma * np.sqrt(self.T))
        return d1
    
    def _d2(self, d1: nb.float64, sigma: nb.float64) -> nb.float64:
        return d1 - sigma * np.sqrt(self.T)
    
    def _dDelta_dK(self, sigma: nb.float64) -> nb.float64:
        d1 = self._d1(sigma)
        return - normal_pdf(d1) / (self.K * np.sqrt(self.T) * sigma)
    