import numpy as np
import numba as nb

from .utils import *
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
    ("grad_eps", nb.float64),
    ("delta_tol", nb.float64),
    ("strike_lower", nb.float64),
    ("strike_upper", nb.float64),
    ("delta_grad_eps", nb.float64)
])
class BSCalc:
    def __init__(self):
        self.tol = 10**-6
        self.sigma_lower = 10**-3
        self.sigma_upper = 3
        self.grad_eps = 1e-6
        self.delta_tol = 10**-12
        self.strike_lower = 0.1
        self.strike_upper = 10.
        self.delta_grad_eps = 1e-4
        
    def strike_from_delta(self, forward: Forward, delta: Delta, implied_vol: ImpliedVol) -> Strike:
        K_l = self.strike_lower*forward.S
        K_r = self.strike_upper*forward.S
        option_type = OptionType(delta.pv >= 0.)
        
        def g(K):
            return self._delta(forward, Strike(K), option_type, implied_vol) - delta.pv

        def g_prime(K):
            return self._dDelta_dK(forward, Strike(K), implied_vol)
        
        if g(K_l) * g(K_r) > 0:
            raise ValueError('No solution within strikes interval')
        
        K = (K_l+K_r) / 2
        epsilon = g(K)
        grad = g_prime(K)
        while abs(epsilon) > self.delta_tol: 
            if abs(grad) > self.delta_grad_eps:
                K -= epsilon / grad
                if K > K_r or K < K_l: 
                    K = (K_l + K_r) / 2
                    if g(K_l)*g(K) > 0:
                        K_l = K
                    else:
                        K_r = K
                    K = (K_l + K_r) / 2
            else:
                if g(K_l)*epsilon > 0:
                    K_l = K
                else:
                    K_r = K
                K = (K_l + K_r) / 2

            epsilon = g(K)
            grad = g_prime(K)
            
        return Strike(K)
        
    def implied_vol(self, forward: Forward, strike: Strike, premium: Premium) -> ImpliedVol:
        pv = premium.pv
        fv = forward.forward_rate().fv

        def g(sigma):
            return pv - self._premium(forward, strike, OptionType(strike.K >= fv), ImpliedVol(sigma))

        def g_prime(sigma):
            return -self._vega(forward, strike, ImpliedVol(sigma))
        
        sigma_l = self.sigma_lower
        sigma_r = self.sigma_upper
        
        if g(sigma_l) * g(sigma_r) > 0:
            raise ValueError('No solution within implied vol interval')
        
        sigma = (sigma_l + sigma_r) / 2
        epsilon = g(sigma)
        grad = g_prime(sigma)
        while abs(epsilon) > self.tol:   
            if abs(grad) > self.grad_eps:
                sigma -= epsilon / grad
                if sigma > sigma_r or sigma < sigma_l:
                    sigma = (sigma_l + sigma_r) / 2
                    if g(sigma_l)*g(sigma) > 0:
                        sigma_l = sigma
                    else:
                        sigma_r = sigma
                    sigma = (sigma_l + sigma_r) / 2
            else:
                if g(sigma_l)*epsilon > 0:
                    sigma_l = sigma
                else:
                    sigma_r = sigma
                sigma = (sigma_l + sigma_r) / 2
            
            epsilon = g(sigma)
            grad = g_prime(sigma) 

        return ImpliedVol(sigma)
    
    def implied_vols(self, forward: Forward, strikes: Strikes, premiums: Premiums) -> ImpliedVols:
        if not strikes.data.shape == premiums.data.shape:
            raise ValueError('Inconsistent data between strikes and premiums')
        n = len(strikes.data)
        fv = forward.forward_rate().fv
        ivols = np.zeros(n, dtype=np.float64)
        for index in range(n):
            K = strikes.data[index]
            PV = premiums.data[index]
            ivols[index] = self.implied_vol(
                forward,
                Strike(K), 
                Premium(PV)).sigma
        return ImpliedVols(ivols)
    
    def _premium(self, forward: Forward, strike: Strike, option_type: OptionType, implied_vol: ImpliedVol) -> nb.float64:
        pm = 1 if option_type.is_call else -1
        d1 = self._d1(forward, strike, implied_vol)
        d2 = self._d2(d1, forward, implied_vol)
        return pm * forward.S * normal_cdf(pm * d1) - pm * strike.K * \
            np.exp(-forward.r * forward.T) * normal_cdf(pm * d2)
    
    def premium(self, forward: Forward, vanilla: Vanilla, implied_vol: ImpliedVol) -> Premium:
        assert forward.T == vanilla.T
        return Premium(vanilla.N * self._premium(forward, vanilla.strike(), vanilla.option_type(), implied_vol))
    
    def premiums(self, forward: Forward, vanillas: Vanillas, implied_vols: ImpliedVols) -> Premiums:
        assert forward.T == vanillas.T
        ivs = implied_vols.data
        Ks = vanillas.Ks
        assert ivs.shape == Ks.shape
        res_premiums = np.zeros_like(ivs)
        is_calls = vanillas.is_call
        for i in range(len(ivs)):
            res_premiums[i] = self._premium(forward, Strike(Ks[i]), OptionType(is_calls[i]), ImpliedVol(ivs[i]))
        return Premiums(vanillas.Ns * res_premiums)
    
    def _delta(self, forward: Forward, strike: Strike, option_type: OptionType, implied_vol: ImpliedVol) -> nb.float64:
        d1 = self._d1(forward, strike, implied_vol)
        return normal_cdf(d1) if option_type.is_call else normal_cdf(d1) - 1.0
        
    
    def delta(self, forward: Forward, vanilla: Vanilla, implied_vol: ImpliedVol) -> Delta:
        assert forward.T == vanilla.T
        return Delta(vanilla.N*\
            self._delta(forward, vanilla.strike(), vanilla.option_type(), implied_vol)
        )

    def deltas(self, forward: Forward, vanillas: Vanillas, implied_vols: ImpliedVols) -> Deltas:
        assert forward.T == vanillas.T
        ivs = implied_vols.data
        Ks = vanillas.Ks
        assert ivs.shape == Ks.shape
        res_deltas = np.zeros_like(ivs)
        is_call = vanillas.is_call
        for i in range(len(ivs)):
            res_deltas[i] = self._delta(forward, Strike(Ks[i]), OptionType(is_call[i]), ImpliedVol(ivs[i]))
        return Deltas(vanillas.Ns * res_deltas)
    
    def _gamma(self, forward: Forward, strike: Strike, implied_vol: ImpliedVol) -> nb.float64:
        d1 = self._d1(forward, strike, implied_vol) 
        return normal_pdf(d1) / (forward.S * implied_vol.sigma * np.sqrt(forward.T))
    
    def gamma(self, forward: Forward, vanilla: Vanilla, implied_vol: ImpliedVol) -> Gamma:
        assert forward.T == vanilla.T
        return Gamma(vanilla.N*\
            self._gamma(forward, vanilla.strike(), implied_vol)
        )

    def gammas(self, forward: Forward, vanillas: Vanillas, implied_vols: ImpliedVols) -> Gammas:
        assert forward.T == vanillas.T
        ivs = implied_vols.data
        Ks = vanillas.Ks
        assert ivs.shape == Ks.shape
        res_gammas = np.zeros_like(ivs)
        for i in range(len(ivs)):
            res_gammas[i] = self._gamma(forward, Strike(Ks[i]), ImpliedVol(ivs[i]))
        return Gammas(vanillas.Ns * res_gammas)
    
    def _vega(self, forward: Forward, strike: Strike, implied_vol: ImpliedVol) -> nb.float64:
        return forward.S * np.sqrt(forward.T) * normal_pdf(self._d1(forward, strike, implied_vol))
    
    
    def vega(self, forward: Forward, vanilla: Vanilla, implied_vol: ImpliedVol) -> Vega:
        assert forward.T == vanilla.T
        return Vega(vanilla.N*\
            self._vega(forward, vanilla.strike(), implied_vol)
        )
    
    def vegas(self, forward: Forward, vanillas: Vanillas, implied_vols: ImpliedVols) -> Vegas:
        assert forward.T == vanillas.T
        ivs = implied_vols.data
        Ks = vanillas.Ks
        assert ivs.shape == Ks.shape
        res_gammas = np.zeros_like(ivs)
        for i in range(len(ivs)):
            res_gammas[i] = self._gamma(forward, Strike(Ks[i]), ImpliedVol(ivs[i]))
        return Gammas(vanillas.Ns * res_gammas)
    
    
    def _vanna(self, forward: Forward, strike: Strike, implied_vol: ImpliedVol) -> nb.float64:
        d2 = self._d2(self._d1(forward, strike, implied_vol), forward, implied_vol)
        return self._vega(forward, strike, implied_vol) * d2 / (implied_vol.sigma * forward.S)
    
    def vanna(self, forward: Forward, vanilla: Vanilla, implied_vol: ImpliedVol) -> Vanna:
        assert forward.T == vanilla.T
        return Vanna(vanilla.N*\
            self._vanna(forward, vanilla.strike(), implied_vol)
        )
    
    def vannas(self, forward: Forward, vanillas: Vanillas, implied_vols: ImpliedVols) -> Vannas:
        assert forward.T == vanillas.T
        ivs = implied_vols.data
        Ks = vanillas.Ks
        assert ivs.shape == Ks.shape
        res_vannas = np.zeros_like(ivs)
        for i in range(len(ivs)):
            res_vannas[i] = self._vanna(forward, Strike(Ks[i]), ImpliedVol(ivs[i]))
        return Vannas(vanillas.Ns * res_vannas)
    
    def _volga(self, forward: Forward, strike: Strike, implied_vol: ImpliedVol) -> nb.float64:
        d1 = self._d1(forward, strike, implied_vol)
        d2 = self._d2(d1, forward, implied_vol)
        return self._vega(forward, strike, implied_vol) * d1 * d2 / implied_vol.sigma
        
    
    def volga(self, forward: Forward, vanilla: Vanilla, implied_vol: ImpliedVol) -> Volga:
        assert forward.T == vanilla.T
        return Volga(vanilla.N*\
            self._volga(forward, vanilla.strike(), implied_vol)
        )
    
    def volgas(self, forward: Forward, vanillas: Vanillas, implied_vols: ImpliedVols) -> Volgas:
        assert forward.T == vanillas.T
        ivs = implied_vols.data
        Ks = vanillas.Ks
        assert ivs.shape == Ks.shape
        res_volgas = np.zeros_like(ivs)
        for i in range(len(ivs)):
            res_volgas[i] = self._volga(forward, Strike(Ks[i]), ImpliedVol(ivs[i]))
        return Volgas(vanillas.Ns * res_volgas)
    
    def _d1(self, forward: Forward, strike: Strike, implied_vol: ImpliedVol) -> nb.float64:
        d1 = (np.log(forward.S / strike.K) + (forward.r + implied_vol.sigma**2 / 2) * forward.T) / (implied_vol.sigma * np.sqrt(forward.T))
        return d1
    
    def _d2(self, d1: nb.float64, forward: Forward, implied_vol: ImpliedVol) -> nb.float64:
        return d1 - implied_vol.sigma * np.sqrt(forward.T)
    
    def _dDelta_dK(self, forward: Forward, strike: Strike, implied_vol: ImpliedVol) -> nb.float64:
        d1 = self._d1(forward, strike, implied_vol)
        return - normal_pdf(d1) / (strike.K * np.sqrt(forward.T) * implied_vol.sigma)
    