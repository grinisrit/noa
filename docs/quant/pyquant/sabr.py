import numpy as np
import numba as nb

from .common import *
from .black_scholes import *
from .vol_surface import *


@nb.experimental.jitclass([
    ("beta", nb.float64)
])
class Backbone:
    def __init__(self, beta: nb.float64):
        assert beta <= 1
        assert beta >= 0
        self.beta = beta


@nb.experimental.jitclass([
    ("alpha", nb.float64)
])
class Volatility:
    def __init__(self, alpha: nb.float64):
        self.alpha = alpha 


@nb.experimental.jitclass([
    ("rho", nb.float64)
])
class Correlation:
    def __init__(self, rho: nb.float64):
        self.rho = rho 


@nb.experimental.jitclass([
    ("v", nb.float64)
])
class VolOfVol:
    def __init__(self, v: nb.float64):
        self.v = v 


@nb.experimental.jitclass([
    ("alpha", nb.float64),
    ("rho", nb.float64),
    ("v", nb.float64)
])
class SABRParams:
    def __init__(
        self, volatility: Volatility, correlation: Correlation, volvol: VolOfVol
    ): 
        self.alpha = volatility.alpha
        self.rho = correlation.rho
        self.v = volvol.v

        
@nb.experimental.jitclass([
    ("beta", nb.float64)
])
class Backbone:
    def __init__(self, beta: nb.float64):
        assert beta <= 1
        assert beta >= 0
        self.beta = beta
     
        
@nb.experimental.jitclass([
    ("beta", nb.float64),
    ("cached_params", nb.float64[:]),
    ("calibration_error", nb.float64),
    ("num_iter", nb.int64),
    ("tol", nb.float64),
    ("strike_lower", nb.float64),
    ("strike_upper", nb.float64),
    ("delta_tol", nb.float64)
])        
class SABRCalc:
    def __init__(self, backbone: Backbone):
        self.beta = backbone.beta
        self.cached_params = np.array([1., -0.1, 0.0])
        self.calibration_error = 0.
        self.num_iter = 500
        self.tol = 1e-5
        self.strike_lower = 0.1
        self.strike_upper = 10.
        self.delta_tol = 10**-12

    def update_cached_params(self, params: SABRParams):
        self.cached_params = np.array([params.alpha, params.rho, params.v])

    def calibrate(self, chain: VolSmileChain) -> SABRParams:
        forward = chain.f
        tenor = chain.T
        strikes = chain.K
        implied_vols = chain.sigma
    
        def clip_params(params):
            eps = 1e-4
            alpha, rho, v = params[0], params[1], params[2]
            alpha = np_clip(alpha, eps, 50.0)
            v = np_clip(v, eps, 50.0)
            rho = np_clip(rho, -1.0 + eps, 1.0 - eps)
            sabr_params = np.array([alpha, rho, v])
            return sabr_params
        
        def get_residuals(params):
            J = np.stack( 
                self._jacobian_sabr(
                    forward,
                    tenor,
                    strikes,
                    params
                )
            )
            iv = self._vol_sabr(
                forward,
                tenor,
                strikes,
                params
            )
            weights = np.ones_like(strikes)
            weights = weights / np.sum(weights)
            res = iv - implied_vols
            return res * weights, J @ np.diag(weights)
        
        def levenberg_marquardt(f, proj, x0):
            x = x0.copy()

            mu = 100.0
            nu1 = 2.0
            nu2 = 2.0

            res, J = f(x)
            F = np.linalg.norm(res)

            result_x = x
            result_error = F

            for i in range(self._num_iter):
                multipl = J @ J.T
                I = np.diag(np.diag(multipl)) + 1e-5 * np.eye(len(x))
                dx = np.linalg.solve(mu * I + multipl, J @ res)
                x_ = proj(x - dx)
                res_, J_ = f(x_)
                F_ = np.linalg.norm(res_)
                if F_ < F:
                    x, F, res, J = x_, F_, res_, J_
                    mu /= nu1
                    result_error = F
                else:
                    i -= 1
                    mu *= nu2
                    continue
                if F < self.tol:
                    break
                result_x = x
            return result_x, result_error

        self.cached_params, self._calibration_error \
            = levenberg_marquardt(500, get_residuals, clip_params, self.cached_params)
        
    def implied_vol(self, forward: Forward, strike: Strike, params: SABRParams) -> ImpliedVol:
        return ImpliedVol(
            self._vol_sabr(forward.forward_rate().fv, forward.T, np.array([strike.K]), np.array([params.alpha, params.rho, params.v]))[0]
        )

    def implied_vols(self, forward: Forward, strikes: Strikes, params: SABRParams) -> ImpliedVols:
        return ImpliedVols(
            self._vol_sabr(forward.forward_rate().fv, forward.T, strikes.data, np.array([params.alpha, params.rho, params.v]))
        )

    def strike_from_delta(self, forward: Forward, delta: Delta, params: SABRParams) -> Strike:

        K_l = self.strike_lower*forward.S
        K_r = self.strike_lower*forward.S

        option_type = OptionType(delta.pv >= 0.)

        def g_deltaspc(K, delta, T, F, r, is_call=True):
            market = MarketParameters(
                F=F,
                r=r,
                T=T,
                K=np.array([np.float64(K)]),
                # can lay zero, not needed to calculate volatility
                iv=np.array([np.float64(0.0)]),
                types=np.array([np.bool(is_call)]),
            )
            sigma = get_vol(model, market)[0]
            delta_diff = get_delta_bsm(is_call, sigma, K, T, F, r) - delta
            return delta_diff


        def g_deltaspc_prime(model, K, T, F, r, is_call):
            market = MarketParameters(
                F=F,
                r=r,
                T=T,
                K=np.array([np.float64(K)]),
                # can lay zero, not needed to calculate iv
                iv=np.array([np.float64(0.0)]),
                types=np.array([np.bool(is_call)]),
            )
            sigma = get_vol(model, market)[0]
            dsigma_dk = get_dsigma_dK(model, K, T, F)
            d1 = get_d1(sigma, K, T, F, r)
            return np.exp(-d1**2 / 2)/np.sqrt(T)*(- 1/(K*sigma) - dsigma_dk*np.log(F/K)/sigma**2 - r*T*dsigma_dk/sigma**2 + T*dsigma_dk)
    

        delta_left_strike = g_deltaspc(model, K_l, delta, T, F, r, is_call)
        delta_right_strike = g_deltaspc(model, K_r, delta, T, F, r, is_call)

        if delta_left_strike*delta_right_strike > 0:
            print('no zero at the initial interval')
            return 0.
        else:
            K = (K_l + K_r) / 2
            epsilon = g_deltaspc(model, K, delta, T, F, r, is_call)
            grad = g_deltaspc_prime(model, K, T, F, r, is_call)
            i = 0
            while abs(epsilon) > tol and i < 100: 
                if abs(grad) > 1e-6:
                    K -= epsilon / grad
                    if K > K_r or K < K_l:
                        K = (K_l + K_r) / 2
                        if g_deltaspc(model, K_l, delta, T, F, r, is_call)*epsilon > 0:
                            K_l = K
                        else:
                            K_r = K
                        K = (K_l + K_r) / 2
                else:
                    if g_deltaspc(model, K_l, delta, T, F, r, is_call)*epsilon > 0:
                        K_l = K
                    else:
                        K_r = K
                    K = (K_l + K_r) / 2
                
                epsilon = g_deltaspc(model, K, delta, T, F, r, is_call)
                grad = g_deltaspc_prime(model, K, T, F, r, is_call)
                i += 1
            print("eps", epsilon)
            return K

    def delta_space(self, params: SABRParams, forward: Forward) -> VolSmileDeltaSpace:
        pass

    def _vol_sabr(
        self,
        F: nb.float64,
        T: nb.float64,
        Ks: nb.float64[:],
        params: nb.float64[:]
    ) ->  nb.float64[:]: 
        
        alpha, rho, v = params[0], params[1], params[2]
        beta = self.beta
        n = len(Ks)
        sigmas = np.zeros(n, dtype=np.float64)
        for index in range(n):
            K = Ks[index]
            x = np.log(F / K)
            I_H_1 = (
                alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 / 24
                + alpha * beta * rho * v * (K * F) ** (beta / 2 + -1 / 2) / 4
                + v**2 * (2 - 3 * rho**2) / 24
            )
            if x == 0.0:
                I_B_0 = K ** (beta - 1) * alpha
            elif v == 0.0:
                I_B_0 = alpha * (1 - beta) * x / (-(K ** (1 - beta)) + F ** (1 - beta))
            else:
                if beta == 1.0:
                    z = v * x / alpha
                elif beta < 1.0:
                    z = v * (F ** (1 - beta) - K ** (1 - beta)) / (alpha * (1 - beta))
                sqrt = np.sqrt(1 - 2 * rho * z + z**2)
                epsilon = np.log((-sqrt + rho - z) / (rho - 1))
                I_B_0 = v * x / epsilon

            sigma = I_B_0 * (1 + I_H_1 * T)
            sigmas[index] = sigma
        return sigmas

    def _jacobian_sabr(
        self,
        F: nb.float64,
        T: nb.float64,
        Ks: nb.float64[:],
        params: nb.float64[:]       
    ) ->  tuple[nb.float64[:]]: 
        
        n = len(Ks)
        alpha, rho, v = params[0], params[1], params[2]
        beta = self.beta

        ddalpha = np.zeros(n, dtype=np.float64)
        ddv = np.zeros(n, dtype=np.float64)
        ddrho = np.zeros(n, dtype=np.float64)
        
        for index in range(n):
            K = Ks[index]
            x = np.log(F / K)
            I_H = (
                alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 / 24
                + alpha * beta * rho * v * (K * F) ** (beta / 2 - 1 / 2) / 4
                + v**2 * (2 - 3 * rho**2) / 24
            )
            dI_H_1_dalpha = (
                alpha * (K * F) ** (beta - 1) * (1 - beta) ** 2 / 12
                + beta * rho * v * (K * F) ** (beta / 2 - 1 / 2) / 4
            )
            dI_h_v = (
                alpha * beta * rho * (K * F) ** (beta / 2 + -1 / 2) / 4
                + v * (2 - 3 * rho**2) / 12
            )
            dI_H_rho = (
                alpha * beta * v * (K * F) ** (beta / 2 + -1 / 2) / 4 - rho * v**2 / 4
            )

            if x == 0.0:
                I_B = alpha * K ** (beta - 1)
                dI_B_0_dalpha = K ** (beta - 1)
                dI_B_0_dv = 0.0
                dI_B_0_drho = 0.0

            elif v == 0.0:
                I_B = alpha * (1 - beta) * x / (F ** (1 - beta) - (K ** (1 - beta)))
                dI_B_0_dalpha = (beta - 1) * x / (K ** (1 - beta) - F ** (1 - beta))
                dI_B_0_dv = 0.0
                dI_B_0_drho = 0.0

            elif beta == 1.0:
                z = v * x / alpha
                sqrt = np.sqrt(1 - 2 * rho * z + z**2)
                epsilon = np.log((-sqrt + rho - z) / (rho - 1))
                I_B = v * x / epsilon
                dI_B_0_dalpha = v * x * z / (alpha * sqrt * epsilon**2)
                dI_B_0_dv = (
                    x * (alpha * sqrt * epsilon - v * x) / (alpha * sqrt * epsilon**2)
                )
                dI_B_0_drho = (
                    v
                    * x
                    * ((rho - 1) * (z + sqrt) + (-rho + z + sqrt) * sqrt)
                    / ((rho - 1) * (-rho + z + sqrt) * sqrt * epsilon**2)
                )

            elif beta < 1.0:
                z = v * (F ** (1 - beta) - K ** (1 - beta)) / (alpha * (1 - beta))
                sqrt = np.sqrt(1 - 2 * rho * z + z**2)
                epsilon = np.log((-sqrt + rho - z) / (rho - 1))
                I_B = v * x / epsilon
                dI_B_0_dalpha = v * x * z / (alpha * sqrt * epsilon**2)
                dI_B_0_dv = (
                    x
                    * (
                        sqrt * alpha * (beta - 1) * epsilon
                        - v * (K ** (1 - beta) - F ** (1 - beta))
                    )
                    / (sqrt * alpha * (beta - 1) * epsilon**2)
                )

                dI_B_0_drho = (
                    v
                    * x
                    * (sqrt * (sqrt - rho + z) + (sqrt + z) * (rho - 1))
                    / (sqrt * (rho - 1) * (sqrt - rho + z) * epsilon**2)
                )

            sig_alpha = dI_B_0_dalpha * (1 + I_H * T) + dI_H_1_dalpha * I_B * T T
            sig_v = dI_B_0_dv * (1 + I_H * T) + dI_h_v * I_B * T
            sig_rho = dI_B_0_drho * (1 + I_H * T) + dI_H_rho * I_B * T

            ddalpha[index] = sig_alpha
            ddv[index] = sig_v
            ddrho[index] = sig_rho

        return ddalpha, ddrho, ddv
       