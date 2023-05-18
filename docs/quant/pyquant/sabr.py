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
    ("beta", nb.float64),
    ("rho", nb.float64),
    ("v", nb.float64)
])
class SABRParams:
    def __init__(
        self, volatility: Volatility, correlation: Correlation, volvol: VolOfVol, backbone: Backbone
    ): 
        self.alpha = volatility.alpha
        self.rho = correlation.rho
        self.v = volvol.v
        self.beta = backbone.beta
    
    def array(self) -> nb.float64[:]:
        return np.array([self.alpha, self.rho, self.v])
    
        
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
    def __init__(self):
        self.cached_params = np.array([1., -0.1, 0.0])
        self.calibration_error = 0.
        self.num_iter = 500
        self.tol = 1e-5
        self.strike_lower = 0.1
        self.strike_upper = 10.
        self.delta_tol = 10**-12

    def update_cached_params(self, params: SABRParams):
        self.cached_params = params.array()

    def calibrate(self, chain: VolSmileChain, backbone: Backbone) -> SABRParams:
        forward = chain.f
        tenor = chain.T
        strikes = chain.K
        implied_vols = chain.sigma
        beta = backbone.beta
    
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
                    params,
                    beta
                )
            )
            iv = self._vol_sabr(
                forward,
                tenor,
                strikes,
                params,
                beta
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

            for i in range(self.num_iter):
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

        self.cached_params, self.calibration_error \
            = levenberg_marquardt(get_residuals, clip_params, self.cached_params)
        
        return SABRParams(
            Volatility(self.cached_params[0]),
            Correlation(self.cached_params[1]),
            VolOfVol(self.cached_params[2]),
            backbone
        )
        
    def implied_vol(self, forward: Forward, strike: Strike, params: SABRParams) -> ImpliedVol:
        return ImpliedVol(
            self._vol_sabr(forward.forward_rate().fv, forward.T, np.array([strike.K]), params.array(), params.beta)[0]
        )
    

    def implied_vols(self, forward: Forward, strikes: Strikes, params: SABRParams) -> ImpliedVols:
        return ImpliedVols(
            self._vol_sabr(forward.forward_rate().fv, forward.T, strikes.data,  params.array(), params.beta)
        )

    def strike_from_delta(self, forward: Forward, delta: Delta, params: SABRParams) -> Strike:
        K_l = self.strike_lower*forward.S
        K_r = self.strike_lower*forward.S
        F = forward.forward_rate().fv
        T = forward.T

        bs = BlackScholesCalc()
        option_type = OptionType(delta.pv >= 0.)

        def g(K):
            return bs.delta(forward, Strike(K), self.implied_vol(forward, Strike(K), params), option_type).pv - delta.pv

        def g_prime(K): 
            iv = self.implied_vol(forward, Strike(K), params).sigma
            dsigma_dk = self._dsigma_dK(F, T, K, params)
            d1 = bs._d1(forward, Strike(K), iv)
            sigma = iv.sigma
            return np.exp(-d1**2 / 2)/np.sqrt(T)*(- 1/(K*sigma) - dsigma_dk*np.log(F/K)/sigma**2\
                                                   - forward.r*T*dsigma_dk/sigma**2 + T*dsigma_dk)
    
        assert g(K_l)*g(K_r) <= 0.
        
        K = (K_l + K_r) / 2
        epsilon = g(K)
        grad = g_prime(K)

        i = 0
        while abs(epsilon) > self.delta_tol and i < 100: 
            if abs(grad) > 1e-6:
                K -= epsilon / grad
                if K > K_r or K < K_l:
                    K = (K_l + K_r) / 2
                    if g(K_l)*epsilon > 0:
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
            i += 1
        
        return Strike(K)

    def delta_space(self, forward: Forward, params: SABRParams) -> VolSmileDeltaSpace:
        
        atm = self.implied_vol(forward, Strike(forward.forward_rate().fv), params)

        call25_K = self.strike_from_delta(forward, Delta(0.25), params) 
        call25 = self.implied_vol(forward, call25_K, params)

        put25_K = self.strike_from_delta(forward, Delta(-0.25), params)
        put25 = self.implied_vol(forward, put25_K, params)

        call10_K = self.strike_from_delta(forward, Delta(0.1), params)
        call10 = self.implied_vol(forward, call10_K, params)

        put10_K = self.strike_from_delta(forward, Delta(-0.1), params)
        put10 = self.implied_vol(forward, put10_K, params)


        return VolSmileDeltaSpace(
            forward,
            Straddle(atm.sigma),
            RiskReversal(Delta(0.25), call25.sigma - put25.sigma, Tenor(forward.T)),
            Butterfly(Delta(0.25), 0.5*(call25.sigma + put25.sigma) - atm.sigma, Tenor(forward.T)),
            RiskReversal(Delta(0.1), call10.sigma - put10.sigma, Tenor(forward.T)),
            Butterfly(Delta(0.1), 0.5*(call10.sigma + put10.sigma) - atm.sigma, Tenor(forward.T))
        )

    def _vol_sabr(
        self,
        F: nb.float64,
        T: nb.float64,
        Ks: nb.float64[:],
        params: nb.float64[:],
        beta: nb.float64
    ) ->  nb.float64[:]: 
        
        alpha, rho, v = params[0], params[1], params[2]
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
        params: nb.float64[:],
        beta: nb.float64       
    ) ->  tuple[nb.float64[:]]: 
        
        n = len(Ks)
        alpha, rho, v = params[0], params[1], params[2]
       
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

            sig_alpha = dI_B_0_dalpha * (1 + I_H * T) + dI_H_1_dalpha * I_B * T
            sig_v = dI_B_0_dv * (1 + I_H * T) + dI_h_v * I_B * T
            sig_rho = dI_B_0_drho * (1 + I_H * T) + dI_H_rho * I_B * T

            ddalpha[index] = sig_alpha
            ddv[index] = sig_v
            ddrho[index] = sig_rho

        return ddalpha, ddrho, ddv
       
    def _dsigma_dK(self, F: nb.float64, T: nb.float64, K: nb.float64, params: SABRParams) -> nb.float64:
        x = np.log(F / K)
        alpha, rho, v = params.alpha, params.rho, params.v
        beta = params.beta
        if x == 0.0:
            I_B_0 = K ** (beta - 1) * alpha
            dI_B_0_dK = K ** (beta - 2) * alpha * (beta - 1)
        elif v == 0.0:
            I_B_0 = alpha * (1 - beta) * x / (-(K ** (1 - beta)) + F ** (1 - beta))

            dI_B_0_dK = (
                alpha
                * (beta - 1)
                * (
                    -K * (K ** (1 - beta) - F ** (1 - beta))
                    + K ** (2 - beta) * (beta - 1) * np.log(F / K)
                )
                / (K**2 * (K ** (1 - beta) - F ** (1 - beta)) ** 2)
            )
        else:
            if beta == 1.0:
                z = v * x / alpha
                sqrt = np.sqrt(1 - 2 * rho * z + z**2)
                epsilon = np.log((-sqrt + rho - z) / (rho - 1))
                dI_B_0_dK = (
                    v
                    * (-sqrt * alpha * epsilon + v * x)
                    / (K * sqrt * alpha * epsilon**2)
                )
            elif beta < 1.0:
                z = v * (F ** (1 - beta) - K ** (1 - beta)) / (alpha * (1 - beta))
                sqrt = np.sqrt(1 - 2 * rho * z + z**2)
                epsilon = np.log((-sqrt + rho - z) / (rho - 1))

                dI_B_0_dK = (
                    v
                    * (K * v * x - K**beta * sqrt * alpha * epsilon)
                    / (K * K**beta * sqrt * alpha * epsilon**2)
                )
            I_B_0 = v * x / (epsilon)

        I_H_1 = (
            alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 / 24
            + alpha * beta * rho * v * (K * F) ** (beta / 2 + -1 / 2) / 4
            + v**2 * (2 - 3 * rho**2) / 24
        )
        dI_H_1_dK = alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 * (beta - 1) / (
            24 * K
        ) + alpha * beta * rho * v * (K * F) ** (beta / 2 - 1 / 2) * (beta / 2 + -1 / 2) / (
            4 * K
        )

        dsigma_dK = dI_B_0_dK * (1 + I_H_1 * T) + dI_H_1_dK * I_B_0 * T
        return dsigma_dK