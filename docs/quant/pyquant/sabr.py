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
        if not (beta <= 1 and beta >= 0):
            raise ValueError('Backbone not within [0,1]')
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
    
    def backbone(self) -> Backbone:
        return Backbone(self.beta)
    
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
    ("beta", nb.float64),
    ("cached_params", nb.float64[:]),
    ("calibration_error", nb.float64),
    ("num_iter", nb.int64),
    ("delta_num_iter", nb.int64),
    ("tol", nb.float64),
    ("strike_lower", nb.float64),
    ("strike_upper", nb.float64),
    ("delta_tol", nb.float64),
    ("delta_grad_eps", nb.float64)
])        
class SABRCalc:
    bs_calc: BSCalc

    def __init__(self):
        self.cached_params = np.array([1., 0.0, 0.0])
        self.calibration_error = 0.
        self.num_iter = 50
        self.tol = 1e-8
        self.strike_lower = 0.1
        self.strike_upper = 10.
        self.delta_tol = 1e-8
        self.delta_num_iter = 500
        self.delta_grad_eps = 1e-4
        self.bs_calc = BSCalc()

    def update_cached_params(self, params: SABRParams):
        self.cached_params = params.array()

    def calibrate(self, chain: VolSmileChainSpace, backbone: Backbone, calibration_weights: CalibrationWeights) -> SABRParams:
        strikes = chain.Ks
        w = calibration_weights.w
        
        if not strikes.shape == w.shape:
            raise ValueError('Inconsistent data between strikes and calibration weights')
            
        m_n = len(strikes) - 2
        
        if not m_n > 0:
            raise ValueError('Need at least 3 points to calibrate SABR')
     
        weights = w / w.sum()
        forward = chain.f
        time_to_maturity = chain.T
        implied_vols = chain.sigmas
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
                    time_to_maturity,
                    strikes,
                    params,
                    beta
                )
            )
            iv = self._vol_sabr(
                forward,
                time_to_maturity,
                strikes,
                params,
                beta
            )
            res = iv - implied_vols
            return res * weights, J @ np.diag(weights)
        
        def levenberg_marquardt(f, proj, x0):
            x = x0.copy()

            mu = 1e-2
            nu1 = 2.0
            nu2 = 2.0

            res, J = f(x)
            F = res.T @ res

            result_x = x
            result_error = F / m_n

            for i in range(self.num_iter):
                if result_error < self.tol:
                    break
                multipl = J @ J.T
                I = np.diag(np.diag(multipl)) + 1e-5 * np.eye(len(x))
                dx = np.linalg.solve(mu * I + multipl, J @ res)
                x_ = proj(x - dx)
                res_, J_ = f(x_)
                F_ = res_.T @ res_
                if F_ < F:
                    x, F, res, J = x_, F_, res_, J_
                    mu /= nu1
                    result_error = F / m_n
                else:
                    i -= 1
                    mu *= nu2
                    continue
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
    
    def premium(self, forward: Forward, strike: Strike, option_type: OptionType, params: SABRParams) -> Premium:
        sigma =\
            self._vol_sabr(forward.forward_rate().fv, forward.T, np.array([strike.K]), params.array(), params.beta)[0]
        return self.bs_calc.premium(forward, strike, option_type, ImpliedVol(sigma))
   
    def premiums(self, forward: Forward, strikes: Strikes, params: SABRParams) -> Premiums:
        f = forward.forward_rate().fv
        sigmas =\
            self._vol_sabr(f, forward.T, strikes.data,  params.array(), params.beta)
        Ks = strikes.data
        res = np.zeros_like(sigmas)
        n = len(sigmas)
        for i in range(n):
            K = Ks[i]
            res[i] = self.bs_calc.premium(forward, Strike(K), OptionType(K >= f), ImpliedVol(sigmas[i])).pv
        return Premiums(res)
    
    def vanilla_premium(self, spot: Spot, forward_yield: ForwardYield, vanilla: Vanilla, params: SABRParams) -> Premium:
        forward = Forward(spot, forward_yield, vanilla.time_to_maturity())
        return Premium(vanilla.N*\
            self.premium(forward, vanilla.strike(), vanilla.option_type(), params).pv
        )
    
    def vanillas_premiums(self, spot: Spot, forward_yield: ForwardYield, vanillas: Vanillas, params: SABRParams) -> Premiums:
        forward = Forward(spot, forward_yield, vanillas.time_to_maturity())
        f = forward.forward_rate().fv
        Ks = vanillas.Ks
        sigmas =\
            self._vol_sabr(f, forward.T, Ks,  params.array(), params.beta)
        Ns = vanillas.Ns
        res = np.zeros_like(sigmas)
        n = len(sigmas)
        for i in range(n):
            K = Ks[i]
            N = Ns[i]
            is_call = vanillas.is_call[i]
            res[i] = N*self.bs_calc.premium(forward, Strike(K), OptionType(is_call), ImpliedVol(sigmas[i])).pv
        return Premiums(res)
        

    def strike_from_delta(self, forward: Forward, delta: Delta, params: SABRParams) -> Strike:
        F = forward.forward_rate().fv
        K_l = self.strike_lower*F
        K_r = self.strike_upper*F
        T = forward.T
        option_type = OptionType(delta.pv >= 0.)

        def g(K):
            iv = self.implied_vol(forward, Strike(K), params)  
            return self.bs_calc.delta(forward, Strike(K), option_type, iv).pv - delta.pv

        def g_prime(K): 
            iv = self.implied_vol(forward, Strike(K), params)
            dsigma_dk = self._dsigma_dK(F, T, K, params)
            d1 = self.bs_calc._d1(forward, Strike(K), iv)
            sigma = iv.sigma
            return np.exp(-d1**2 / 2)/np.sqrt(T)*(- 1/(K*sigma) - dsigma_dk*np.log(F/K)/sigma**2\
                                                   - forward.r*T*dsigma_dk/sigma**2 + T*dsigma_dk)
        if g(K_l)*g(K_r) > 0.:
            raise ValueError('No solution within strikes interval')
        
        K = (K_l + K_r) / 2
        epsilon = g(K)
        grad = g_prime(K)
        ii = 0
        while abs(epsilon) > self.delta_tol and ii < self.delta_num_iter: 
            ii = ii + 1    
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

    def delta_space(self, forward: Forward, params: SABRParams) -> VolSmileDeltaSpace:
        
        atm = self.implied_vol(forward, Strike(forward.forward_rate().fv), params).sigma

        call25_K = self.strike_from_delta(forward, Delta(0.25), params) 
        call25 = self.implied_vol(forward, call25_K, params).sigma

        put25_K = self.strike_from_delta(forward, Delta(-0.25), params)
        put25 = self.implied_vol(forward, put25_K, params).sigma

        call10_K = self.strike_from_delta(forward, Delta(0.1), params)
        call10 = self.implied_vol(forward, call10_K, params).sigma
       
        put10_K = self.strike_from_delta(forward, Delta(-0.1), params)
        put10 = self.implied_vol(forward, put10_K, params).sigma


        return VolSmileDeltaSpace(
            forward,
            Straddle(ImpliedVol(atm), TimeToMaturity(forward.T)),
            RiskReversal(Delta(0.25), VolatilityQuote(call25 - put25), TimeToMaturity(forward.T)),
            Butterfly(Delta(0.25), 
                      VolatilityQuote(0.5*(call25 + put25) - atm), TimeToMaturity(forward.T)),
            RiskReversal(Delta(0.1), 
                         VolatilityQuote(call10 - put10), TimeToMaturity(forward.T)),
            Butterfly(Delta(0.1), 
                      VolatilityQuote(0.5*(call10 + put10) - atm), TimeToMaturity(forward.T))
        )
    
    def delta(self, forward: Forward, strike: Strike, option_type: OptionType, params: SABRParams) -> Delta:
        F = forward.forward_rate().fv
        D = forward.numeraire().pv
        sigma = self.implied_vol(forward, strike, params)

        delta_bsm = self.bs_calc.delta(forward, strike, option_type, sigma).pv
        vega_bsm = self.bs_calc.vega(forward, strike, sigma).pv
        dsigma_df = self._dsigma_df(F, forward.T, strike.K, params)

        dsigma_dalpha = self._jacobian_sabr(F, forward.T,
            np.array([strike.K]),
            params.array(),
            params.beta
        )[0][0]
        
        return Delta(
            delta_bsm + (1/D) * vega_bsm * (dsigma_df + dsigma_dalpha * params.rho * params.v / F**params.beta)
        )
    
    def vanilla_delta(self, spot: Spot, forward_yield: ForwardYield, vanilla: Vanilla, params: SABRParams) -> Delta:
        forward = Forward(spot, forward_yield, vanilla.time_to_maturity())
        return Delta(vanilla.N *\
                     self.delta(forward, vanilla.strike(), vanilla.option_type(), params).pv
        )
        
    def deltas(self, forward: Forward, strikes: Strikes, params: SABRParams) -> Deltas:
        F = forward.forward_rate().fv
        D = forward.numeraire().pv
        sigmas = self.implied_vols(forward, strikes, params).data
        Ks = strikes.data
        n = len(sigmas)
        delta_bsm = np.zeros_like(sigmas)
        vega_bsm = np.zeros_like(sigmas)
        dsigma_df = np.zeros_like(sigmas)

        for i in range(n):
            K = Ks[i]
            sigma = sigmas[i]
            delta_bsm[i] = self.bs_calc.delta(forward, Strike(K), OptionType(K>=F), ImpliedVol(sigma)).pv
            vega_bsm[i] = self.bs_calc.vega(forward, Strike(K), ImpliedVol(sigma)).pv

            dsigma_df[i] = self._dsigma_df(F, forward.T, K, params)

        dsigma_dalpha = self._jacobian_sabr(F, forward.T,
            Ks,
            params.array(),
            params.beta
        )[0] 
        return Deltas(
            delta_bsm + (1/D) * vega_bsm * (dsigma_df + dsigma_dalpha * params.rho * params.v / F**params.beta)
        )
 
    def vanillas_deltas(self, spot: Spot, forward_yield: ForwardYield, vanillas: Vanillas, params: SABRParams) -> Delta:
        forward = Forward(spot, forward_yield, vanillas.time_to_maturity())
        F = forward.forward_rate().fv
        D = forward.numeraire().pv
        Ks = vanillas.Ks
        sigmas = self.implied_vols(forward, Strikes(Ks), params).data
        n = len(sigmas)
        delta_bsm = np.zeros_like(sigmas)
        vega_bsm = np.zeros_like(sigmas)
        dsigma_df = np.zeros_like(sigmas)

        for i in range(n):
            K = Ks[i]
            sigma = sigmas[i]
            is_call = vanillas.is_call[i]
            delta_bsm[i] = self.bs_calc.delta(forward, Strike(K), OptionType(is_call), ImpliedVol(sigma)).pv
            vega_bsm[i] = self.bs_calc.vega(forward, Strike(K), ImpliedVol(sigma)).pv

            dsigma_df[i] = self._dsigma_df(F, forward.T, K, params)

        dsigma_dalpha = self._jacobian_sabr(F, forward.T,
            Ks,
            params.array(),
            params.beta
        )[0] 
        return Deltas(vanillas.Ns*(
            delta_bsm + (1/D) * vega_bsm * (dsigma_df + dsigma_dalpha * params.rho * params.v / F**params.beta)
        ))
        
     
    def gamma(self, forward: Forward, strike: Strike, params: SABRParams) -> Gamma:
        F = forward.forward_rate().fv
        D = forward.numeraire().pv
        sigma = self.implied_vol(forward, strike, params)

        gamma_bsm = self.bs_calc.gamma(forward, strike, sigma).pv
        vega_bsm = self.bs_calc.vega(forward, strike, sigma).pv
        vanna_bsm = self.bs_calc.vanna(forward, strike, sigma).pv
        volga_bsm = self.bs_calc.volga(forward, strike, sigma).pv
                
        dsigma_df = self._dsigma_df(F, forward.T, strike.K, params)
        d2_sigma_df2 = self._d2_sigma_df2(F, forward.T, strike.K, params)
        d2_sigma_dalpha_df = self._d2_sigma_dalpha_df(F, forward.T, strike.K, params)

        dsigma_dalpha = self._jacobian_sabr(F, forward.T,
            np.array([strike.K]),
            params.array(),
            params.beta
        )[0][0]

        sticky_component = dsigma_df + dsigma_dalpha * params.rho * params.v / F**params.beta
        return Gamma(
            gamma_bsm
            + (2/D) * vanna_bsm * sticky_component
            + (volga_bsm / (D**2)) * sticky_component**2
            + (vega_bsm / (D**2)) 
            * (
                d2_sigma_df2
                + d2_sigma_dalpha_df * params.rho * params.v / F**params.beta
                - dsigma_dalpha * params.beta * params.rho * params.v / F ** (params.beta + 1)
            )
        )
    
    def vanilla_gamma(self, spot: Spot, forward_yield: ForwardYield, vanilla: Vanilla, params: SABRParams) -> Gamma:
        forward = Forward(spot, forward_yield, vanilla.time_to_maturity())
        return Gamma(vanilla.N*\
            self.gamma(forward, vanilla.strike(), params).pv
        )
     
    def gammas(self, forward: Forward, strikes: Strikes, params: SABRParams) -> Gammas:
        F = forward.forward_rate().fv
        D = forward.numeraire().pv
        sigmas = self.implied_vols(forward, strikes, params).data
        Ks = strikes.data
        n = len(sigmas)

        gamma_bsm = np.zeros_like(sigmas)
        vega_bsm = np.zeros_like(sigmas)
        vanna_bsm = np.zeros_like(sigmas)
        volga_bsm = np.zeros_like(sigmas)

        dsigma_df = np.zeros_like(sigmas)
        d2_sigma_df2 = np.zeros_like(sigmas)
        d2_sigma_dalpha_df = np.zeros_like(sigmas)


        for i in range(n):
            K = Ks[i]
            sigma = ImpliedVol(sigmas[i])
            gamma_bsm[i] = self.bs_calc.gamma(forward, Strike(K), sigma).pv
            vega_bsm[i] = self.bs_calc.vega(forward, Strike(K), sigma).pv
            vanna_bsm[i] = self.bs_calc.vanna(forward, Strike(K), sigma).pv
            volga_bsm[i] = self.bs_calc.volga(forward, Strike(K), sigma).pv

            dsigma_df[i] = self._dsigma_df(F, forward.T, K, params)
            d2_sigma_df2[i] = self._d2_sigma_df2(F, forward.T, K, params)
            d2_sigma_dalpha_df[i] = self._d2_sigma_dalpha_df(F, forward.T, K, params)

        dsigma_dalpha = self._jacobian_sabr(F, forward.T,
            Ks,
            params.array(),
            params.beta
        )[0]

        sticky_component = dsigma_df + dsigma_dalpha * params.rho * params.v / F**params.beta
        return Gammas(
            gamma_bsm
            + (2/D) * vanna_bsm * sticky_component
            + (volga_bsm / (D**2)) * sticky_component**2
            + (vega_bsm / (D**2)) 
            * (
                d2_sigma_df2
                + d2_sigma_dalpha_df * params.rho * params.v / F**params.beta
                - dsigma_dalpha * params.beta * params.rho * params.v / F ** (params.beta + 1)
            )
        )
    
    def vanillas_gammas(self, spot: Spot, forward_yield: ForwardYield, vanillas: Vanillas, params: SABRParams) -> Gammas:
        forward = Forward(spot, forward_yield, vanillas.time_to_maturity())
        return Gammas(vanillas.Ns*\
            self.gammas(forward, vanillas.strikes(), params).data
        )  
    
    def vega(self, forward: Forward, strike: Strike, params: SABRParams) -> Vega:
        F = forward.forward_rate().fv
        sigma = self.implied_vol(forward, strike, params)

        vega_bsm = self.bs_calc.vega(forward, strike, sigma).pv

        dsigma_df = self._dsigma_df(F, forward.T, strike.K, params)

        dsigma_dalpha = self._jacobian_sabr(F, forward.T,
            np.array([strike.K]),
            params.array(),
            params.beta
        )[0][0]

        if params.v > 0 : 
            res = vega_bsm * (dsigma_dalpha + dsigma_df * params.rho * F**params.beta / params.v)
        else:
            res = vega_bsm * dsigma_dalpha

        return Vega(res)
    
    def vanilla_vega(self, spot: Spot, forward_yield: ForwardYield, vanilla: Vanilla, params: SABRParams) -> Vega:
        forward = Forward(spot, forward_yield, vanilla.time_to_maturity())
        return Vega(vanilla.N*\
            self.vega(forward, vanilla.strike(), params).pv
        )
        
    def vegas(self, forward: Forward, strikes: Strikes, params: SABRParams) -> Vegas:
        F = forward.forward_rate().fv
        sigmas = self.implied_vols(forward, strikes, params).data
        Ks = strikes.data
        n = len(sigmas)
       
        vega_bsm = np.zeros_like(sigmas)
        dsigma_df = np.zeros_like(sigmas)

        for i in range(n):
            K = Ks[i]
            sigma = ImpliedVol(sigmas[i])
            vega_bsm[i] = self.bs_calc.vega(forward, Strike(K), sigma).pv
            dsigma_df[i] = self._dsigma_df(F, forward.T, K, params)

        dsigma_dalpha = self._jacobian_sabr(F, forward.T,
            Ks,
            params.array(),
            params.beta
        )[0] 

        if params.v > 0 : 
            res = vega_bsm * (dsigma_dalpha + dsigma_df * params.rho * F**params.beta / params.v)
        else:
            res = vega_bsm * dsigma_dalpha

        return Vegas(res)
   
    def vanillas_vegas(self, spot: Spot, forward_yield: ForwardYield, vanillas: Vanilla, params: SABRParams) -> Vegas:
        forward = Forward(spot, forward_yield, vanillas.time_to_maturity())
        return Vegas(vanillas.Ns*\
            self.vegas(forward, vanillas.strikes(), params).data
        )
    
    
    def rega(self, forward: Forward, strike: Strike, params: SABRParams) -> Rega:
        F = forward.forward_rate().fv
        sigma = self.implied_vol(forward, strike, params)

        vega_bsm = self.bs_calc.vega(forward, strike, sigma).pv

        dsigma_drho = self._jacobian_sabr(F, forward.T,
            np.array([strike.K]),
            params.array(),
            params.beta
        )[1][0]

        return Rega(
            vega_bsm * dsigma_drho
        )
    
    def vanilla_rega(self, spot: Spot, forward_yield: ForwardYield, vanilla: Vanilla, params: SABRParams) -> Rega:
        forward = Forward(spot, forward_yield, vanilla.time_to_maturity())
        return Rega(vanilla.N*\
            self.rega(forward, vanilla.strike(), params).pv
        )    
        
    def regas(self, forward: Forward, strikes: Strikes, params: SABRParams) -> Regas:
        F = forward.forward_rate().fv
        sigmas = self.implied_vols(forward, strikes, params).data
        Ks = strikes.data
        n = len(sigmas)
       
        vega_bsm = np.zeros_like(sigmas)
        for i in range(n):
            K = Ks[i]
            sigma = ImpliedVol(sigmas[i])
            vega_bsm[i] = self.bs_calc.vega(forward, Strike(K), sigma).pv

        dsigma_drho = self._jacobian_sabr(F, forward.T,
            Ks,
            params.array(),
            params.beta
        )[1] 
        return Regas(
            vega_bsm * dsigma_drho
        )
    
    def vanillas_regas(self, spot: Spot, forward_yield: ForwardYield, vanillas: Vanillas, params: SABRParams) -> Regas:
        forward = Forward(spot, forward_yield, vanillas.time_to_maturity())
        return Regas(vanillas.Ns*\
            self.regas(forward, vanillas.strikes(), params).data
        )    
    
    def sega(self, forward: Forward, strike: Strike, params: SABRParams) -> Sega:
        F = forward.forward_rate().fv
        sigma = self.implied_vol(forward, strike, params)

        vega_bsm = self.bs_calc.vega(forward, strike, sigma).pv

        dsigma_dv = self._jacobian_sabr(F, forward.T,
            np.array([strike.K]),
            params.array(),
            params.beta
        )[2][0]

        return Sega(
            vega_bsm * dsigma_dv
        )

    def vanilla_sega(self, spot: Spot, forward_yield: ForwardYield, vanilla: Vanilla, params: SABRParams) -> Sega:
        forward = Forward(spot, forward_yield, vanilla.time_to_maturity())
        return Sega(vanilla.N*\
            self.sega(forward, vanilla.strike(), params).pv
        ) 
        
    def segas(self, forward: Forward, strikes: Strikes, params: SABRParams) -> Segas:
        F = forward.forward_rate().fv
        sigmas = self.implied_vols(forward, strikes, params).data
        Ks = strikes.data
        n = len(sigmas)
       
        vega_bsm = np.zeros_like(sigmas)
        for i in range(n):
            K = Ks[i]
            sigma = ImpliedVol(sigmas[i])
            vega_bsm[i] = self.bs_calc.vega(forward, Strike(K), sigma).pv

        dsigma_dv = self._jacobian_sabr(F, forward.T,
            Ks,
            params.array(),
            params.beta
        )[2] 
        return Segas(
            vega_bsm * dsigma_dv
        )

    def vanillas_segas(self, spot: Spot, forward_yield: ForwardYield, vanillas: Vanillas, params: SABRParams) -> Segas:
        forward = Forward(spot, forward_yield, vanillas.time_to_maturity())
        return Segas(vanillas.Ns*\
            self.segas(forward, vanillas.strikes(), params).data
        )    
     
    def volga(self, forward: Forward, strike: Strike, params: SABRParams) -> Volga:
        F = forward.forward_rate().fv
        sigma = self.implied_vol(forward, strike, params)

        vega_bsm = self.bs_calc.vega(forward, strike, sigma).pv
        volga_bsm = self.bs_calc.volga(forward, strike, sigma).pv
                
        dsigma_df = self._dsigma_df(F, forward.T, strike.K, params)
        d2_sigma_dalpha2 = self._d2_sigma_dalpha2(F, forward.T, strike.K, params)
        d2_sigma_dalpha_df = self._d2_sigma_dalpha_df(F, forward.T, strike.K, params)

        dsigma_dalpha = self._jacobian_sabr(F, forward.T,
            np.array([strike.K]),
            params.array(),
            params.beta
        )[0][0]

        if params.v > 0 : 
            res = volga_bsm * (
                dsigma_dalpha + dsigma_df * params.rho * F**params.beta / params.v
            ) ** 2 + vega_bsm * (d2_sigma_dalpha2 + d2_sigma_dalpha_df * params.rho * F**params.beta / params.v)
        else:
            res = volga_bsm * (dsigma_dalpha) ** 2 + vega_bsm * d2_sigma_dalpha2

        return Volga(res)

    def vanilla_volga(self, spot: Spot, forward_yield: ForwardYield, vanilla: Vanilla, params: SABRParams) -> Volga:
        forward = Forward(spot, forward_yield, vanilla.time_to_maturity())
        return Volga(vanilla.N*\
            self.volga(forward, vanilla.strike(), params).pv
        ) 
     
    def volgas(self, forward: Forward, strikes: Strikes, params: SABRParams) -> Volgas:
        F = forward.forward_rate().fv
        sigmas = self.implied_vols(forward, strikes, params).data
        Ks = strikes.data
        n = len(sigmas)

        vega_bsm = np.zeros_like(sigmas)
        volga_bsm = np.zeros_like(sigmas)

        dsigma_df = np.zeros_like(sigmas)
        d2_sigma_dalpha2 = np.zeros_like(sigmas)
        d2_sigma_dalpha_df = np.zeros_like(sigmas)


        for i in range(n):
            K = Ks[i]
            sigma = ImpliedVol(sigmas[i])
            vega_bsm[i] = self.bs_calc.vega(forward, Strike(K), sigma).pv
            volga_bsm[i] = self.bs_calc.volga(forward, Strike(K), sigma).pv

            dsigma_df[i] = self._dsigma_df(F, forward.T, K, params)
            d2_sigma_dalpha2[i] = self._d2_sigma_dalpha2(F, forward.T, K, params)
            d2_sigma_dalpha_df[i] = self._d2_sigma_dalpha_df(F, forward.T, K, params)

        dsigma_dalpha = self._jacobian_sabr(F, forward.T,
            Ks,
            params.array(),
            params.beta
        )[0]

        if params.v > 0 : 
            res = volga_bsm * (
                dsigma_dalpha + dsigma_df * params.rho * F**params.beta / params.v
            ) ** 2 + vega_bsm * (d2_sigma_dalpha2 + d2_sigma_dalpha_df * params.rho * F**params.beta / params.v)
        else:
            res = volga_bsm * (dsigma_dalpha) ** 2 + vega_bsm * d2_sigma_dalpha2

        return Volgas(res)

    def vanillas_volgas(self, spot: Spot, forward_yield: ForwardYield, vanillas: Vanilla, params: SABRParams) -> Volgas:
        forward = Forward(spot, forward_yield, vanillas.time_to_maturity())
        return Volgas(vanillas.Ns*\
            self.volgas(forward, vanillas.strikes(), params).data
        )
    
     
    def vanna(self, forward: Forward, strike: Strike, params: SABRParams) -> Vanna:
        F = forward.forward_rate().fv
        D = forward.numeraire().pv
        sigma = self.implied_vol(forward, strike, params)

        vega_bsm = self.bs_calc.vega(forward, strike, sigma).pv
        vanna_bsm = self.bs_calc.vanna(forward, strike, sigma).pv
        volga_bsm = self.bs_calc.volga(forward, strike, sigma).pv
                
        dsigma_df = self._dsigma_df(F, forward.T, strike.K, params)
        d2_sigma_df2 = self._d2_sigma_df2(F, forward.T, strike.K, params)
        d2_sigma_dalpha_df = self._d2_sigma_dalpha_df(F, forward.T, strike.K, params)

        dsigma_dalpha = self._jacobian_sabr(F, forward.T,
            np.array([strike.K]),
            params.array(),
            params.beta
        )[0][0]


        if params.v > 0 : 
            res = (
                vanna_bsm + (volga_bsm/D) * (dsigma_df + dsigma_dalpha * params.rho * params.v / F**params.beta)
                ) * (dsigma_dalpha + dsigma_df * params.rho * F**params.beta / params.v) + (vega_bsm/D) * (
                    d2_sigma_dalpha_df
                    + d2_sigma_df2 * params.rho * F**params.beta / params.v
                    + dsigma_df * params.beta * params.rho * F ** (params.beta - 1) / params.v
                )
        else:
            res = (
                vanna_bsm + (volga_bsm/D) * dsigma_df 
                ) * dsigma_dalpha + (vega_bsm/D) * d2_sigma_dalpha_df

        return Vanna(res)
    
    def vanilla_vanna(self, spot: Spot, forward_yield: ForwardYield, vanilla: Vanilla, params: SABRParams) -> Vanna:
        forward = Forward(spot, forward_yield, vanilla.time_to_maturity())
        return Vanna(vanilla.N*\
            self.vanna(forward, vanilla.strike(), params).pv
        )    
     
    def vannas(self, forward: Forward, strikes: Strikes, params: SABRParams) -> Vannas:
        F = forward.forward_rate().fv
        D = forward.numeraire().pv
        sigmas = self.implied_vols(forward, strikes, params).data
        Ks = strikes.data
        n = len(sigmas)

        vega_bsm = np.zeros_like(sigmas)
        vanna_bsm = np.zeros_like(sigmas)
        volga_bsm = np.zeros_like(sigmas)

        dsigma_df = np.zeros_like(sigmas)
        d2_sigma_df2 = np.zeros_like(sigmas)
        d2_sigma_dalpha_df = np.zeros_like(sigmas)

        for i in range(n):
            K = Ks[i]
            sigma = ImpliedVol(sigmas[i])
            vega_bsm[i] = self.bs_calc.vega(forward, Strike(K), sigma).pv
            vanna_bsm[i] = self.bs_calc.vanna(forward, Strike(K), sigma).pv
            volga_bsm[i] = self.bs_calc.volga(forward, Strike(K), sigma).pv

            dsigma_df[i] = self._dsigma_df(F, forward.T, K, params)
            d2_sigma_df2[i] = self._d2_sigma_df2(F, forward.T, K, params)
            d2_sigma_dalpha_df[i] = self._d2_sigma_dalpha_df(F, forward.T, K, params)

        dsigma_dalpha = self._jacobian_sabr(F, forward.T,
            Ks,
            params.array(),
            params.beta
        )[0]

        if params.v > 0 : 
            res = (
                vanna_bsm + (volga_bsm/D) * (dsigma_df + dsigma_dalpha * params.rho * params.v / F**params.beta)
                ) * (dsigma_dalpha + dsigma_df * params.rho * F**params.beta / params.v) + (vega_bsm/D) * (
                    d2_sigma_dalpha_df
                    + d2_sigma_df2 * params.rho * F**params.beta / params.v
                    + dsigma_df * params.beta * params.rho * F ** (params.beta - 1) / params.v
                )
        else:
            res = (
                vanna_bsm + (volga_bsm/D) * dsigma_df 
                ) * dsigma_dalpha + (vega_bsm/D) * d2_sigma_dalpha_df

        return Vannas(res)
    
    def vanillas_vannas(self, spot: Spot, forward_yield: ForwardYield, vanillas: Vanillas, params: SABRParams) -> Vannas:
        forward = Forward(spot, forward_yield, vanillas.time_to_maturity())
        return Vannas(vanillas.Ns*\
            self.vannas(forward, vanillas.strikes(), params).data
        )
    
    def blip_vega(self, forward: Forward, strike: Strike, params: SABRParams, calibration_weights: CalibrationWeights) -> Vega:
        option_type = OptionType(forward.forward_rate().fv >= strike.K)
        premium = self.premium(forward, strike, option_type, params).pv
        delta_space = self.delta_space(forward, params)

        blipped_chain = delta_space.blip_ATM().to_chain_space()
        blipped_params = self.calibrate(blipped_chain, params.backbone(), calibration_weights)
        blipped_premium = self.premium(forward, strike, option_type, blipped_params).pv

        return Vega((blipped_premium - premium) / delta_space.atm_blip)
    
    def vanilla_blip_vega(self, spot: Spot, forward_yield: ForwardYield, vanilla: Vanilla, params: SABRParams, calibration_weights: CalibrationWeights) -> Vega:
        forward = Forward(spot, forward_yield, vanilla.time_to_maturity())
        print(vanilla.strike().K)
        return Vega(vanilla.N*\
            self.blip_vega(forward, vanilla.strike(), params, calibration_weights).pv
        )    
    
    def blip_vegas(self, forward: Forward, strikes: Strikes, params: SABRParams, calibration_weights: CalibrationWeights) -> Vegas:
        premiums = self.premiums(forward,  strikes, params).data
        delta_space = self.delta_space(forward, params)
  
        blipped_chain = delta_space.blip_ATM().to_chain_space()
        blipped_params = self.calibrate(blipped_chain, params.backbone(), calibration_weights)
        blipped_premiums = self.premiums(forward, strikes, blipped_params).data
    
        return Vegas((blipped_premiums - premiums) / delta_space.atm_blip)
    
    def vanillas_blip_vegas(self, spot: Spot, forward_yield: ForwardYield, vanillas: Vanillas, params: SABRParams, calibration_weights: CalibrationWeights) -> Vegas:
        forward = Forward(spot, forward_yield, vanillas.time_to_maturity())
        return Vegas(vanillas.Ns*\
            self.blip_vegas(forward, vanillas.strikes(), params, calibration_weights).data
        )
    
    def blip_rega(self, forward: Forward, strike: Strike, params: SABRParams, calibration_weights: CalibrationWeights) -> Rega:
        option_type = OptionType(forward.forward_rate().fv >= strike.K)
        premium = self.premium(forward,  strike, option_type, params).pv
        delta_space = self.delta_space(forward, params)

        blipped_chain = delta_space.blip_25RR().blip_10RR().to_chain_space()
        blipped_params = self.calibrate(blipped_chain, params.backbone(), calibration_weights)
        blipped_premium = self.premium(forward, strike, option_type, blipped_params).pv

        return Rega((blipped_premium - premium) / delta_space.rr25_blip)
    
    def vanilla_blip_rega(self, spot: Spot, forward_yield: ForwardYield, vanilla: Vanilla, params: SABRParams, calibration_weights: CalibrationWeights) -> Rega:
        forward = Forward(spot, forward_yield, vanilla.time_to_maturity())
        return Rega(vanilla.N*\
            self.blip_rega(forward, vanilla.strike(), params, calibration_weights).pv
        )     
    
    def blip_regas(self, forward: Forward, strikes: Strikes, params: SABRParams, calibration_weights: CalibrationWeights) -> Regas:
        premiums = self.premiums(forward,  strikes, params).data
        delta_space = self.delta_space(forward, params)

        blipped_chain = delta_space.blip_25RR().blip_10RR().to_chain_space()
        blipped_params = self.calibrate(blipped_chain, params.backbone(), calibration_weights)
        blipped_premiums = self.premiums(forward, strikes, blipped_params).data

        return Regas((blipped_premiums - premiums) / delta_space.rr25_blip) 
    
    def vanillas_blip_regas(self, spot: Spot, forward_yield: ForwardYield, vanillas: Vanillas, params: SABRParams, calibration_weights: CalibrationWeights) -> Regas:
        forward = Forward(spot, forward_yield, vanillas.time_to_maturity())
        return Regas(vanillas.Ns*\
            self.blip_regas(forward, vanillas.strikes(), params, calibration_weights).data
        )
    
    def blip_sega(self, forward: Forward, strike: Strike, params: SABRParams, calibration_weights: CalibrationWeights) -> Sega:
        option_type = OptionType(forward.forward_rate().fv >= strike.K)
        premium = self.premium(forward,  strike, option_type, params).pv
        delta_space = self.delta_space(forward, params)

        blipped_chain = delta_space.blip_25BF().blip_10BF().to_chain_space()
        blipped_params = self.calibrate(blipped_chain, params.backbone(), calibration_weights)
        blipped_premium = self.premium(forward, strike, option_type, blipped_params).pv

        return Sega((blipped_premium - premium) / delta_space.bf25_blip) 
    
    def vanilla_blip_sega(self, spot: Spot, forward_yield: ForwardYield, vanilla: Vanilla, params: SABRParams, calibration_weights: CalibrationWeights) -> Sega:
        forward = Forward(spot, forward_yield, vanilla.time_to_maturity())
        return Sega(vanilla.N*\
            self.blip_sega(forward, vanilla.strike(), params, calibration_weights).pv
        )     
    
    def blip_segas(self, forward: Forward, strikes: Strikes, params: SABRParams, calibration_weights: CalibrationWeights) -> Segas:
        premiums = self.premiums(forward,  strikes, params).data
        delta_space = self.delta_space(forward, params)
        blipped_chain = delta_space.blip_25BF().blip_10BF().to_chain_space()
        blipped_params = self.calibrate(blipped_chain, params.backbone(), calibration_weights)
        blipped_premiums = self.premiums(forward, strikes, blipped_params).data

        return Segas((blipped_premiums - premiums) / delta_space.bf25_blip) 
    
    def vanillas_blip_segas(self, spot: Spot, forward_yield: ForwardYield, vanillas: Vanillas, params: SABRParams, calibration_weights: CalibrationWeights) -> Segas:
        forward = Forward(spot, forward_yield, vanillas.time_to_maturity())
        return Segas(vanillas.Ns*\
            self.blip_segas(forward, vanillas.strikes(), params, calibration_weights).data
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
                if beta == 1.0:
                    I_B_0 = alpha
                else:
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
    ) -> Tuple[nb.float64[:]]:
        
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
                if beta == 1.0:
                    I_B = alpha
                    dI_B_0_dalpha = 1.
                    dI_B_0_dv = 0.0
                    dI_B_0_drho = 0.0
                else:
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
            if beta == 1.0:
                I_B_0 = alpha
                dI_B_0_dK = 0
            else:
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
    
    
    def _dsigma_df(self, F: nb.float64, T: nb.float64, K: nb.float64, params: SABRParams) -> nb.float64:
        alpha, beta, v, rho = params.alpha, params.beta, params.v, params.rho
        x = np.log(F / K)
        if x == 0.0:
            dI_B_0_dF = 0.0
            I_B_0 = K ** (beta - 1) * alpha
        elif v == 0.0:
            if beta == 1.0:
                I_B_0 = alpha
                dI_B_0_dF = 0
            else:
                I_B_0 = alpha * (1 - beta) * x / (-(K ** (1 - beta)) + F ** (1 - beta))
                dI_B_0_dF = (
                    alpha
                    * (beta - 1)
                    * (
                        F * (K ** (1 - beta) - F ** (1 - beta))
                        - F ** (2 - beta) * (beta - 1) * x
                    )
                    / (F**2 * (K ** (1 - beta) - F ** (1 - beta)) ** 2)
                )
        else:
            if beta == 1.0:
                z = v * x / alpha
                sqrt = np.sqrt(1 - 2 * rho * z + z**2)
                epsilon = np.log((-sqrt + rho - z) / (rho - 1))
                dI_B_0_dF = (
                    v * (alpha * sqrt * epsilon - v * x) / (alpha * F * sqrt * epsilon**2)
                )
            elif beta < 1.0:
                z = v * (F ** (1 - beta) - K ** (1 - beta)) / (alpha * (1 - beta))
                sqrt = np.sqrt(1 - 2 * rho * z + z**2)
                epsilon = np.log((-sqrt + rho - z) / (rho - 1))
                dI_B_0_dF = (
                    v
                    * (
                        alpha * F * (-rho + z + sqrt) * sqrt * epsilon
                        + F ** (2 - beta) * v * x * (rho - z - sqrt)
                    )
                    / (alpha * F**2 * (-rho + z + sqrt) * sqrt * epsilon**2)
                )
            I_B_0 = v * x / (epsilon)

        I_H_1 = (
            alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 / 24
            + alpha * beta * rho * v * (K * F) ** (beta / 2 + -1 / 2) / 4
            + v**2 * (2 - 3 * rho**2) / 24
        )
        dI_H_1_dF = alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 * (beta - 1) / (
            24 * F
        ) + alpha * beta * rho * v * (K * F) ** (beta / 2 - 1 / 2) * (beta / 2 - 1 / 2) / (
            4 * F
        )

        dsigma_df = dI_B_0_dF * (1 + I_H_1 * T) + dI_H_1_dF * I_B_0 * T
        return dsigma_df
    
    
    def _d2_sigma_dalpha_df(self, F: nb.float64, T: nb.float64, K: nb.float64, params: SABRParams) -> nb.float64:
        x = np.log(F / K)
        alpha, beta, v, rho = params.alpha, params.beta, params.v, params.rho
        if x == 0.0:
            d2I_B_0_dalpha_df = 0.0
            dI_B_0_dalpha = K ** (beta - 1)
            I_B_0 = K ** (beta - 1) * alpha
            dI_B_0_dF = 0.0
        elif v == 0.0:
            if beta == 1.0:
                I_B_0 = alpha
                dI_B_0_dalpha = 1.0
                dI_B_0_dF = 0.0
                d2I_B_0_dalpha_df = 0.0                
            else:
                d2I_B_0_dalpha_df = -(F ** (1 - beta)) * x * (1 - beta) ** 2 / (
                    F * (-(K ** (1 - beta)) + F ** (1 - beta)) ** 2
                ) + (1 - beta) / (F * (-(K ** (1 - beta)) + F ** (1 - beta)))
                dI_B_0_dalpha = (beta - 1) * x / (K ** (1 - beta) - F ** (1 - beta))
                I_B_0 = alpha * (1 - beta) * x / (-(K ** (1 - beta)) + F ** (1 - beta))
                dI_B_0_dF = (
                    alpha
                    * (beta - 1)
                    * (
                        F * (K ** (1 - beta) - F ** (1 - beta))
                        - F ** (2 - beta) * (beta - 1) * x
                    )
                    / (F**2 * (K ** (1 - beta) - F ** (1 - beta)) ** 2)
                )
        else:
            if beta == 1.0:
                z = v * x / alpha
                sqrt = np.sqrt(1 - 2 * rho * z + z**2)
                epsilon = np.log((-sqrt + rho - z) / (rho - 1))
                d2I_B_0_dalpha_df = (
                    v
                    * (
                        sqrt**2 * alpha * z * epsilon
                        + sqrt**2 * v * x * epsilon
                        - 2 * sqrt * v * x * z
                        + rho * v * x * z * epsilon
                        - v * x * z**2 * epsilon
                    )
                    / (sqrt**3 * alpha**2 * F * epsilon**3)
                )
                dI_B_0_dalpha = v * x * z / (alpha * sqrt * epsilon**2)
                dI_B_0_dF = (
                    v * (alpha * sqrt * epsilon - v * x) / (alpha * F * sqrt * epsilon**2)
                )
            elif beta < 1.0:
                z = v * (F ** (1 - beta) - K ** (1 - beta)) / (alpha * (1 - beta))
                sqrt = np.sqrt(1 - 2 * rho * z + z**2)
                epsilon = np.log((-sqrt + rho - z) / (rho - 1))
                d2I_B_0_dalpha_df = (
                    v
                    * (
                        sqrt**2 * alpha * F**beta * z * epsilon
                        + sqrt**2 * F * v * x * epsilon
                        - 2 * sqrt * F * v * x * z
                        + F * rho * v * x * z * epsilon
                        - F * v * x * z**2 * epsilon
                    )
                    / (sqrt**3 * alpha**2 * F * F**beta * epsilon**3)
                )
                dI_B_0_dalpha = v * x * z / (alpha * sqrt * epsilon**2)
                dI_B_0_dF = (
                    v
                    * (
                        alpha * F * (-rho + z + sqrt) * sqrt * epsilon
                        + F ** (2 - beta) * v * x * (rho - z - sqrt)
                    )
                    / (alpha * F**2 * (-rho + z + sqrt) * sqrt * epsilon**2)
                )
            I_B_0 = v * x / (epsilon)

        I_H_1 = (
            alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 / 24
            + alpha * beta * rho * v * (K * F) ** (beta / 2 + -1 / 2) / 4
            + v**2 * (2 - 3 * rho**2) / 24
        )
        dI_H_1_dF = alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 * (beta - 1) / (
            24 * F
        ) + alpha * beta * rho * v * (K * F) ** (beta / 2 - 1 / 2) * (beta / 2 - 1 / 2) / (
            4 * F
        )
        dI_H_1_dalpha = (
            alpha * (K * F) ** (beta - 1) * (1 - beta) ** 2 / 12
            + beta * rho * v * (K * F) ** (beta / 2 - 1 / 2) / 4
        )
        d2I_H_1_dalpha_df = alpha * (K * F) ** (beta - 1) * (1 - beta) ** 2 * (beta - 1) / (
            12 * F
        ) + beta * rho * v * (K * F) ** (beta / 2 - 1 / 2) * (beta / 2 - 1 / 2) / (4 * F)

        d2_sigma_dalpha_df2 = (
            d2I_B_0_dalpha_df * (1 + I_H_1 * T)
            + dI_B_0_dalpha * dI_H_1_dF * T
            + d2I_H_1_dalpha_df * I_B_0 * T
            + dI_H_1_dalpha * dI_B_0_dF * T
        )
        return d2_sigma_dalpha_df2

    
    def _d2_sigma_df2(self, F: nb.float64, T: nb.float64, K: nb.float64, params: SABRParams) -> nb.float64:
        alpha, beta, v, rho = params.alpha, params.beta, params.v, params.rho
        x = np.log(F / K)
        if x == 0.0:
            I_B_0 = K ** (beta - 1) * alpha
            d2I_B_0_d2f = 0.0
            dI_B_0_dF = 0.0
        elif v == 0.0:
            if beta == 1.0:
                I_B_0 = alpha 
                d2I_B_0_d2f = 0.0
                dI_B_0_dF = 0.0
            else:
                I_B_0 = alpha * (1 - beta) * x / (-(K ** (1 - beta)) + F ** (1 - beta))
                d2I_B_0_d2f = (
                    alpha
                    * (beta - 1)
                    * (
                        -(F**4) * (K ** (1 - beta) - F ** (1 - beta)) ** 2
                        + F ** (5 - beta)
                        * (K ** (1 - beta) - F ** (1 - beta))
                        * (beta - 1)
                        * (x * (beta - 1) + x - 2)
                        + 2 * F ** (6 - 2 * beta) * x * (beta - 1) ** 2
                    )
                    / (F**6 * (K ** (1 - beta) - F ** (1 - beta)) ** 3)
                )
                dI_B_0_dF = (
                    alpha
                    * (beta - 1)
                    * (
                        F * (K ** (1 - beta) - F ** (1 - beta))
                        - F ** (2 - beta) * (beta - 1) * x
                    )
                    / (F**2 * (K ** (1 - beta) - F ** (1 - beta)) ** 2)
                )
        else:
            if beta == 1.0:
                z = v * x / alpha
                sqrt = np.sqrt(1 - 2 * rho * z + z**2)
                epsilon = np.log((-sqrt + rho - z) / (rho - 1))
                d2I_B_0_d2f = (
                    v
                    * (
                        -(sqrt**3) * alpha**2 * epsilon**2
                        + sqrt**2 * alpha * v * x * epsilon
                        - 2 * sqrt**2 * alpha * v * epsilon
                        + 2 * sqrt * v**2 * x
                        - rho * v**2 * x * epsilon
                        + v**2 * x * z * epsilon
                    )
                    / (sqrt**3 * alpha**2 * F**2 * epsilon**3)
                )
                dI_B_0_dF = (
                    v * (alpha * sqrt * epsilon - v * x) / (alpha * F * sqrt * epsilon**2)
                )
            elif beta < 1.0:
                z = v * (F ** (1 - beta) - K ** (1 - beta)) / (alpha * (1 - beta))
                sqrt = np.sqrt(1 - 2 * rho * z + z**2)
                epsilon = np.log((-sqrt + rho - z) / (rho - 1))
                d2I_B_0_d2f = (
                    v
                    * (
                        -(sqrt**3) * alpha**2 * F * (sqrt - rho + z) * epsilon**2
                        - 2
                        * sqrt**2
                        * alpha
                        * F ** (2 - beta)
                        * v
                        * (sqrt - rho + z)
                        * epsilon
                        + sqrt
                        * F ** (3 - 2 * beta)
                        * v**2
                        * x
                        * (sqrt - rho + z)
                        * epsilon
                        + 2 * sqrt * F ** (3 - 2 * beta) * v**2 * x * (sqrt - rho + z)
                        + F
                        * v
                        * x
                        * (
                            sqrt**3 * alpha * beta * F ** (1 - beta)
                            + sqrt**2
                            * (
                                alpha * F ** (1 - beta) * (-rho * (beta - 1) - rho + z)
                                - v
                                * (
                                    F ** (1 - beta) * (-(K ** (1 - beta)) + F ** (1 - beta))
                                    + F ** (2 - 2 * beta)
                                )
                            )
                            + F ** (2 - 2 * beta) * v * (rho - z) ** 2
                        )
                        * epsilon
                    )
                    / (sqrt**3 * alpha**2 * F**3 * (sqrt - rho + z) * epsilon**3)
                )
                dI_B_0_dF = (
                    v
                    * (sqrt * alpha * F**beta * epsilon - F * v * x)
                    / (sqrt * alpha * F * F**beta * epsilon**2)
                )
            I_B_0 = v * x / (epsilon)

        I_H_1 = (
            alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 / 24
            + alpha * beta * rho * v * (K * F) ** (beta / 2 + -1 / 2) / 4
            + v**2 * (2 - 3 * rho**2) / 24
        )

        dI_H_1_dF = alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 * (beta - 1) / (
            24 * F
        ) + alpha * beta * rho * v * (K * F) ** (beta / 2 - 1 / 2) * (beta / 2 - 1 / 2) / (
            4 * F
        )
        d2I_H_1_d2f = (
            alpha**2
            * (K * F) ** (beta - 1)
            * (1 - beta) ** 2
            * (beta - 1) ** 2
            / (24 * F**2)
            - alpha**2
            * (K * F) ** (beta - 1)
            * (1 - beta) ** 2
            * (beta - 1)
            / (24 * F**2)
            + alpha
            * beta
            * rho
            * v
            * (K * F) ** (beta / 2 + -1 / 2)
            * (beta / 2 - 1 / 2) ** 2
            / (4 * F**2)
            - alpha
            * beta
            * rho
            * v
            * (K * F) ** (beta / 2 - 1 / 2)
            * (beta / 2 - 1 / 2)
            / (4 * F**2)
        )
        d2_sigma_df2 = d2I_B_0_d2f + T * (
            d2I_B_0_d2f * I_H_1 + d2I_H_1_d2f * I_B_0 + 2 * dI_B_0_dF * dI_H_1_dF
        )
        return d2_sigma_df2

    
    def _d2_sigma_dalpha2(self, F: nb.float64, T: nb.float64, K: nb.float64, params: SABRParams) -> nb.float64:
        alpha, beta, v, rho = params.alpha, params.beta, params.v, params.rho
        x = np.log(F / K)
        if x == 0.0:
            I_B_0 = K ** (beta - 1) * alpha
            dI_B_0_dalpha = K ** (beta - 1)
            d2I_B_0_dalpha2 = 0.0
        elif v == 0.0:
            if beta == 1.0:
                I_B_0 = alpha 
                dI_B_0_dalpha = 1.0
                d2I_B_0_dalpha2 = 0.0
            else:
                I_B_0 = alpha * (1 - beta) * x / (-(K ** (1 - beta)) + F ** (1 - beta))
                dI_B_0_dalpha = (beta - 1) * x / (K ** (1 - beta) - F ** (1 - beta))
                d2I_B_0_dalpha2 = 0.0
        else:
            if beta == 1.0:
                z = v * x / alpha
                sqrt = np.sqrt(1 - 2 * rho * z + z**2)
                epsilon = np.log((-sqrt + rho - z) / (rho - 1))
                dI_B_0_dalpha = v * x * z / (alpha * sqrt * epsilon**2)
                d2I_B_0_dalpha2 = (
                    v
                    * x
                    * z
                    * (
                        -2 * sqrt**2 * epsilon
                        + 2 * sqrt * z
                        - rho * z * epsilon
                        + z**2 * epsilon
                    )
                    / (sqrt**3 * alpha**2 * epsilon**3)
                )
            elif beta < 1.0:
                z = v * (F ** (1 - beta) - K ** (1 - beta)) / (alpha * (1 - beta))
                sqrt = np.sqrt(1 - 2 * rho * z + z**2)
                epsilon = np.log((-sqrt + rho - z) / (rho - 1))
                dI_B_0_dalpha = v * x * z / (alpha * sqrt * epsilon**2)
                d2I_B_0_dalpha2 = (
                    v
                    * x
                    * z
                    * (
                        -2 * sqrt**2 * epsilon
                        + 2 * sqrt * z
                        - rho * z * epsilon
                        + z**2 * epsilon
                    )
                    / (sqrt**3 * alpha**2 * epsilon**3)
                )
            I_B_0 = v * x / (epsilon)

        I_H_1 = (
            alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 / 24
            + alpha * beta * rho * v * (K * F) ** (beta / 2 + -1 / 2) / 4
            + v**2 * (2 - 3 * rho**2) / 24
        )
        dI_H_1_dalpha = (
            alpha * (K * F) ** (beta - 1) * (1 - beta) ** 2 / 12
            + beta * rho * v * (K * F) ** (beta / 2 - 1 / 2) / 4
        )
        d2I_H_1_d2alpha = (K * F) ** (beta - 1) * (1 - beta) ** 2 / 12

        d2_sigma_dalpha2 = d2I_B_0_dalpha2 + T * (
            d2I_B_0_dalpha2 * I_H_1
            + d2I_H_1_d2alpha * I_B_0
            + 2 * dI_B_0_dalpha * dI_H_1_dalpha
        )
        return d2_sigma_dalpha2
    