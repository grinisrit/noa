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

    def scale_alpha(self, s: nb.float64) -> 'SABRParams':
        return SABRParams(Volatility(s*self.alpha), Correlation(self.rho), VolOfVol(self.v), Backbone(self.beta))

    def scale_rho(self, s: nb.float64) -> 'SABRParams':
        return SABRParams(Volatility(self.alpha), Correlation(s*self.rho), VolOfVol(self.v), Backbone(self.beta))

    def scale_v(self, s: nb.float64) -> 'SABRParams':
        return SABRParams(Volatility(self.alpha), Correlation(self.rho), VolOfVol(s*self.v), Backbone(self.beta))
            
        
@nb.experimental.jitclass([
    ("beta", nb.float64),
    ("cached_params", nb.float64[:]),
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

    def calibrate(self, chain: VolSmileChainSpace, backbone: Backbone, calibration_weights: CalibrationWeights) -> Tuple[SABRParams, CalibrationError]:
        strikes = chain.Ks
        w = calibration_weights.w
        
        if not strikes.shape == w.shape:
            raise ValueError('Inconsistent data between strikes and calibration weights')

        n_points = len(strikes)
        PARAMS_TO_CALIBRATE = 3
        if not n_points - PARAMS_TO_CALIBRATE >= 0:
            raise ValueError('Need at least 3 points to calibrate SABR model')
     
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
            result_error = F / n_points

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
                    result_error = F / n_points
                else:
                    i -= 1
                    mu *= nu2
                    continue
                result_x = x
                
            return result_x, result_error

        calc_params, calibration_error \
            = levenberg_marquardt(get_residuals, clip_params, self.cached_params)
        
        return SABRParams(
            Volatility(calc_params[0]),
            Correlation(calc_params[1]),
            VolOfVol(calc_params[2]),
            backbone
        ), CalibrationError(calibration_error)
  
    def implied_vol(self, forward: Forward, strike: Strike, params: SABRParams) -> ImpliedVol:
        return ImpliedVol(
            self._vol_sabr(forward.forward_rate().fv, forward.T, np.array([strike.K]), params.array(), params.beta)[0]
        )
    
    def implied_vols(self, forward: Forward, strikes: Strikes, params: SABRParams) -> ImpliedVols:
        return ImpliedVols(
            self._vol_sabr(forward.forward_rate().fv, forward.T, strikes.data,  params.array(), params.beta)
        )
    
    def premium(self, forward: Forward, vanilla: Vanilla, params: SABRParams) -> Premium:
        assert forward.T == vanilla.T
        sigma =\
            self._vol_sabr(forward.forward_rate().fv, forward.T, np.array([vanilla.K]), params.array(), params.beta)[0]
        return self.bs_calc.premium(forward, vanilla, ImpliedVol(sigma))
   
    def premiums(self, forward: Forward, vanillas: SingleMaturityVanillas, params: SABRParams) -> Premiums:
        assert forward.T == vanillas.T
        f = forward.forward_rate().fv
        Ks = vanillas.Ks
        sigmas =\
            self._vol_sabr(f, forward.T, Ks,  params.array(), params.beta)
        res = np.zeros_like(sigmas)
        n = len(sigmas)
        for i in range(n):
            K = Ks[i]
            is_call = vanillas.is_call[i]
            res[i] = self.bs_calc._premium(forward, Strike(K), OptionType(is_call), ImpliedVol(sigmas[i]))
        return Premiums(vanillas.Ns * res)
        
    def strike_from_delta(self, forward: Forward, delta: Delta, params: SABRParams) -> Strike:
        F = forward.forward_rate().fv
        K_l = self.strike_lower*F
        K_r = self.strike_upper*F
        T = forward.T
        option_type = OptionType(delta.pv >= 0.)

        def g(K):
            iv = self.implied_vol(forward, Strike(K), params)  
            return self.bs_calc._delta(forward, Strike(K), option_type, iv) - delta.pv

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
    
    def smile_to_delta_space(self, chain: VolSmileChainSpace, backbone: Backbone) -> VolSmileDeltaSpace:
        params,_ = self.calibrate(chain, backbone, CalibrationWeights(np.ones_like(chain.Ks)))
        return self.delta_space(chain.forward(), params)
    
    def surface_to_delta_space(self, surface_chain: VolSurfaceChainSpace, backbone: Backbone) -> VolSurfaceDeltaSpace:
        times_to_maturities = surface_chain.times_to_maturities()
        Ts = times_to_maturities.data

        atm = np.zeros_like(Ts)
        rr25 = np.zeros_like(Ts)
        bf25 = np.zeros_like(Ts)
        rr10 = np.zeros_like(Ts)
        bf10 = np.zeros_like(Ts)

        for i in nb.prange(len(Ts)):
            T = Ts[i]
            smile_chain = surface_chain.get_vol_smile(TimeToMaturity(T))
            smile_delta = self.smile_to_delta_space(smile_chain, backbone)
            atm[i] = smile_delta.ATM
            rr25[i] = smile_delta.RR25
            bf25[i] = smile_delta.BF25
            rr10[i] = smile_delta.RR10
            bf10[i] = smile_delta.BF10

        return VolSurfaceDeltaSpace(
            surface_chain.forward_curve(),
            Straddles(ImpliedVols(atm), times_to_maturities),
            RiskReversals(Delta(0.25), VolatilityQuotes(rr25), times_to_maturities),
            Butterflies(Delta(0.25), VolatilityQuotes(bf25), times_to_maturities),
            RiskReversals(Delta(0.1), VolatilityQuotes(rr10), times_to_maturities),
            Butterflies(Delta(0.1), VolatilityQuotes(bf10), times_to_maturities)
        )

    def surface_grid_ivs(self, surface: VolSurfaceDeltaSpace, strikes: Strikes, times_to_maturity: TimesToMaturity, backbone: Backbone) -> ImpliedVols:
        Ks = strikes.data
        Ts = times_to_maturity.data
        n = len(Ts)
        m = len(Ks)
        ivs = np.zeros(n*m)
        
        for i in nb.prange(n):
            smile = surface.get_vol_smile(TimeToMaturity(Ts[i]))
            smile_params,_ = self.calibrate(smile.to_chain_space(), backbone, CalibrationWeights(np.ones(5)))
            ivs[i*m: (i+1)*m] = self.implied_vols(smile.forward(), strikes, smile_params).data

        return ImpliedVols(ivs)

    def sticky_delta(self, forward: Forward, vanilla: Vanilla, params: SABRParams, sticky_strike: StickyStrike) -> Delta:
        assert forward.T == vanilla.T
        
        F = forward.forward_rate().fv
        D = forward.numeraire().pv
        strike = vanilla.strike()
       
        sigma = self.implied_vol(forward, strike, params)

        delta_bsm = self.bs_calc._delta(forward, strike, vanilla.option_type(), sigma)
        vega_bsm = self.bs_calc._vega(forward, strike, sigma)
        dsigma_df = self._dsigma_df(F, forward.T, strike.K, params)

        dsigma_dalpha = self._jacobian_sabr(F, forward.T,
            np.array([strike.K]),
            params.array(),
            params.beta
        )[0][0] if not sticky_strike.v else 0.

        res_delta = delta_bsm + (1/D) * vega_bsm * (dsigma_df + dsigma_dalpha * params.rho * params.v / F**params.beta)
        return Delta(vanilla.N*res_delta)
      
    def sticky_deltas(self, forward: Forward, vanillas: SingleMaturityVanillas, params: SABRParams, sticky_strike: StickyStrike) -> Delta:
        assert forward.T == vanillas.T
        F = forward.forward_rate().fv
        D = forward.numeraire().pv
        strikes = vanillas.strikes()
        Ks = strikes.data
        sigmas = self.implied_vols(forward, strikes, params).data
        n = len(sigmas)
        delta_bsm = np.zeros_like(sigmas)
        vega_bsm = np.zeros_like(sigmas)
        dsigma_df = np.zeros_like(sigmas)

        for i in range(n):
            K = Ks[i]
            sigma = sigmas[i]
            is_call = vanillas.is_call[i]
            delta_bsm[i] = self.bs_calc._delta(forward, Strike(K), OptionType(is_call), ImpliedVol(sigma))
            vega_bsm[i] = self.bs_calc._vega(forward, Strike(K), ImpliedVol(sigma))

            dsigma_df[i] = self._dsigma_df(F, forward.T, K, params)

        if sticky_strike.v:
            res_delta = delta_bsm + (1/D) * vega_bsm * dsigma_df

        else:
            dsigma_dalpha = self._jacobian_sabr(F, forward.T,
                Ks,
                params.array(),
                params.beta
            )[0]
            res_delta = delta_bsm + (1/D) * vega_bsm * (dsigma_df + dsigma_dalpha * params.rho * params.v / F**params.beta)

        return Deltas(vanillas.Ns*res_delta)  
     
    def sticky_gamma(self, forward: Forward, vanilla: Vanilla, params: SABRParams, sticky_strike: StickyStrike) -> Gamma:
        assert forward.T == vanilla.T
        
        F = forward.forward_rate().fv
        D = forward.numeraire().pv
        strike = vanilla.strike()
        sigma = self.implied_vol(forward, strike, params)

        gamma_bsm = self.bs_calc._gamma(forward, strike, sigma)
        vega_bsm = self.bs_calc._vega(forward, strike, sigma)
        vanna_bsm = self.bs_calc._vanna(forward, strike, sigma)
        volga_bsm = self.bs_calc._volga(forward, strike, sigma)
                
        dsigma_df = self._dsigma_df(F, forward.T, strike.K, params)
        d2_sigma_df2 = self._d2_sigma_df2(F, forward.T, strike.K, params)

        d2_sigma_dalpha_df = self._d2_sigma_dalpha_df(F, forward.T, strike.K, params) if not sticky_strike.v else 0.

        dsigma_dalpha = 0. if sticky_strike else self._jacobian_sabr(F, forward.T,
            np.array([strike.K]),
            params.array(),
            params.beta
        )[0][0]

        sticky_component = dsigma_df + dsigma_dalpha * params.rho * params.v / F**params.beta
        res_gamma = gamma_bsm +\
            (2/D) * vanna_bsm * sticky_component +\
            (volga_bsm / (D**2)) * sticky_component**2 +\
            (vega_bsm / (D**2)) * (
                d2_sigma_df2
                + d2_sigma_dalpha_df * params.rho * params.v / F**params.beta
                - dsigma_dalpha * params.beta * params.rho * params.v / F ** (params.beta + 1)
            )
        return Gamma(vanilla.N*res_gamma)
    
    def sticky_gammas(self, forward: Forward, vanillas: SingleMaturityVanillas, params: SABRParams, sticky_strike: StickyStrike) -> Gammas:
        assert forward.T == vanillas.T
        F = forward.forward_rate().fv
        D = forward.numeraire().pv
        sigmas = self.implied_vols(forward, vanillas.strikes(), params).data
        Ks = vanillas.Ks
        n = len(sigmas)

        gamma_bsm = np.zeros_like(sigmas)
        vega_bsm = np.zeros_like(sigmas)
        vanna_bsm = np.zeros_like(sigmas)
        volga_bsm = np.zeros_like(sigmas)

        dsigma_df = np.zeros_like(sigmas)
        d2_sigma_df2 = np.zeros_like(sigmas)

        for i in range(n):
            K = Ks[i]
            sigma = ImpliedVol(sigmas[i])
            gamma_bsm[i] = self.bs_calc._gamma(forward, Strike(K), sigma)
            vega_bsm[i] = self.bs_calc._vega(forward, Strike(K), sigma)
            vanna_bsm[i] = self.bs_calc._vanna(forward, Strike(K), sigma)
            volga_bsm[i] = self.bs_calc._volga(forward, Strike(K), sigma)

            dsigma_df[i] = self._dsigma_df(F, forward.T, K, params)
            d2_sigma_df2[i] = self._d2_sigma_df2(F, forward.T, K, params)

        if not sticky_strike.v:
            res_gammas = gamma_bsm + (2/D) * vanna_bsm * dsigma_df + (volga_bsm / (D**2)) * dsigma_df**2 + \
                (vega_bsm / (D**2)) * d2_sigma_df2

        else:
            d2_sigma_dalpha_df = np.zeros_like(sigmas)
            for i in range(n):
                K = Ks[i]
                d2_sigma_dalpha_df[i] = self._d2_sigma_dalpha_df(F, forward.T, K, params)

            dsigma_dalpha = self._jacobian_sabr(F, forward.T,
                Ks,
                params.array(),
                params.beta
            )[0]
            sticky_component = dsigma_df + dsigma_dalpha * params.rho * params.v / F**params.beta

            res_gammas = gamma_bsm + \
                (2/D) * vanna_bsm * sticky_component + \
                (volga_bsm / (D**2)) * sticky_component**2 + \
                (vega_bsm / (D**2)) * (
                    d2_sigma_df2
                    + d2_sigma_dalpha_df * params.rho * params.v / F**params.beta
                    - dsigma_dalpha * params.beta * params.rho * params.v / F ** (params.beta + 1)
                )
            
        return Gammas(vanillas.Ns*res_gammas)
        
    def sticky_vega(self, forward: Forward, vanilla: Vanilla, params: SABRParams, sticky_strike: StickyStrike) -> Vega:
        assert forward.T == vanilla.T
        
        F = forward.forward_rate().fv
        strike = vanilla.strike()
        sigma = self.implied_vol(forward, strike, params)

        vega_bsm = self.bs_calc._vega(forward, strike, sigma)

        dsigma_df = self._dsigma_df(F, forward.T, strike.K, params)

        dsigma_dalpha = self._jacobian_sabr(F, forward.T,
            np.array([strike.K]),
            params.array(),
            params.beta
        )[0][0] if not sticky_strike.v else 0.

        if params.v > 0 and not sticky_strike.v: 
            res = vega_bsm * (dsigma_dalpha + dsigma_df * params.rho * F**params.beta / params.v)
        else:
            res = vega_bsm * dsigma_dalpha

        return Vega(vanilla.N * res)
    
    def sticky_vegas(self, forward: Forward, vanillas: SingleMaturityVanillas, params: SABRParams, sticky_strike: StickyStrike) -> Vegas:
        assert forward.T == vanillas.T
        
        F = forward.forward_rate().fv
        sigmas = self.implied_vols(forward, vanillas.strikes(), params).data
        Ks = vanillas.Ks
        n = len(sigmas)
       
        vega_bsm = np.zeros_like(sigmas)

        for i in range(n):
            K = Ks[i]
            sigma = ImpliedVol(sigmas[i])
            vega_bsm[i] = self.bs_calc._vega(forward, Strike(K), sigma)

        dsigma_dalpha = self._jacobian_sabr(F, forward.T,
            Ks,
            params.array(),
            params.beta
        )[0] 

        if params.v > 0 and not sticky_strike.v: 
            dsigma_df = np.zeros_like(sigmas)
            for i in range(n):
                dsigma_df[i] = self._dsigma_df(F, forward.T, Ks[i], params)
            res = vega_bsm * (dsigma_dalpha + dsigma_df * params.rho * F**params.beta / params.v)
        else:
            res = vega_bsm * dsigma_dalpha

        return Vegas(vanillas.Ns * res)
   
    def rega_rho(self, forward: Forward, vanilla: Vanilla, params: SABRParams) -> Rega:
        assert forward.T == vanilla.T
        F = forward.forward_rate().fv
        strike = vanilla.strike()
        sigma = self.implied_vol(forward, strike, params)

        vega_bsm = self.bs_calc._vega(forward, strike, sigma)

        dsigma_drho = self._jacobian_sabr(F, forward.T,
            np.array([strike.K]),
            params.array(),
            params.beta
        )[1][0]

        return Rega(
            vanilla.N * vega_bsm * dsigma_drho
        )
              
    def regas_rho(self, forward: Forward, vanillas: SingleMaturityVanillas, params: SABRParams) -> Regas:
        assert forward.T == vanillas.T
        F = forward.forward_rate().fv
        sigmas = self.implied_vols(forward, vanillas.strikes(), params).data
        Ks = vanillas.Ks
        n = len(sigmas)
       
        vega_bsm = np.zeros_like(sigmas)
        for i in range(n):
            K = Ks[i]
            sigma = ImpliedVol(sigmas[i])
            vega_bsm[i] = self.bs_calc._vega(forward, Strike(K), sigma)

        dsigma_drho = self._jacobian_sabr(F, forward.T,
            Ks,
            params.array(),
            params.beta
        )[1] 
        return Regas(
            vanillas.Ns * vega_bsm * dsigma_drho
        )    
    
    def sega_volvol(self, forward: Forward, vanilla: Vanilla, params: SABRParams) -> Sega:
        assert forward.T == vanilla.T
        F = forward.forward_rate().fv
        strike = vanilla.strike()
        sigma = self.implied_vol(forward, strike, params)

        vega_bsm = self.bs_calc._vega(forward, strike, sigma)

        dsigma_dv = self._jacobian_sabr(F, forward.T,
            np.array([strike.K]),
            params.array(),
            params.beta
        )[2][0]

        return Sega(
            vanilla.N * vega_bsm * dsigma_dv
        )
        
    def segas_volvol(self, forward: Forward, vanillas: SingleMaturityVanillas, params: SABRParams) -> Segas:
        assert forward.T == vanillas.T
        F = forward.forward_rate().fv
        sigmas = self.implied_vols(forward, vanillas.strikes(), params).data
        Ks = vanillas.Ks
        n = len(sigmas)
       
        vega_bsm = np.zeros_like(sigmas)
        for i in range(n):
            K = Ks[i]
            sigma = ImpliedVol(sigmas[i])
            vega_bsm[i] = self.bs_calc._vega(forward, Strike(K), sigma)

        dsigma_dv = self._jacobian_sabr(F, forward.T,
            Ks,
            params.array(),
            params.beta
        )[2] 
        return Segas(
            vanillas.Ns * vega_bsm * dsigma_dv
        )
     
    def sticky_volga(self, forward: Forward, vanilla: Vanilla, params: SABRParams, sticky_strike: StickyStrike) -> Volga:
        assert forward.T == vanilla.T
        
        F = forward.forward_rate().fv
        strike = vanilla.strike()
        sigma = self.implied_vol(forward, strike, params)

        vega_bsm = self.bs_calc._vega(forward, strike, sigma)
        volga_bsm = self.bs_calc._volga(forward, strike, sigma)

        d2_sigma_dalpha2 = self._d2_sigma_dalpha2(F, forward.T, strike.K, params)
        
                
        dsigma_df = self._dsigma_df(F, forward.T, strike.K, params) if not sticky_strike.v else 0.
        d2_sigma_dalpha_df = self._d2_sigma_dalpha_df(F, forward.T, strike.K, params) if not sticky_strike.v else 0.

        dsigma_dalpha = self._jacobian_sabr(F, forward.T,
            np.array([strike.K]),
            params.array(),
            params.beta
        )[0][0]

        if params.v > 0 and not sticky_strike.v : 
            res = volga_bsm * (
                dsigma_dalpha + dsigma_df * params.rho * F**params.beta / params.v
            ) ** 2 + vega_bsm * (d2_sigma_dalpha2 + d2_sigma_dalpha_df * params.rho * F**params.beta / params.v)
        else:
            res = volga_bsm * (dsigma_dalpha) ** 2 + vega_bsm * d2_sigma_dalpha2

        return Volga(vanilla.N*res)
     
    def sticky_volgas(self, forward: Forward, vanillas: SingleMaturityVanillas, params: SABRParams, sticky_strike: StickyStrike) -> Volgas:
        assert forward.T == vanillas.T
        
        F = forward.forward_rate().fv
        sigmas = self.implied_vols(forward, vanillas.strikes(), params).data
        Ks = vanillas.Ks
        n = len(sigmas)

        vega_bsm = np.zeros_like(sigmas)
        volga_bsm = np.zeros_like(sigmas)

        d2_sigma_dalpha2 = np.zeros_like(sigmas)

        for i in range(n):
            K = Ks[i]
            sigma = ImpliedVol(sigmas[i])
            vega_bsm[i] = self.bs_calc._vega(forward, Strike(K), sigma)
            volga_bsm[i] = self.bs_calc._volga(forward, Strike(K), sigma)
            d2_sigma_dalpha2[i] = self._d2_sigma_dalpha2(F, forward.T, K, params)
           

        dsigma_dalpha = self._jacobian_sabr(F, forward.T,
            Ks,
            params.array(),
            params.beta
        )[0]

        if params.v > 0 and not sticky_strike.v: 
            dsigma_df = np.zeros_like(sigmas)
            d2_sigma_dalpha_df = np.zeros_like(sigmas)
            for i in range(n):
                dsigma_df[i] = self._dsigma_df(F, forward.T, Ks[i], params)
                d2_sigma_dalpha_df[i] = self._d2_sigma_dalpha_df(F, forward.T, Ks[i], params)

            res = volga_bsm * (
                dsigma_dalpha + dsigma_df * params.rho * F**params.beta / params.v
            ) ** 2 + vega_bsm * (d2_sigma_dalpha2 + d2_sigma_dalpha_df * params.rho * F**params.beta / params.v)
        else:
            res = volga_bsm * (dsigma_dalpha) ** 2 + vega_bsm * d2_sigma_dalpha2

        return Volgas(vanillas.Ns * res)
    
    def sticky_vanna(self, forward: Forward, vanilla: Vanilla, params: SABRParams, sticky_strike: StickyStrike) -> Vanna:
        assert forward.T == vanilla.T
        
        F = forward.forward_rate().fv
        D = forward.numeraire().pv
        strike = vanilla.strike()
        sigma = self.implied_vol(forward, strike, params)

        vega_bsm = self.bs_calc._vega(forward, strike, sigma)
        vanna_bsm = self.bs_calc._vanna(forward, strike, sigma)
        volga_bsm = self.bs_calc._volga(forward, strike, sigma)
                
        dsigma_df = self._dsigma_df(F, forward.T, strike.K, params)
        d2_sigma_dalpha_df = self._d2_sigma_dalpha_df(F, forward.T, strike.K, params)

        dsigma_dalpha = self._jacobian_sabr(F, forward.T,
            np.array([strike.K]),
            params.array(),
            params.beta
        )[0][0]


        if params.v > 0 and not sticky_strike.v: 
            d2_sigma_df2 = self._d2_sigma_df2(F, forward.T, strike.K, params)
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

        return Vanna(vanilla.N * res)  
     
    def sticky_vannas(self, forward: Forward, vanillas: SingleMaturityVanillas, params: SABRParams, sticky_strike: StickyStrike) -> Vannas:
        assert forward.T == vanillas.T
        
        F = forward.forward_rate().fv
        D = forward.numeraire().pv
        strikes = vanillas.strikes()
        sigmas = self.implied_vols(forward, strikes, params).data
        Ks = vanillas.Ks
        n = len(sigmas)

        vega_bsm = np.zeros_like(sigmas)
        vanna_bsm = np.zeros_like(sigmas)
        volga_bsm = np.zeros_like(sigmas)

        dsigma_df = np.zeros_like(sigmas)
        d2_sigma_dalpha_df = np.zeros_like(sigmas)

        for i in range(n):
            K = Ks[i]
            sigma = ImpliedVol(sigmas[i])
            vega_bsm[i] = self.bs_calc._vega(forward, Strike(K), sigma)
            vanna_bsm[i] = self.bs_calc._vanna(forward, Strike(K), sigma)
            volga_bsm[i] = self.bs_calc._volga(forward, Strike(K), sigma)

            dsigma_df[i] = self._dsigma_df(F, forward.T, K, params)
            d2_sigma_dalpha_df[i] = self._d2_sigma_dalpha_df(F, forward.T, K, params)

        dsigma_dalpha = self._jacobian_sabr(F, forward.T,
            Ks,
            params.array(),
            params.beta
        )[0]

        if params.v > 0 and not sticky_strike.v: 
            d2_sigma_df2 = np.zeros_like(sigmas)
            for i in range(n):
                d2_sigma_df2[i] = self._d2_sigma_df2(F, forward.T, Ks[i], params)   

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

        return Vannas(vanillas.Ns * res)
    
    def blip_vega(self, forward: Forward, vanilla: Vanilla, params: SABRParams) -> Vega:
        premium = self.premium(forward, vanilla, params).pv
        delta_space = self.delta_space(forward, params)

        blipped_chain = delta_space.blip_ATM().to_chain_space()
        blipped_params,_ = self.calibrate(blipped_chain, params.backbone(), CalibrationWeights(np.ones(5)))
        blipped_premium = self.premium(forward, vanilla, blipped_params).pv

        return Vega((blipped_premium - premium) / delta_space.atm_blip)
    
    def blip_vegas(self, forward: Forward, vanillas: SingleMaturityVanillas, params: SABRParams) -> Vegas:
        premiums = self.premiums(forward, vanillas, params).data
        delta_space = self.delta_space(forward, params)
  
        blipped_chain = delta_space.blip_ATM().to_chain_space()
        blipped_params,_ = self.calibrate(blipped_chain, params.backbone(), CalibrationWeights(np.ones(5)))
        blipped_premiums = self.premiums(forward, vanillas, blipped_params).data
    
        return Vegas((blipped_premiums - premiums) / delta_space.atm_blip)
    
    def blip_rega(self, forward: Forward, vanilla: Vanilla, params: SABRParams) -> Rega:
        premium = self.premium(forward, vanilla, params).pv
        delta_space = self.delta_space(forward, params)

        blipped_chain = delta_space.blip_25RR().blip_10RR().to_chain_space()
        blipped_params,_ = self.calibrate(blipped_chain, params.backbone(), CalibrationWeights(np.ones(5)))
        blipped_premium = self.premium(forward, vanilla, blipped_params).pv

        return Rega((blipped_premium - premium) / delta_space.rr25_blip)
     
    def blip_regas(self, forward: Forward, vanillas: SingleMaturityVanillas, params: SABRParams) -> Regas:
        premiums = self.premiums(forward, vanillas, params).data
        delta_space = self.delta_space(forward, params)

        blipped_chain = delta_space.blip_25RR().blip_10RR().to_chain_space()
        blipped_params,_ = self.calibrate(blipped_chain, params.backbone(), CalibrationWeights(np.ones(5)))
        blipped_premiums = self.premiums(forward, vanillas, blipped_params).data

        return Regas((blipped_premiums - premiums) / delta_space.rr25_blip) 

    def blip_sega(self, forward: Forward, vanilla: Vanilla, params: SABRParams) -> Sega:
        premium = self.premium(forward, vanilla, params).pv
        delta_space = self.delta_space(forward, params)

        blipped_chain = delta_space.blip_25BF().blip_10BF().to_chain_space()
        blipped_params,_ = self.calibrate(blipped_chain, params.backbone(), CalibrationWeights(np.ones(5)))
        blipped_premium = self.premium(forward, vanilla, blipped_params).pv

        return Sega((blipped_premium - premium) / delta_space.bf25_blip)     
    
    def blip_segas(self, forward: Forward, vanillas: SingleMaturityVanillas, params: SABRParams) -> Segas:
        premiums = self.premiums(forward,  vanillas, params).data
        delta_space = self.delta_space(forward, params)
        blipped_chain = delta_space.blip_25BF().blip_10BF().to_chain_space()
        blipped_params,_ = self.calibrate(blipped_chain, params.backbone(), CalibrationWeights(np.ones(5)))
        blipped_premiums = self.premiums(forward, vanillas, blipped_params).data

        return Segas((blipped_premiums - premiums) / delta_space.bf25_blip) 
     
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
    