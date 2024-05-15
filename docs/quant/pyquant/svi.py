import numba as nb
import numpy as np

from .black_scholes import *
from .common import *
from .vol_surface import *


@nb.experimental.jitclass([("a", nb.float64)])
class A:
    def __init__(self, a: nb.float64):
        self.a = a


@nb.experimental.jitclass([("b", nb.float64)])
class B:
    def __init__(self, b: nb.float64):
        if not (b >= 0):
            raise ValueError("B not >= 0")
        self.b = b


@nb.experimental.jitclass([("rho", nb.float64)])
class Rho:
    def __init__(self, rho: nb.float64):
        if not (rho >= -1 and rho <= 1):
            raise ValueError("Rho not from [-1, 1]")
        self.rho = rho


@nb.experimental.jitclass([("m", nb.float64)])
class M:
    def __init__(self, m: nb.float64):
        self.m = m


@nb.experimental.jitclass([("sigma", nb.float64)])
class Sigma:
    def __init__(self, sigma: nb.float64):
        if not (sigma > 0):
            raise ValueError("Sigma not > 0")
        self.sigma = sigma


@nb.experimental.jitclass(
    [
        ("a", nb.float64),
        ("b", nb.float64),
        ("rho", nb.float64),
        ("m", nb.float64),
        ("sigma", nb.float64),
    ]
)
class SVIRawParams:
    def __init__(self, a: A, b: B, rho: Rho, m: M, sigma: Sigma):
        self.a = a.a
        self.b = b.b
        self.rho = rho.rho
        self.m = m.m
        self.sigma = sigma.sigma

    def array(self) -> nb.float64[:]:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])
    
    def scale_a(self, s: nb.float64) -> nb.float64:
        return SVIRawParams(A(s*self.a), B(self.b), Rho(self.rho), M(self.m), Sigma(self.sigma))

    def scale_b(self, s: nb.float64) -> nb.float64:
        return SVIRawParams(A(self.a), B(s*self.b), Rho(self.rho), M(self.m), Sigma(self.sigma))

    def scale_rho(self, s: nb.float64) -> nb.float64:
        return SVIRawParams(A(self.a), B(self.b), Rho(s*self.rho), M(self.m), Sigma(self.sigma))
    
    def scale_m(self, s: nb.float64) -> nb.float64:
        return SVIRawParams(A(self.a), B(self.b), Rho(self.rho), M(s*self.m), Sigma(self.sigma))
    
    def scale_sigma(self, s: nb.float64) -> nb.float64:
        return SVIRawParams(A(self.a), B(self.b), Rho(self.rho), M(self.m), Sigma(s*self.sigma))


@nb.experimental.jitclass([("v", nb.float64)])
class V:
    def __init__(self, v: nb.float64):
        self.v = v


@nb.experimental.jitclass([("psi", nb.float64)])
class Psi:
    def __init__(self, psi: nb.float64):
        self.psi = psi


@nb.experimental.jitclass([("p", nb.float64)])
class P:
    def __init__(self, p: nb.float64):
        self.p = p


@nb.experimental.jitclass([("c", nb.float64)])
class C:
    def __init__(self, c: nb.float64):
        self.c = c


@nb.experimental.jitclass([("v_tilda", nb.float64)])
class V_tilda:
    def __init__(self, v_tilda: nb.float64):
        self.v_tilda = v_tilda


@nb.experimental.jitclass(
    [
        ("v", nb.float64),
        ("psi", nb.float64),
        ("p", nb.float64),
        ("c", nb.float64),
        ("v_tilda", nb.float64),
    ]
)
class SVIJumpWingParams:
    def __init__(self, v: V, psi: Psi, p: P, c: C, v_tilda: V_tilda):
        self.v = v.v
        self.psi = psi.psi
        self.p = p.p
        self.c = c.c
        self.v_tilda = v_tilda.v_tilda

    def array(self) -> nb.float64[:]:
        return np.array([self.v, self.psi, self.p, self.c, self.v_tilda])


@nb.experimental.jitclass([("pv", nb.float64)])
class AGreek:
    def __init__(self, a_greek: nb.float64):
        self.pv = a_greek


@nb.experimental.jitclass([("data", nb.float64[:])])
class AGreeks:
    def __init__(self, a_greeks: nb.float64[:]):
        self.data = a_greeks


@nb.experimental.jitclass([("pv", nb.float64)])
class BGreek:
    def __init__(self, b_greek: nb.float64):
        self.pv = b_greek


@nb.experimental.jitclass([("data", nb.float64[:])])
class BGreeks:
    def __init__(self, b_greeks: nb.float64[:]):
        self.data = b_greeks


@nb.experimental.jitclass([("pv", nb.float64)])
class RhoGreek:
    def __init__(self, rho_greek: nb.float64):
        self.pv = rho_greek


@nb.experimental.jitclass([("data", nb.float64[:])])
class RhoGreeks:
    def __init__(self, rho_greeks: nb.float64[:]):
        self.data = rho_greeks


@nb.experimental.jitclass([("pv", nb.float64)])
class MGreek:
    def __init__(self, m_greek: nb.float64):
        self.pv = m_greek


@nb.experimental.jitclass([("data", nb.float64[:])])
class MGreeks:
    def __init__(self, m_greeks: nb.float64[:]):
        self.data = m_greeks


@nb.experimental.jitclass([("pv", nb.float64)])
class SigmaGreek:
    def __init__(self, sigma_greek: nb.float64):
        self.pv = sigma_greek


@nb.experimental.jitclass([("data", nb.float64[:])])
class SigmaGreeks:
    def __init__(self, sigma_greeks: nb.float64[:]):
        self.data = sigma_greeks


@nb.experimental.jitclass([
    ("raw_cached_params", nb.float64[:]),
    ("jump_wing_cached_params", nb.float64[:]),
    ("num_iter", nb.int64),
    ("delta_num_iter", nb.int64),
    ("max_mu", nb.float64),
    ("min_mu", nb.float64),
    ("tol", nb.float64),
    ("strike_lower", nb.float64),
    ("strike_upper", nb.float64),
    ("delta_tol", nb.float64),
    ("delta_grad_eps", nb.float64)
])  
class SVICalc:
    bs_calc: BSCalc

    def __init__(self):
        self.raw_cached_params = np.array([0.0, 0.1, 0.0, 1.0, 0.1])
        self.jump_wing_cached_params = self.raw_cached_params
        self.num_iter = 10000
        self.max_mu = 1e4
        self.min_mu = 1e-6
        self.tol = 1e-12
        self.strike_lower = 0.1
        self.strike_upper = 10.0
        self.delta_tol = 1e-8
        self.delta_num_iter = 500
        self.delta_grad_eps = 1e-4
        self.bs_calc = BSCalc()

    def update_raw_cached_params(self, params: SVIRawParams):
        self.raw_cached_params = params.array()

    def update_jump_wing_cached_params(self, params: SVIJumpWingParams):
        self.jump_wing_cached_params = params.array()

    def calibrate(
        self, chain: VolSmileChainSpace, calibration_weights: CalibrationWeights, 
        update_cached_params: bool = False
    ) -> Tuple[SVIRawParams, CalibrationError]:
        strikes = chain.Ks
        w = calibration_weights.w

        if not strikes.shape == w.shape:
            raise ValueError('Inconsistent data between strikes and calibration weights')
        
        if not strikes.shape == w.shape:
            raise ValueError(
                "Inconsistent data between strikes and calibration weights"
            )

        n_points = len(strikes)
        PARAMS_TO_CALIBRATE = 5
        if not n_points >= 0:
            raise ValueError('Need at least 5 points to calibrate SVI model')

        weights = w / w.sum()
        forward = chain.f
        tot_vars = chain.T * (chain.sigmas ** 2)

        def clip_params(params):
            eps = 1e-4
            a, b, rho, m, sigma = params[0], params[1], params[2], params[3], params[4]
            b = np_clip(b, eps, 1000000.0)
            rho = np_clip(rho, -1.0 + eps, 1.0 - eps)
            sigma = np_clip(sigma, eps, 1000000.0)
            # TODO: a + b*sigma*sqrt(1 - rho^2) >= 0
            svi_params = np.array([a, b, rho, m, sigma])
            return svi_params

        def get_residuals(params):
            J = np.stack(
                self._jacobian_total_implied_var_svi_raw(
                    forward,
                    strikes,
                    params,
                )
            )
            svi_w = self._total_implied_var_svi(
                forward,
                strikes,
                params,
            )
            res = svi_w - tot_vars
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
                    mu = max(self.min_mu, mu/nu1)
                    result_error = F / n_points
                else:
                    i -= 1
                    mu = min(self.max_mu, mu*nu2)
                    continue
                result_x = x
            return result_x, result_error

        calc_params, calibration_error = levenberg_marquardt(
            get_residuals, clip_params, self.raw_cached_params
        )

        raw_params = SVIRawParams(
            A(calc_params[0]),
            B(calc_params[1]),
            Rho(calc_params[2]),
            M(calc_params[3]),
            Sigma(calc_params[4]),
        )
        if update_cached_params:
            self.raw_cached_params = calc_params

        return raw_params, CalibrationError(calibration_error)

    def get_jump_wing_params(
        self, params: SVIRawParams, time_to_maturity: TimeToMaturity
    ) -> SVIJumpWingParams:
        T = time_to_maturity.T
        a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
        v = (a + b * (-rho * m + np.sqrt(m**2 + sigma**2))) / T
        w = v * T
        psi = 1 / np.sqrt(w) * b / 2 * (-m / np.sqrt(m**2 + sigma**2) + rho)
        p = b * (1 - rho) / np.sqrt(w)
        c = b * (1 + rho) / np.sqrt(w)
        v_tilda = 1 / T * (a + b * sigma * np.sqrt(1 - rho**2))
        return SVIJumpWingParams(
            V(v),
            Psi(psi),
            P(p),
            C(c),
            V_tilda(v_tilda),
        )

    def implied_vol(
        self, forward: Forward, strike: Strike, params: SVIRawParams
    ) -> ImpliedVol:
        return ImpliedVol(
            self._vol_svi(
                forward.forward_rate().fv,
                forward.T,
                np.array([strike.K]),
                params.array(),
            )[0]
        )

    def implied_vols(
        self, forward: Forward, strikes: Strikes, params: SVIRawParams
    ) -> ImpliedVols:
        return ImpliedVols(
            self._vol_svi(
                forward.forward_rate().fv, forward.T, strikes.data, params.array()
            )
        )

    def premium(self, forward: Forward, vanilla: Vanilla, params: SVIRawParams) -> Premium:
        assert forward.T == vanilla.T
        sigma =\
            self._vol_svi(forward.forward_rate().fv, forward.T, np.array([vanilla.K]), params.array())[0]
        return self.bs_calc.premium(forward, vanilla, ImpliedVol(sigma))
   
    def premiums(self, forward: Forward, vanillas: SingleMaturityVanillas, params: SVIRawParams) -> Premiums:
        assert forward.T == vanillas.T
        f = forward.forward_rate().fv
        Ks = vanillas.Ks
        sigmas =\
            self._vol_svi(f, forward.T, Ks,  params.array())
        res = np.zeros_like(sigmas)
        n = len(sigmas)
        for i in range(n):
            K = Ks[i]
            is_call = vanillas.is_call[i]
            res[i] = self.bs_calc._premium(forward, Strike(K), OptionType(is_call), ImpliedVol(sigmas[i]))
        return Premiums(vanillas.Ns * res)

    def strike_from_delta(
        self, forward: Forward, delta: Delta, params: SVIRawParams
    ) -> Strike:
        F = forward.forward_rate().fv
        K_l = self.strike_lower * F
        K_r = self.strike_upper * F
        T = forward.T
        option_type = OptionType(delta.pv >= 0.0)

        def g(K):
            iv = self.implied_vol(forward, Strike(K), params)
            return self.bs_calc._delta(forward, Strike(K), option_type, iv) - delta.pv

        def g_prime(K):
            iv = self.implied_vol(forward, Strike(K), params)
            dsigma_dk = self._dsigma_dK(F, T, K, params)
            d1 = self.bs_calc._d1(forward, Strike(K), iv)
            sigma = iv.sigma
            return (
                np.exp(-(d1**2) / 2)
                / np.sqrt(T)
                * (
                    -1 / (K * sigma)
                    - dsigma_dk * np.log(F / K) / sigma**2
                    - forward.r * T * dsigma_dk / sigma**2
                    + T * dsigma_dk
                )
            )

        if g(K_l) * g(K_r) > 0.0:
            raise ValueError("No solution within strikes interval")

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
                    if g(K_l) * g(K) > 0:
                        K_l = K
                    else:
                        K_r = K
                    K = (K_l + K_r) / 2
            else:
                if g(K_l) * epsilon > 0:
                    K_l = K
                else:
                    K_r = K
                K = (K_l + K_r) / 2

            epsilon = g(K)
            grad = g_prime(K)

        return Strike(K)

    def delta_space(self, forward: Forward, params: SVIRawParams) -> VolSmileDeltaSpace:
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
            RiskReversal(
                Delta(0.25), VolatilityQuote(call25 - put25), TimeToMaturity(forward.T)
            ),
            Butterfly(
                Delta(0.25),
                VolatilityQuote(0.5 * (call25 + put25) - atm),
                TimeToMaturity(forward.T),
            ),
            RiskReversal(
                Delta(0.1), VolatilityQuote(call10 - put10), TimeToMaturity(forward.T)
            ),
            Butterfly(
                Delta(0.1),
                VolatilityQuote(0.5 * (call10 + put10) - atm),
                TimeToMaturity(forward.T),
            ),
        )

    def smile_to_delta_space(self, chain: VolSmileChainSpace) -> VolSmileDeltaSpace:
        params, _= self.calibrate(chain, CalibrationWeights(np.ones_like(chain.Ks)))
        return self.delta_space(chain.forward(), params)

    def surface_to_delta_space(self, surface_chain: VolSurfaceChainSpace) -> VolSurfaceDeltaSpace:
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
            smile_delta = self.smile_to_delta_space(smile_chain)
            atm[i] = smile_delta.ATM
            rr25[i] = smile_delta.RR25
            bf25[i] = smile_delta.BF25
            rr10[i] = smile_delta.RR10
            bf10[i] = smile_delta.BF10

        print(atm)

        return VolSurfaceDeltaSpace(
            surface_chain.forward_curve(),
            Straddles(ImpliedVols(atm), times_to_maturities),
            RiskReversals(Delta(0.25), VolatilityQuotes(rr25), times_to_maturities),
            Butterflies(Delta(0.25), VolatilityQuotes(bf25), times_to_maturities),
            RiskReversals(Delta(0.1), VolatilityQuotes(rr10), times_to_maturities),
            Butterflies(Delta(0.1), VolatilityQuotes(bf10), times_to_maturities)
        )

    def surface_grid_ivs(self, surface: VolSurfaceDeltaSpace, strikes: Strikes, times_to_maturity: TimesToMaturity) -> ImpliedVols:
        Ks = strikes.data
        Ts = times_to_maturity.data
        n = len(Ts)
        m = len(Ks)
        ivs = np.zeros(n*m)

        for i in range(n):
            smile = surface.get_vol_smile(TimeToMaturity(Ts[i]))
            smile_params, err = self.calibrate(smile.to_chain_space(), CalibrationWeights(np.ones(5)))
            ivs[i*m: (i+1)*m] = self.implied_vols(smile.forward(), strikes, smile_params).data

        return ImpliedVols(ivs)

    # ================ Greeks by ray and f, f^2(delta, gamma) ================
    def _delta(
        self,
        forward: Forward,
        strike: Strike,
        option_type: OptionType,
        params: SVIRawParams,
    ) -> nb.float64:
        F = forward.forward_rate().fv
        D = forward.numeraire().pv
        sigma = self.implied_vol(forward, strike, params)
        delta_bsm = self.bs_calc._delta(forward, strike, option_type, sigma)
        vega_bsm = self.bs_calc._vega(forward, strike, sigma)
        dsigma_df = self._dsigma_df(F, forward.T, strike.K, params)
        return delta_bsm + (1 / D) * vega_bsm * dsigma_df

    def delta(
        self,
        forward: Forward,
        vanilla: Vanilla,
        params: SVIRawParams
    ) -> Delta:
        assert forward.T == vanilla.T
        return Delta(vanilla.N * self._delta(forward, vanilla.strike(), vanilla.option_type(), params))

    def deltas(
        self,
        forward: Forward,
        vanillas: SingleMaturityVanillas,
        params: SVIRawParams,
    ) -> Deltas:
        assert forward.T == vanillas.T
        Ks = vanillas.Ks
        Ot = vanillas.is_call
        n = len(Ks)
        deltas = np.zeros_like(Ks)
        for i in range(n):
            deltas[i] = self._delta(forward, Strike(Ks[i]), OptionType(Ot[i]), params)

        return Deltas(vanillas.Ns * deltas)

    def _gamma(self, forward: Forward, strike: Strike, params: SVIRawParams) -> nb.float64:
        F = forward.forward_rate().fv
        D = forward.numeraire().pv
        sigma = self.implied_vol(forward, strike, params)
        vega_bsm = self.bs_calc._vega(forward, strike, sigma)
        vanna_bsm = self.bs_calc._vanna(forward, strike, sigma)
        dsigma_df = self._dsigma_df(F, forward.T, strike.K, params)
        d2_sigma_df2 = self._d2_sigma_df2(F, forward.T, strike.K, params)
        return (2 / D) * vanna_bsm * dsigma_df + vega_bsm * d2_sigma_df2 / (D**2)
    
    def gamma(self, forward: Forward, vanilla: Vanilla, params: SVIRawParams) -> Gamma:
        assert forward.T == vanilla.T
        return Gamma(vanilla.N * self._gamma(forward, vanilla.strike(), params))

    def gammas(
        self, forward: Forward, vanillas: SingleMaturityVanillas, params: SVIRawParams
    ) -> Gammas:
        assert forward.T == vanillas.T
        Ks = vanillas.Ks
        n = len(Ks)
        gammas = np.zeros_like(Ks)
        for i in range(n):
            gammas[i] = self._gamma(forward, Strike(Ks[i]), params)

        return Gammas(vanillas.Ns * gammas)

    def _a_greek(self, forward: Forward, strike: Strike, params: SVIRawParams) -> nb.float64:
        F = forward.forward_rate().fv
        sigma = self.implied_vol(forward, strike, params)
        vega_bsm = self.bs_calc._vega(forward, strike, sigma)
        dsigma_da = self._jacobian_vol_svi_raw(
            F, forward.T, np.array([strike.K]), params.array()
        )[0][0]
        return vega_bsm * dsigma_da
    
    def a_greek(self, forward: Forward, vanilla: Vanilla, params: SVIRawParams) -> AGreek:
        assert forward.T == vanilla.T
        return AGreek(vanilla.N * self._a_greek(forward, vanilla.strike(), params))

    def a_greeks(
        self, forward: Forward, vanillas: SingleMaturityVanillas, params: SVIRawParams
    ) -> AGreeks:
        assert forward.T == vanillas.T
        Ks = vanillas.Ks
        n = len(Ks)
        a_greeks = np.zeros_like(Ks)
        for i in range(n):
            a_greeks[i] = self._a_greek(forward, Strike(Ks[i]), params)

        return AGreeks(vanillas.Ns * a_greeks)

    def _b_greek(self, forward: Forward, strike: Strike, params: SVIRawParams) -> nb.float64:
        F = forward.forward_rate().fv
        sigma = self.implied_vol(forward, strike, params)
        vega_bsm = self.bs_calc._vega(forward, strike, sigma)
        dsigma_db = self._jacobian_vol_svi_raw(
            F, forward.T, np.array([strike.K]), params.array()
        )[1][0]
        return vega_bsm * dsigma_db
    
    def b_greek(self, forward: Forward, vanilla: Vanilla, params: SVIRawParams) -> BGreek:
        assert forward.T == vanilla.T
        return BGreek(vanilla.N * self._b_greek(forward, vanilla.strike(), params))

    def b_greeks(
        self, forward: Forward, vanillas: SingleMaturityVanillas, params: SVIRawParams
    ) -> BGreeks:
        assert forward.T == vanillas.T
        Ks = vanillas.Ks
        n = len(Ks)
        b_greeks = np.zeros_like(Ks)
        for i in range(n):
            b_greeks[i] = self._b_greek(forward, Strike(Ks[i]), params)

        return BGreeks(vanillas.Ns * b_greeks)

    def _rho_greek(
        self, forward: Forward, strike: Strike, params: SVIRawParams
    ) -> nb.float64:
        F = forward.forward_rate().fv
        sigma = self.implied_vol(forward, strike, params)
        vega_bsm = self.bs_calc._vega(forward, strike, sigma)
        dsigma_drho = self._jacobian_vol_svi_raw(
            F, forward.T, np.array([strike.K]), params.array()
        )[2][0]
        return vega_bsm * dsigma_drho
    
    def rho_greek(
        self, forward: Forward, vanilla: Vanilla, params: SVIRawParams) -> RhoGreek:
        assert forward.T == vanilla.T
        return RhoGreek(vanilla.N * self._rho_greek(forward, vanilla.strike(), params))

    def rho_greeks(
        self, forward: Forward, vanillas: SingleMaturityVanillas, params: SVIRawParams
    ) -> RhoGreeks:
        assert forward.T == vanillas.T
        Ks = vanillas.Ks
        n = len(Ks)
        rho_greeks = np.zeros_like(Ks)
        for i in range(n):
            rho_greeks[i] = self._rho_greek(forward, Strike(Ks[i]), params)

        return RhoGreeks(vanillas.Ns * rho_greeks)

    def _m_greek(self, forward: Forward, strike: Strike, params: SVIRawParams) -> nb.float64:
        F = forward.forward_rate().fv
        sigma = self.implied_vol(forward, strike, params)
        vega_bsm = self.bs_calc._vega(forward, strike, sigma)
        dsigma_dm = self._jacobian_vol_svi_raw(
            F, forward.T, np.array([strike.K]), params.array()
        )[3][0]
        return vega_bsm * dsigma_dm
    
    def m_greek(
        self, forward: Forward, vanilla: Vanilla, params: SVIRawParams) -> MGreek:
        assert forward.T == vanilla.T
        return MGreek(vanilla.N * self._m_greek(forward, vanilla.strike(), params))   

    def m_greeks(
        self, forward: Forward, vanillas: SingleMaturityVanillas, params: SVIRawParams
    ) -> MGreeks:
        assert forward.T == vanillas.T
        Ks = vanillas.Ks
        n = len(Ks)
        m_greeks = np.zeros_like(Ks)
        for i in range(n):
            m_greeks[i] = self._m_greek(forward, Strike(Ks[i]), params)

        return MGreeks(vanillas.Ns * m_greeks)

    def _sigma_greek(
        self, forward: Forward, strike: Strike, params: SVIRawParams
    ) -> nb.float64:
        F = forward.forward_rate().fv
        sigma = self.implied_vol(forward, strike, params)
        vega_bsm = self.bs_calc._vega(forward, strike, sigma)
        dsigma_dsigma = self._jacobian_vol_svi_raw(
            F, forward.T, np.array([strike.K]), params.array()
        )[4][0]
        return vega_bsm * dsigma_dsigma
    
    def sigma_greek(
        self, forward: Forward, vanilla: Vanilla, params: SVIRawParams) -> SigmaGreek:
        assert forward.T == vanilla.T
        return SigmaGreek(vanilla.N * self._sigma_greek(forward, vanilla.strike(), params))  
    
    def sigma_greeks(
        self, forward: Forward, vanillas: SingleMaturityVanillas, params: SVIRawParams
    ) -> SigmaGreeks:
        assert forward.T == vanillas.T
        Ks = vanillas.Ks
        n = len(Ks)
        sigma_greeks = np.zeros_like(Ks)
        for i in range(n):
            sigma_greeks[i] = self._sigma_greek(forward, Strike(Ks[i]), params)
        return SigmaGreeks(vanillas.Ns * sigma_greeks)

    # ================ Blip Greeks ================

    def blip_vega(self, forward: Forward, vanilla: Vanilla, params: SVIRawParams) -> Vega:
        premium = self.premium(forward, vanilla, params).pv
        delta_space = self.delta_space(forward, params)

        blipped_chain = delta_space.blip_ATM().to_chain_space()
        blipped_params,_ = self.calibrate(blipped_chain, CalibrationWeights(np.ones(5)))
        blipped_premium = self.premium(forward, vanilla, blipped_params).pv

        return Vega((blipped_premium - premium) / delta_space.atm_blip)
   
    
    def blip_vegas(self, forward: Forward, vanillas: SingleMaturityVanillas, params: SVIRawParams) -> Vegas:
        premiums = self.premiums(forward, vanillas, params).data
        delta_space = self.delta_space(forward, params)
  
        blipped_chain = delta_space.blip_ATM().to_chain_space()
        blipped_params,_ = self.calibrate(blipped_chain, CalibrationWeights(np.ones(5)))
        blipped_premiums = self.premiums(forward, vanillas, blipped_params).data
    
        return Vegas((blipped_premiums - premiums) / delta_space.atm_blip)
    
    
    def blip_rega(self, forward: Forward, vanilla: Vanilla, params: SVIRawParams) -> Rega:
        premium = self.premium(forward, vanilla, params).pv
        delta_space = self.delta_space(forward, params)

        blipped_chain = delta_space.blip_25RR().blip_10RR().to_chain_space()
        blipped_params,_ = self.calibrate(blipped_chain, CalibrationWeights(np.ones(5)))
        blipped_premium = self.premium(forward, vanilla, blipped_params).pv

        return Rega((blipped_premium - premium) / delta_space.rr25_blip)
     
    
    def blip_regas(self, forward: Forward, vanillas: SingleMaturityVanillas, params: SVIRawParams) -> Regas:
        premiums = self.premiums(forward, vanillas, params).data
        delta_space = self.delta_space(forward, params)

        blipped_chain = delta_space.blip_25RR().blip_10RR().to_chain_space()
        blipped_params,_ = self.calibrate(blipped_chain, CalibrationWeights(np.ones(5)))
        blipped_premiums = self.premiums(forward, vanillas, blipped_params).data

        return Regas((blipped_premiums - premiums) / delta_space.rr25_blip) 

    
    def blip_sega(self, forward: Forward, vanilla: Vanilla, params: SVIRawParams) -> Sega:
        premium = self.premium(forward, vanilla, params).pv
        delta_space = self.delta_space(forward, params)

        blipped_chain = delta_space.blip_25BF().blip_10BF().to_chain_space()
        blipped_params,_ = self.calibrate(blipped_chain, CalibrationWeights(np.ones(5)))
        blipped_premium = self.premium(forward, vanilla, blipped_params).pv

        return Sega((blipped_premium - premium) / delta_space.bf25_blip)     
    

    def blip_segas(self, forward: Forward, vanillas: SingleMaturityVanillas, params: SVIRawParams) -> Segas:
        premiums = self.premiums(forward,  vanillas, params).data
        delta_space = self.delta_space(forward, params)
        blipped_chain = delta_space.blip_25BF().blip_10BF().to_chain_space()
        blipped_params,_ = self.calibrate(blipped_chain, CalibrationWeights(np.ones(5)))
        blipped_premiums = self.premiums(forward, vanillas, blipped_params).data

        return Segas((blipped_premiums - premiums) / delta_space.bf25_blip) 
    
    # TODO: better to have function with single strike and loop for several strikes
    def _total_implied_var_svi(
        self,
        F: nb.float64,
        Ks: nb.float64[:],
        params: nb.float64[:],
    ) -> nb.float64[:]:
        a, b, rho, m, sigma = params[0], params[1], params[2], params[3], params[4]
        k = np.log(Ks / F)
        w = a + b * ( rho * (k - m) + np.sqrt((k - m)**2 + sigma**2) )
        return w

    def _vol_svi(
        self,
        F: nb.float64,
        T: nb.float64,
        Ks: nb.float64[:],
        params: nb.float64[:],
    ) -> nb.float64[:]:
        w = self._total_implied_var_svi(F=F, Ks=Ks, params=params)
        iv = np.sqrt(np.abs(w) / T)
        return iv

    def _jacobian_total_implied_var_svi_raw(
        self,
        F: nb.float64,
        Ks: nb.float64[:],
        params: nb.float64[:],
    ) -> tuple[nb.float64[:]]:
        b, rho, m, sigma = params[1], params[2], params[3], params[4]
        n = len(Ks)
        k = np.log(Ks / F)
        sqrt = np.sqrt(sigma**2 + (k - m) ** 2)
        dda = np.ones(n, dtype=np.float64)
        ddb = rho * (k - m) + sqrt
        ddrho = b * (k - m)
        ddm = b * (-rho + (m - k) / sqrt)
        ddsigma = b * sigma / sqrt
        return dda, ddb, ddrho, ddm, ddsigma

    def _jacobian_vol_svi_raw(
        self,
        F: nb.float64,
        T: nb.float64,
        Ks: nb.float64[:],
        params: nb.float64[:],
    ) -> tuple[nb.float64[:]]:
        # iv = sqrt(w/T)
        # div/da = (dw/da)/(2*sqrt(T*w))
        w = self._total_implied_var_svi(F=F, Ks=Ks, params=params)
        dda, ddb, ddrho, ddm, ddsigma = self._jacobian_total_implied_var_svi_raw(
            F=F, Ks=Ks, params=params
        )
        denominator = 2 * np.sqrt(T * w)
        return (
            dda / denominator,
            ddb / denominator,
            ddrho / denominator,
            ddm / denominator,
            ddsigma / denominator,
        )

    # ================ Derivatives of implied vol by raw params a, b, rho, m, sigma ================
    def _dsigma_df(
        self, F: nb.float64, T: nb.float64, K: nb.float64, params: SVIRawParams
    ) -> nb.float64:
        a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
        Ks = np.array([K])
        w = self._total_implied_var_svi(F=F, Ks=Ks, params=params.array())[0] #TODO: remove this
        denominator = 2 * np.sqrt(T * w)
        k = np.log(K / F)
        sqrt = np.sqrt(sigma**2 + (k - m) ** 2)
        ddf = b * (-rho / F - (k - m) / (F * sqrt))
        return ddf / denominator

    def _dsigma_dK(
        self, F: nb.float64, T: nb.float64, K: nb.float64, params: SVIRawParams
    ) -> nb.float64:
        # by strike to compute BBs, RRs
        a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
        Ks = np.array([K])
        w = self._total_implied_var_svi(F=F, Ks=Ks, params=params.array())[0] #TODO: remove this
        denominator = 2 * np.sqrt(T * w)
        k = np.log(K / F)
        sqrt = np.sqrt(sigma**2 + (k - m) ** 2)
        ddk = b * (rho / K + (k - m) / (K * sqrt))
        return (ddk / denominator)

    def _d2_sigma_df2(
        self, F: nb.float64, T: nb.float64, K: nb.float64, params: SVIRawParams
    ) -> nb.float64:
        a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
        Ks = np.array([K])
        w = self._total_implied_var_svi(F=F, Ks=Ks, params=params.array())[0] #TODO: remove this
        denominator = 2 * np.sqrt(T * w)
        k = np.log(K / F)
        sqrt = np.sqrt(sigma**2 + (k - m) ** 2)
        dddff = (
            rho / F**2
            - (k - m) ** 2 / (F**2 * sqrt**3)
            + (k - m) / (F**2 * sqrt)
            + 1 / (F**2 * sqrt)
        )
        dddff = dddff * b
        return dddff / denominator