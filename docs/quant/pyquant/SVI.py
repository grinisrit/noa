import numpy as np
import numba as nb

from .common import *
from .black_scholes import *
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


@nb.experimental.jitclass([("w", nb.float64[:])])
class CalibrationWeights:
    def __init__(self, w: nb.float64):
        if not np.all(w >= 0):
            raise ValueError("Weights must be non-negative")
        if not w.sum() > 0:
            raise ValueError("At least one weight must be non-trivial")
        self.w = w


class SVICalc:
    bs_calc: BSCalc

    def __init__(self):
        self.cached_params = np.array([1.0, 0.0, 0.0])
        self.calibration_error = 0.0
        self.num_iter = 50
        self.tol = 1e-8
        self.strike_lower = 0.1
        self.strike_upper = 10.0
        self.delta_tol = 1e-8
        self.delta_num_iter = 500
        self.delta_grad_eps = 1e-4
        self.bs_calc = BSCalc()

    def update_cached_params(self, params: SVIRawParams):
        self.cached_params = params.array()

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

    def premium(
        self,
        forward: Forward,
        strike: Strike,
        option_type: OptionType,
        params: SVIRawParams,
    ) -> Premium:
        sigma = self._vol_svi(
            forward.forward_rate().fv, forward.T, np.array([strike.K]), params.array()
        )[0]
        return self.bs_calc.premium(forward, strike, option_type, ImpliedVol(sigma))

    def premiums(
        self, forward: Forward, strikes: Strikes, params: SVIRawParams
    ) -> Premiums:
        f = forward.forward_rate().fv
        sigmas = self._vol_svi(f, forward.T, strikes.data, params.array())
        Ks = strikes.data
        res = np.zeros_like(sigmas)
        n = len(sigmas)
        for i in range(n):
            K = Ks[i]
            res[i] = self.bs_calc.premium(
                forward, Strike(K), OptionType(K >= f), ImpliedVol(sigmas[i])
            ).pv
        return Premiums(res)

    def vanilla_premium(
        self,
        spot: Spot,
        forward_yield: ForwardYield,
        vanilla: Vanilla,
        params: SVIRawParams,
    ) -> Premium:
        forward = Forward(spot, forward_yield, vanilla.time_to_maturity())
        return Premium(
            vanilla.N
            * self.premium(forward, vanilla.strike(), vanilla.option_type(), params).pv
        )

    def vanillas_premiums(
        self,
        spot: Spot,
        forward_yield: ForwardYield,
        vanillas: Vanillas,
        params: SVIRawParams,
    ) -> Premiums:
        forward = Forward(spot, forward_yield, vanillas.time_to_maturity())
        f = forward.forward_rate().fv
        Ks = vanillas.Ks
        sigmas = self._vol_svi(f, forward.T, Ks, params.array(), params.beta)
        Ns = vanillas.Ns
        res = np.zeros_like(sigmas)
        n = len(sigmas)
        for i in range(n):
            K = Ks[i]
            N = Ns[i]
            is_call = vanillas.is_call[i]
            res[i] = (
                N
                * self.bs_calc.premium(
                    forward, Strike(K), OptionType(is_call), ImpliedVol(sigmas[i])
                ).pv
            )
        return Premiums(res)

    def _total_implied_var_svi(
        self,
        F: nb.float64,
        Ks: nb.float64[:],
        params: nb.float64[:],
    ) -> nb.float64[:]:
        a, b, rho, m, sigma = params[0], params[1], params[2], params[3], params[4]
        k = np.log(Ks / F)
        w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))
        return w

    def _vol_svi(
        self,
        F: nb.float64,
        T: nb.float64,
        Ks: nb.float64[:],
        params: nb.float64[:],
    ) -> nb.float64[:]:
        w = self._total_implied_var_svi(F=F, T=T, Ks=Ks, params=params)
        iv = np.sqrt(w / T)
        return iv

    def _jacobian_total_implied_var_svi_raw(
        self,
        F: nb.float64,
        Ks: nb.float64[:],
        params: nb.float64[:],
    ) -> tuple[nb.float64[:]]:
        a, b, rho, m, sigma = params[0], params[1], params[2], params[3], params[4]
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

