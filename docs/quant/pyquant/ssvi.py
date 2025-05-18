from typing import Any, Dict, Optional, Union

import numba as nb
import numpy as np

from .black_scholes import *
from .common import *
from .svi import *
from .svi import Rho
from .vol_surface import *


@nb.experimental.jitclass([("delta_param", nb.float64)])
class DeltaParam:
    def __init__(self, delta_param: nb.float64):
        self.delta_param = delta_param


@nb.experimental.jitclass([("mu", nb.float64)])
class Mu:
    def __init__(self, mu: nb.float64):
        self.mu = mu


@nb.experimental.jitclass([("theta", nb.float64)])
class Theta:
    def __init__(self, theta: nb.float64):
        if not (theta >= 0):
            raise ValueError("Theta not >= 0")
        self.theta = theta


@nb.experimental.jitclass([("zeta", nb.float64)])
class Zeta:
    def __init__(self, zeta: nb.float64):
        if not (zeta > 0):
            raise ValueError("Zeta not > 0")
        self.zeta = zeta


@nb.experimental.jitclass([("lambda_", nb.float64)])
class Lambda:
    def __init__(self, lambda_: nb.float64):
        if not (lambda_ >= 0):
            raise ValueError("Lambda not >= 0")
        self.lambda_ = lambda_


@nb.experimental.jitclass([("eta", nb.float64)])
class Eta:
    def __init__(self, eta: nb.float64):
        if not (eta >= 0):
            raise ValueError("Eta not >= 0")
        self.eta = eta


@nb.experimental.jitclass([("beta", nb.float64)])
class Beta:
    def __init__(self, beta: nb.float64):
        if not (beta >= 0):
            raise ValueError("Beta not >= 0")
        self.beta = beta


@nb.experimental.jitclass([("alpha", nb.float64)])
class Alpha:
    def __init__(self, alpha: nb.float64):
        self.alpha = alpha


@nb.experimental.jitclass([("gamma_", nb.float64)])
class Gamma_:
    def __init__(self, gamma_: nb.float64):
        self.gamma_ = gamma_


@nb.experimental.jitclass(
    [
        ("delta_param", nb.float64),
        ("mu", nb.float64),
        ("rho", nb.float64),
        ("theta", nb.float64),
        ("zeta", nb.float64),
        ("lambda_", nb.float64),
        ("eta", nb.float64),
        ("beta", nb.float64),
        ("alpha", nb.float64),
        ("gamma_", nb.float64),
    ]
)
class SVINaturalParams:
    def __init__(
        self, delta_param: DeltaParam, mu: Mu, rho: Rho, theta: Theta, zeta: Zeta
    ):
        self.delta_param = delta_param.delta_param
        self.mu = mu.mu
        self.rho = rho.rho
        self.theta = theta.theta
        self.zeta = zeta.zeta

    def array(self) -> nb.float64[:]:
        return np.array([self.delta_param, self.mu, self.rho, self.theta, self.zeta])


@nb.experimental.jitclass(
    [
        ("num_iter", nb.int64),
        ("max_mu", nb.float64),
        ("min_mu", nb.float64),
        ("tol", nb.float64),
        ("svi", SVICalc.class_type.instance_type),
    ]
)
class SSVICalc:
    def __init__(
        self,
    ) -> None:
        self.num_iter = 10000
        self.max_mu = 1e4
        self.min_mu = 1e-6
        self.tol = 1e-12
        self.svi = SVICalc()

    def calibrate(
        self,
        vol_surface_delta_space: VolSurfaceDeltaSpace,
        number_of_delta_space_dots: int = 20,
    ):
        thetas = np.zeros(number_of_delta_space_dots)
        self.cached_params = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        # we calibrate SVI to the linspace of max and min tenors given in space with given amount of ttm dots
        self.tenors_linspace = np.linspace(
            vol_surface_delta_space.min_T,
            vol_surface_delta_space.max_T,
            number_of_delta_space_dots,
        )
        n_points = 4 * number_of_delta_space_dots
        # calibrate tenor by tenor
        for idx, tenor in enumerate(self.tenors_linspace):
            vol_smile_chain_space: VolSmileChainSpace = (
                vol_surface_delta_space.get_vol_smile(
                    TimeToMaturity(tenor)
                ).to_chain_space()
            )
            svi_raw_params, calibration_error = self.svi.calibrate(
                vol_smile_chain_space,
                CalibrationWeights(np.ones_like(vol_smile_chain_space.Ks)),
                False,
                False,
                True,
            )
            svi_natural_params_array: SVINaturalParams = (
                self.raw_to_natural_parametrization(svi_raw_params)
            )
            thetas[idx] = svi_natural_params_array.theta
            forward = vol_smile_chain_space.forward()

            call25_K = self.svi.strike_from_delta(forward, Delta(0.25), svi_raw_params)
            call25 = self.svi.implied_vol(forward, call25_K, svi_raw_params).sigma

            put25_K = self.svi.strike_from_delta(forward, Delta(-0.25), svi_raw_params)
            put25 = self.svi.implied_vol(forward, put25_K, svi_raw_params).sigma

            call10_K = self.svi.strike_from_delta(forward, Delta(0.1), svi_raw_params)
            call10 = self.svi.implied_vol(forward, call10_K, svi_raw_params).sigma

            put10_K = self.svi.strike_from_delta(forward, Delta(-0.1), svi_raw_params)
            put10 = self.svi.implied_vol(forward, put10_K, svi_raw_params).sigma

        def clip_params(params: np.ndarray) -> np.ndarray:
            eps = 1e-5
            eta, lambda_, alpha, beta, omega = (
                params[0],
                params[1],
                params[2],
                params[3],
                params[4],
            )
            eta = np_clip(eta, 0, 1000000.0)
            beta = np_clip(beta, 0, 1000000.0)
            lambda_ = np_clip(lambda_, eps, 1 - eps)

            ssvi_params = np.array([eta, lambda_, alpha, beta, omega])
            return ssvi_params

        def get_residuals(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
                    mu /= nu1
                    result_error = F / n_points
                else:
                    i -= 1
                    mu *= nu2
                    continue
                result_x = x

            return result_x, result_error

        calc_params, calibration_error = levenberg_marquardt(
            get_residuals, clip_params, self.cached_params
        )

        print(calc_params)
        print(calibration_error)

    def _total_implied_var_ssvi(
        self, F: nb.float64, K: nb.float64, params: nb.float64[:], theta_t: nb.float64
    ) -> nb.float64:

        eta, lambda_, alpha, beta, omega = (
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
        )
        k = np.log(K / F)
        rho_t = alpha * np.exp ** (-beta * theta_t) + omega
        zeta_t = eta * theta_t ** (-lambda_)
        w = (
            theta_t
            / 2
            * (
                1
                + rho_t * zeta_t * k
                + np.sqrt((zeta_t * k + rho_t) ** 2 + 1 - rho_t**2)
            )
        )

        return w

    def raw_to_natural_parametrization(
        self, svi_raw_params: SVIRawParams
    ) -> SVINaturalParams:
        a, b, rho, m, sigma = svi_raw_params.array()
        sqrt = np.sqrt(1 - rho**2)
        theta = 2 * b * sigma / sqrt
        zeta = sqrt / sigma
        mu = m + rho * sigma / sqrt
        delta_param = a - theta / 2 * (1 - rho**2)
        return SVINaturalParams(
            DeltaParam(delta_param), Mu(mu), Rho(rho), Theta(theta), Zeta(zeta)
        )
