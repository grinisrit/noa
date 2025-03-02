from typing import Any, Dict, Optional, Union

import numba as nb
import numpy as np
from scipy.optimize import minimize

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


class SSVICalc:
    def __init__(
        self,
    ) -> None:
        self.svi = SVICalc()

    def calibrate(
        self,
        vol_surface_delta_space: VolSurfaceDeltaSpace,
        number_of_delta_space_dots: int = 20,
    ):
        zetas, rhos, thetas = [], [], []
        # we calibrate SVI with a, m = 0 to the linspace of max and min tenors given in space
        tenors_linspace = np.linspace(
            vol_surface_delta_space.min_T,
            vol_surface_delta_space.max_T,
            number_of_delta_space_dots,
        )
        # calibrate tenor by tenor
        for tenor in tenors_linspace:
            vol_smile_chain_space: VolSmileDeltaSpace = (
                vol_surface_delta_space.get_vol_smile(
                    TimeToMaturity(tenor)
                ).to_chain_space()
            )
            svi_raw_params, _ = self.svi.calibrate(
                vol_smile_chain_space,
                CalibrationWeights(np.ones_like(vol_smile_chain_space.Ks)),
                False,
                False,
                True,
            )
            svi_natural_params_array: SVINaturalParams = (
                self.raw_to_natural_parametrization(svi_raw_params)
            )
            zetas.append(svi_natural_params_array.zeta)
            thetas.append(svi_natural_params_array.theta)
            rhos.append(svi_natural_params_array.rho)
        eta, lambda_ = self._interpolate_eta_lambda(zetas, thetas)
        alpha, beta, gamma = self._interpolate_alpha_beta_gamma(rhos, thetas)
        return eta, lambda_, alpha, beta, gamma

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

    def _interpolate_eta_lambda(
        self, zetas: list[Zeta], thetas: list[Theta]
    ) -> Union[Eta, Lambda]:

        def model(params, thetas):
            eta, lambda_ = params
            return eta * thetas**lambda_

        def loss_function(params):
            predictions = model(params, thetas)
            return np.sum((zetas - predictions) ** 2)

        result = minimize(loss_function, [1, 1])
        optimal_eta, optimal_lambda_ = result.x
        # return Eta(optimal_eta), Lambda(optimal_lambda_)
        return optimal_eta, optimal_lambda_

    def _interpolate_alpha_beta_gamma(
        self, rhos: list[Rho], thetas: list[Theta]
    ) -> Union[Alpha, Beta, Gamma_]:

        def model(params, thetas):
            alpha, beta, gamma = params
            return alpha * np.exp(-beta * np.array(thetas)) + gamma

        def loss_function(params):
            predictions = model(params, thetas)
            return np.sum((rhos - predictions) ** 2)

        result = minimize(loss_function, [1, 1, 1])
        optimal_alpha, optimal_beta, optimal_gamma = result.x
        # return Alpha(optimal_alpha), Beta(optimal_beta), Gamma_(optimal_gamma)
        return optimal_alpha, optimal_beta, optimal_gamma

    def _total_implied_var_ssvi(
        self,
        F: nb.float64,
        K: nb.float64,
        params: nb.float64[:],
    ) -> nb.float64:
        delta_param, mu, rho, theta, zeta = (
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
        )
        k = np.log(K / F)
        w = delta_param + theta / 2 * (
            1
            + zeta * rho * (k - mu) * np.sqrt((zeta * (k - mu) + rho) ** 2 + 1 - rho**2)
        )
        return w
