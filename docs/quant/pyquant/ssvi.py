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
        ("eta", nb.float64),
        ("lambda_", nb.float64),
        ("alpha", nb.float64),
        ("beta", nb.float64),
        ("gamma_", nb.float64),
    ]
)
class SSVIParams:

    def __init__(
        self, eta: Eta, lambda_: Lambda, alpha: Alpha, beta: Beta, gamma_: Gamma_
    ):
        self.eta = eta.eta
        self.lambda_ = lambda_.lambda_
        self.alpha = alpha.alpha
        self.beta = beta.beta
        self.gamma_ = gamma_.gamma_

    def array(self) -> nb.float64[:]:
        return np.array([self.eta, self.lambda_, self.alpha, self.beta, self.gamma_])


@nb.experimental.jitclass(
    [
        ("num_iter", nb.int64),
        ("max_mu", nb.float64),
        ("min_mu", nb.float64),
        ("tol", nb.float64),
        ("svi", SVICalc.class_type.instance_type),
        ("cached_params", nb.float64[:]),
    ]
)
class SSVICalc:
    def __init__(
        self,
    ) -> None:
        # self.num_iter = 10000
        self.num_iter = 100
        self.max_mu = 1e4
        self.min_mu = 1e-6
        self.tol = 1e-12
        self.svi = SVICalc()
        self.cached_params = np.array([1.0, 0.2, 0.05, 0.1, 0.0])
        # eta, lambda, alpha, beta, gamma

    def calibrate(
        self,
        vol_surface_delta_space: VolSurfaceDeltaSpace,
        number_of_delta_space_dots: int = 20,
    ) -> Tuple[SSVIParams, CalibrationError]:
        NUMBER_OF_DOTS_PER_SMILE = 4
        thetas = np.zeros(number_of_delta_space_dots)
        n_points = NUMBER_OF_DOTS_PER_SMILE * number_of_delta_space_dots
        # write final IVs here to ehich we gonna calibrate
        implied_variances = np.zeros(n_points)
        weights = np.ones(n_points)
        weights = weights / weights.sum()
        # array for creating StrikesMaturitiesGrid
        strikes = np.zeros(n_points)

        # we calibrate SVI to the linspace of max and min tenors given in space with given amount of ttm dots
        tenors_linspace = np.linspace(
            vol_surface_delta_space.min_T,
            vol_surface_delta_space.max_T,
            number_of_delta_space_dots,
        )

        # calibrate tenor by tenor
        for idx, tenor in enumerate(tenors_linspace):
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

            chain_space_from_delta_space: VolSmileChainSpace = self.svi.delta_space(
                vol_smile_chain_space.forward(), svi_raw_params
            ).to_chain_space()
            # Do not take ATM, only 0.1 and 0.25 call/put deltas
            strikes[
                NUMBER_OF_DOTS_PER_SMILE * idx : NUMBER_OF_DOTS_PER_SMILE * (idx + 1)
            ] = np.concatenate(
                (
                    chain_space_from_delta_space.Ks[:2],
                    chain_space_from_delta_space.Ks[-2:],
                )
            )
            # NOTE: convert iv-s to implied variances
            implied_variances[
                NUMBER_OF_DOTS_PER_SMILE * idx : NUMBER_OF_DOTS_PER_SMILE * (idx + 1)
            ] = (
                tenor
                * np.concatenate(
                    (
                        chain_space_from_delta_space.sigmas[:2],
                        chain_space_from_delta_space.sigmas[-2:],
                    )
                )
                ** 2
            )
            # TODO: here the arbitrage can be tracked and fixed
        print("Implied variances to calibrate to:", implied_variances)
        print("Strikes from delta-space we calibrate to:", strikes)

        # get all the strikes and maturities grid
        strikes_to_maturities_grid: StrikesMaturitiesGrid = StrikesMaturitiesGrid(
            chain_space_from_delta_space.forward().spot(),  # it is similar in every smile
            TimesToMaturity(np.repeat(tenors_linspace, NUMBER_OF_DOTS_PER_SMILE)),
            Strikes(strikes),
        )
        # make the array of thetas of the same size
        thetas = np.repeat(thetas, NUMBER_OF_DOTS_PER_SMILE)
        print("Thetas by dots:", thetas)

        def clip_params(params: np.array) -> np.array:
            eps = 1e-5
            eta, lambda_, alpha, beta, gamma_ = (
                params[0],
                params[1],
                params[2],
                params[3],
                params[4],
            )
            eta = np_clip(eta, 0.0, 1000000.0)
            # NOTE: need to clip or explodes
            # alpha = np_clip(alpha, 0, 1.0)
            beta = np_clip(beta, 0.0, 1000000.0)
            lambda_ = np_clip(lambda_, eps, 1 - eps)

            ssvi_params = np.array([eta, lambda_, alpha, beta, gamma_])
            return ssvi_params

        def get_residuals(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            ssvi_params = SSVIParams(
                Eta(params[0]),
                Lambda(params[1]),
                Alpha(params[2]),
                Beta(params[3]),
                Gamma_(params[4]),
            )

            ivs = self._grid_implied_variances(
                ssvi_params, strikes_to_maturities_grid, thetas
            )
            residuals = (ivs - implied_variances) * weights
            jacobian = self._jacobian_total_implied_var_ssvi(
                ssvi_params, strikes_to_maturities_grid, thetas
            )
            jacobian = jacobian @ np.diag(weights)
            return residuals, jacobian

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
                print("RESULT:", result_x, result_error)
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
                    mu = max(self.min_mu, mu / nu1)
                    result_error = F / n_points
                else:
                    i -= 1
                    mu = min(self.max_mu, mu * nu2)
                    continue
                result_x = x
            return result_x, result_error

        calc_params, calibration_error = levenberg_marquardt(
            get_residuals, clip_params, self.cached_params
        )
        print(calc_params)
        print(calibration_error)
        return calc_params, calibration_error

    def _jacobian_total_implied_var_ssvi(
        self,
        ssvi_params: SSVIParams,
        grid: StrikesMaturitiesGrid,
        thetas: nb.float64[:],
    ) -> nb.float64[:, :]:
        """Computes Jacobian w.r.t. SSVIParams."""
        Ks = grid.Ks
        Ts = grid.Ts
        n = len(Ks)
        F = grid.S
        deta, dlambda_, dalpha, dbeta, dgamma_ = (
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
        )
        eta, lambda_, alpha, beta, gamma_ = ssvi_params.array()
        jacs = np.zeros((5, n), dtype=np.float64)
        for l in range(n):
            K = Ks[l]
            T = Ts[l]

            theta_t = thetas[l]
            k = np.log(K / F)
            zeta_t = eta * theta_t ** (-lambda_)
            rho_t = alpha * np.exp(-beta * theta_t) + gamma_
            assert -1 <= rho_t <= 1, f"It should be abs(rho)<=1. Now it is {rho_t}"

            deta = (
                theta_t
                * (
                    k * rho_t / theta_t**lambda_
                    + k
                    * (k * zeta_t + rho_t)
                    / (
                        theta_t**lambda_
                        * np.sqrt(-(rho_t**2) + (k * zeta_t + rho_t) ** 2 + 1)
                    )
                )
                / 2
            )

            dlambda_ = (
                theta_t
                * (
                    -k * rho_t * zeta_t * np.log(theta_t)
                    - k
                    * zeta_t
                    * (k * zeta_t + rho_t)
                    * np.log(theta_t)
                    / np.sqrt(-(rho_t**2) + (k * zeta_t + rho_t) ** 2 + 1)
                )
                / 2
            )
            dalpha = (
                theta_t
                * (
                    k * zeta_t * np.exp(-beta * theta_t)
                    + (
                        -rho_t * np.exp(-beta * theta_t)
                        + (k * zeta_t + rho_t) * np.exp(-beta * theta_t)
                    )
                    / np.sqrt(-(rho_t**2) + (k * zeta_t + rho_t) ** 2 + 1)
                )
                / 2
            )
            dbeta = (
                theta_t
                * (
                    -alpha * k * theta_t * zeta_t * np.exp(-beta * theta_t)
                    + (
                        alpha * rho_t * theta_t * np.exp(-beta * theta_t)
                        - alpha
                        * theta_t
                        * (k * zeta_t + rho_t)
                        * np.exp(-beta * theta_t)
                    )
                    / np.sqrt(-(rho_t**2) + (k * zeta_t + rho_t) ** 2 + 1)
                )
                / 2
            )
            dgamma_ = (
                theta_t
                * (
                    k * zeta_t
                    + k * zeta_t / np.sqrt(-(rho_t**2) + (k * zeta_t + rho_t) ** 2 + 1)
                )
                / 2
            )

            jacs[0][l] = deta
            jacs[1][l] = dlambda_
            jacs[2][l] = dalpha
            jacs[3][l] = dbeta
            jacs[4][l] = dgamma_

        return jacs

    def _grid_implied_variances(
        self,
        ssvi_params: SSVIParams,
        grid: StrikesMaturitiesGrid,
        thetas: nb.float64[:],
    ) -> nb.float64[:]:
        """Calculates the premium of vanilla option under the SSVI model."""
        Ks = grid.Ks
        F = grid.S
        eta, lambda_, alpha, beta, gamma_ = ssvi_params.array()

        w = np.zeros_like(Ks)
        for l in range(len(Ks)):
            K = Ks[l]
            theta_t = thetas[l]
            k = np.log(K / F)
            zeta_t = eta * theta_t ** (-lambda_)
            rho_t = alpha * np.exp(-beta * theta_t) + gamma_

            assert -1 <= rho_t <= 1, f"It should be abs(rho)<=1. Now it is {rho_t}"
            w[l] = (
                theta_t
                / 2
                * (
                    1
                    + rho_t * k * zeta_t
                    + np.sqrt(1 - rho_t**2 + (rho_t + zeta_t * k) ** 2)
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
