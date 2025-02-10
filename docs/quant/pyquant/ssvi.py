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


class SSVI:
    def __init__(
        self,
        vol_smile_chain_spaces: list[VolSmileChainSpace],
        is_log: bool = False,
    ) -> None:
        self.is_log = is_log
        self.raw_params_list = []
        self.natural_params_list = []
        self.vol_smile_chain_spaces = vol_smile_chain_spaces

    def calibrate(
        self,
        for_delta_space: bool = False,
    ) -> None:
        for vol_smile_chain_space in self.vol_smile_chain_spaces:
            if self.is_log:
                print("\n")
                print(
                    f"======== Get natural params for tau = {vol_smile_chain_space.T} ======== "
                )
                print(f"Market IV {vol_smile_chain_space.sigmas}")
            # 1. for every time to maturity calibrate it's own SVI with raw params
            svi_calc = SVICalc()

            svi_calibrated_params, svi_error = svi_calc.calibrate(
                vol_smile_chain_space,
                CalibrationWeights(np.ones(len(vol_smile_chain_space.Ks))),
                False,
                for_delta_space,
            )
            if self.is_log:
                print(
                    f"Calibrated tau = {vol_smile_chain_space.T} SVI to market. Error = {svi_error.v}. Raw params = {svi_calibrated_params.array()}"
                )

                svi_log_iv = svi_calc.implied_vols(
                    vol_smile_chain_space.forward(),
                    Strikes(vol_smile_chain_space.Ks),
                    svi_calibrated_params,
                )

                print(f"Calibrated IV: {svi_log_iv.data}")

            a, b, rho, m, sigma = svi_calibrated_params.array()
            natural_params = self.raw_to_natural_parametrization(
                SVIRawParams(A(a), B(b), Rho(rho), M(m), Sigma(sigma))
            )

            self.natural_params_list.append(natural_params)

            if self.is_log:
                print(f"Natural parametrizarion params: {natural_params.array()}")

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

    def __interpolate_nu_lambda(zetas: list[Zeta], thetas: list[Theta]):
        pass

    def __interpolate_alpha_beta(rhos: list[Rho], thetas: list[Theta]):
        pass

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
