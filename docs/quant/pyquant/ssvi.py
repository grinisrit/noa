import numba as nb
import numpy as np

from .black_scholes import *
from .common import *
from .svi import *
from .vol_surface import *


class SSVI:
    def __init__(
        self, vol_slime_chain_spaces: list[VolSmileChainSpace], is_log: bool = False
    ) -> None:
        self.is_log = is_log
        self.delta_space_params_list = []
        self.vol_slime_chain_spaces = vol_slime_chain_spaces
        for vol_slime_chain_space in self.vol_slime_chain_spaces:
            if self.is_log:
                print("\n")
                print(
                    f"======== Get natural params for tau = {vol_slime_chain_space.T} ======== "
                )
                print(f"Market IV {vol_slime_chain_space.sigmas}")
            # for every time to maturity calibrate it's own SVI with raw params
            svi_calc = SVICalc()
            svi_calibrated_params, svi_error = svi_calc.calibrate(
                vol_slime_chain_space,
                CalibrationWeights(np.ones(len(vol_slime_chain_space.Ks))),
                False,
                False,
            )
            if self.is_log:
                print(
                    f"Calibrated to market. Error = {svi_error.v}. Raw params = {svi_calibrated_params.array()}"
                )

                svi_test_iv = svi_calc.implied_vols(
                    vol_slime_chain_space.forward(),
                    Strikes(vol_slime_chain_space.Ks),
                    svi_calibrated_params,
                )

                print(f"Calibrated IV {svi_test_iv.data}")

            # Now get delta-space quotes
            svi_delta_space_chain = svi_calc.delta_space(
                vol_slime_chain_space.forward(),
                svi_calibrated_params,
            ).to_chain_space()
            if self.is_log:
                print(f"Delta-space strikes: {svi_delta_space_chain.strikes().data}")

            # Calibrate to delta-space
            svi_calibrated_params_delta, __ = svi_calc.calibrate(
                svi_delta_space_chain, CalibrationWeights(np.ones(5)), False, True
            )
            if self.is_log:
                print("Delta space params:", svi_calibrated_params_delta.array())
                svi_test_iv_delta = svi_calc.implied_vols(
                    vol_slime_chain_space.forward(),
                    Strikes(vol_slime_chain_space.Ks),
                    svi_calibrated_params_delta,
                )
                print(f"Delta-space market IV: {svi_test_iv_delta.data}")

            self.delta_space_params_list.append(svi_calibrated_params_delta.array())
            if self.is_log:
                print(
                    f"Natural parametrizarion delt-space params: {self.raw_to_natural_parametrization(svi_calibrated_params_delta)}"
                )

    def raw_to_natural_parametrization(self, svi_raw_params: SVIRawParams):
        a, b, rho, m, sigma = svi_raw_params.array()
        sqrt = np.sqrt(1 - rho**2)
        omega = 2 * b * rho / sqrt
        xi = sqrt / sigma
        mu = m + rho * sigma / sqrt
        delta = a - omega / 2 * (1 - rho**2)
        return [delta, mu, omega, xi]

    def calibrate(self):
        pass
