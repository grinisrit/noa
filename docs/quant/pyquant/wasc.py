import numba as nb
import numpy as np

from .common import *
from .vol_surface import *


@nb.experimental.jitclass([("R", nb.float64[:, ::1])])
class R:
    def __init__(self, R: nb.float64[:, ::1]):
        if not len(R) == len(R[0]):
            raise ValueError("Matrix not square")
        self.R = R


@nb.experimental.jitclass([("Q", nb.float64[:, ::1])])
class Q:
    def __init__(self, Q: nb.float64[:, ::1]):
        if not len(Q) == len(Q[0]):
            raise ValueError("Matrix not square")
        self.Q = Q


@nb.experimental.jitclass([("sigma", nb.float64[:, ::1])])
class Sigma:
    def __init__(self, sigma: nb.float64[:, ::1]):
        if not len(sigma) == len(sigma[0]):
            raise ValueError("Matrix not square")
        self.sigma = sigma


@nb.experimental.jitclass(
    [
        ("R", nb.float64[:, ::1]),
        ("Q", nb.float64[:, ::1]),
        ("sigma", nb.float64[:, ::1]),
    ]
)
class WASCParams:
    def __init__(self, R: R, Q: Q, sigma: Sigma) -> None:
        if not len(Q.Q) == len(R.R) == len(sigma.sigma):
            raise ValueError("Matrixes are not of equal dimension")
        self.R = R.R
        self.Q = Q.Q
        self.sigma = sigma.sigma

    def array(self) -> nb.float64[:]:
        return np.concatenate(
            (self.R.reshape(-1), self.Q.reshape(-1), self.sigma.reshape(-1))
        )


def array_to_wasc_matrixes(array: nb.float64[:]) -> WASCParams:
    num_elements = len(array)
    if num_elements % 3 != 0:
        raise ValueError(
            "Array size must be divisible by 3 to form three equal square matrices."
        )
    elements_per_matrix = num_elements // 3
    matrix_size = int(np.sqrt(elements_per_matrix))
    # Check if elements_per_matrix is a perfect square
    if matrix_size * matrix_size != elements_per_matrix:
        raise ValueError(
            "Each part of the array must be a perfect square to form square matrices."
        )

    # Reshape the array into three square matrices
    matrices = np.split(array.reshape(3, matrix_size, matrix_size), 3)
    return WASCParams(R(matrices[0][0]), Q(matrices[1][0]), Sigma(matrices[2][0]))


class WASC:
    def __init__(self, params_dim: float = 2):
        self.num_iter = 10000
        self.max_mu = 1e4
        self.min_mu = 1e-6
        self.tol = 1e-4
        self.params_dim = params_dim
        # Create param flatten array of 3 params and each matrix of params_dim*params_dim size
        # self.raw_cached_params = np.ones(3 * params_dim**2)
        self.raw_cached_params = np.random.normal(
            loc=1, scale=0.05, size=3 * params_dim**2
        )

    def _vol_wasc(
        self,
        F: nb.float64,
        Ks: nb.float64[:],
        params: WASCParams,
    ) -> nb.float64[:]:
        R, Q, sigma = params.R, params.Q, params.sigma
        mf = np.log(Ks / F)
        sigma_trace = np.trace(sigma)
        # return np.sqrt(
        return (
            sigma_trace
            + np.trace(R @ Q @ sigma) / sigma_trace * mf
            + mf**2
            / sigma_trace**2
            * (
                1 / 3 * np.trace(Q.T @ Q @ sigma)
                + 1 / 3 * np.trace(R @ Q @ (Q.T @ R.T + R @ Q) @ sigma)
                - 5 / 4 * np.trace(R @ Q @ sigma) ** 2 / sigma_trace
            )
        )

    def _jacobian_implied_vol_single_strike_wasc(
        self,
        F: nb.float64,
        K: nb.float64,
        params: WASCParams,
    ) -> nb.float64[:]:
        R, Q, sigma = params.R, params.Q, params.sigma
        mf = np.log(K / F)
        sigma_trace = np.trace(sigma)
        tr_RQ_sigma = np.trace(R @ Q @ sigma)
        tr_QTQ_sigma = np.trace(Q.T @ Q @ sigma)

        # ============ for Q ==============
        term2 = (mf / sigma_trace) * R.T @ sigma

        term3 = (mf**2 / (3 * sigma_trace**2)) * (Q @ sigma + sigma @ Q)

        part1 = R.T @ sigma @ R @ Q.T + R.T @ sigma @ Q @ R
        part2 = R.T @ sigma.T @ Q @ R + R.T @ Q.T @ sigma @ R
        term4 = (mf**2 / (3 * sigma_trace**2)) * (part1 + part2)

        term5 = (-5 * mf**2 / (2 * sigma_trace**3)) * tr_RQ_sigma * R.T @ sigma

        Q_diff = term2 + term3 + term4 + term5

        # ============ for R ==============

        term2 = (mf / sigma_trace) * sigma.T @ Q.T

        part1 = sigma.T @ R @ Q @ Q.T + sigma.T @ Q @ Q.T @ R
        part2 = sigma @ Q.T @ R @ Q + sigma @ Q @ Q.T @ R
        term3 = (mf**2 / (3 * sigma_trace**2)) * (part1 + part2)

        term4 = (-5 * mf**2 / (2 * sigma_trace**3)) * tr_RQ_sigma * sigma.T @ Q.T

        R_diff = term2 + term3 + term4

        # ============ for sigma ==============

        term1 = np.eye(sigma.shape[0])

        term2_part1 = (mf / sigma_trace) * (R @ Q)
        term2_part2 = -(mf * tr_RQ_sigma / sigma_trace**2) * np.eye(sigma.shape[0])
        term2 = term2_part1 + term2_part2

        term3_part1 = (mf**2 / (3 * sigma_trace**2)) * (Q.T @ Q)
        term3_part2 = -(2 * mf**2 * tr_QTQ_sigma / (3 * sigma_trace**3)) * np.eye(
            sigma.shape[0]
        )
        term3 = term3_part1 + term3_part2

        complex_term = R @ Q @ (Q.T @ R.T + R @ Q)
        term4_part1 = (mf**2 / (3 * sigma_trace**2)) * complex_term
        term4_part2 = -(
            2 * mf**2 * np.trace(complex_term) / (3 * sigma_trace**3)
        ) * np.eye(sigma.shape[0])
        term4 = term4_part1 + term4_part2

        term5_part1 = (-5 * mf**2 * tr_RQ_sigma / (2 * sigma_trace**3)) * (R @ Q)
        term5_part2 = (
            15 * mf**2 * tr_RQ_sigma**2 / (4 * sigma_trace**4)
        ) * np.eye(sigma.shape[0])
        term5 = term5_part1 + term5_part2

        sigma_diff = term1 + term2 + term3 + term4 + term5

        w = self._vol_wasc(F, K, params)
        # denominator = 2 * np.sqrt(w)

        # return np.concatenate(
        #     (
        #         (Q_diff / denominator).flatten(),
        #         (R_diff / denominator).flatten(),
        #         (sigma_diff / denominator).flatten(),
        #     ),
        #     axis=0,
        # )
        return np.concatenate(
            (
                (2 * w * Q_diff).flatten(),
                (2 * w * R_diff).flatten(),
                (2 * w * sigma_diff).flatten(),
            ),
            axis=0,
        )

    def _jacobian_implied_vol_wasc(
        self,
        F: nb.float64,
        Ks: nb.float64[:],
        params: WASCParams,
    ) -> nb.float64[:]:
        jacs = []
        for K in Ks:
            jac = self._jacobian_implied_vol_single_strike_wasc(F, K, params)
            jacs.append(jac)
        return np.vstack(jacs).T

    def calibrate(
        self,
        chain: VolSmileChainSpace,
        calibration_weights: CalibrationWeights,
        update_cached_params: bool = True,
    ) -> Tuple[WASCParams, CalibrationError]:
        strikes = chain.Ks
        w = calibration_weights.w

        if not strikes.shape == w.shape:
            raise ValueError(
                "Inconsistent data between strikes and calibration weights"
            )

        n_points = len(strikes)

        weights = w / w.sum()
        forward = chain.f
        vars = chain.sigmas

        def clip_params(params):
            return params

        def get_residuals(params):
            J = self._jacobian_implied_vol_wasc(
                forward,
                strikes,
                params,
            )

            wasc_w = self._vol_wasc(
                forward,
                strikes,
                params,
            )
            res = wasc_w - vars
            return res * weights, J @ np.diag(weights)

        def levenberg_marquardt(f, proj, x0):
            x = x0.copy()

            mu = 1e-2
            nu1 = 2.0
            nu2 = 2.0

            res, J = f(array_to_wasc_matrixes(x))
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
                res_, J_ = f(array_to_wasc_matrixes(x_))
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
                print(result_x, result_error)
            return result_x, result_error

        calc_params, calibration_error = levenberg_marquardt(
            get_residuals, clip_params, self.raw_cached_params
        )
        wasc_params: WASCParams = array_to_wasc_matrixes(calc_params)

        if update_cached_params:
            self.raw_cached_params = wasc_params

        return wasc_params, CalibrationError(calibration_error)
