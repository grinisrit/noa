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


class WASC:
    def __init__(self):
        self.num_iter = 10000
        self.max_mu = 1e4
        self.min_mu = 1e-6
        self.tol = 1e-12

    def _vol_wasc(
        self,
        F: nb.float64,
        Ks: nb.float64[:],
        params: WASCParams,
    ) -> nb.float64[:]:
        R, Q, sigma = params.R, params.Q, params.sigma
        mf = np.log(Ks / F)
        sigma_trace = np.trace(sigma)
        return np.sqrt(
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

    def _jacobian_implied_var_wasc(
        self,
        F: nb.float64,
        Ks: nb.float64,
        params: WASCParams,
    ):
        R, Q, sigma = params.R, params.Q, params.sigma
        mf = np.log(Ks / F)
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

        w = self._vol_wasc(F, Ks, params)
        denominator = 2 * np.sqrt(w)
        return Q_diff / denominator, R_diff / denominator, sigma_diff / denominator
