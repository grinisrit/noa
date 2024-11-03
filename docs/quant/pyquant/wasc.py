import numba as nb
import numpy as np

from .common import *
from .vol_surface import *

# from noa.docs.quant.pyquant.common import *
# from noa.docs.quant.pyquant.vol_surface import *


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
