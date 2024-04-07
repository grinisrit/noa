import math
import numpy as np
import numba as nb
from typing import Tuple, List


@nb.njit()
def _brennan_schwartz(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, b, g):
    """Computes solution to Ax - b >= 0 ; x >= g and (Ax-b)'(x-g)=0.
    A is a tridiagonal matrix with alpha, beta, gamma coefficients.

    Args:
        alpha: Main diagonal of A.  Shape: (npoints_S,).
        beta:  Upper diagonal of A. Shape: (npoints_S - 1,).
        gamma: Lower diagonal of A. Shape: (npoints_S - 1,).
        b, g
    """
    n = len(alpha)
    alpha_hat = np.zeros(n, dtype=np.float64)
    b_hat = np.zeros(n, dtype=np.float64)
    alpha_hat[-1] = alpha[-1]
    b_hat[-1] = b[-1]

    for i in range(n - 2, -1, -1):
        alpha_hat[i] = alpha[i] - beta[i] * gamma[i] / alpha_hat[i + 1]
        b_hat[i] = b[i] - beta[i] * b_hat[i + 1] / alpha_hat[i + 1]

    x = np.zeros(n, dtype=np.float64)
    x[0] = np.maximum(b_hat[0] / alpha_hat[0], g[0])
    for i in range(1, n):
        x[i] = np.maximum((b_hat[i] - gamma[i - 1] * x[i - 1]) / alpha_hat[i], g[i])
    return x


@nb.njit()
def _get_A_diags(size, lambda_):
    """Computes diagonals of the A matrix (for implicit step)."""
    alpha = (1 + lambda_) * np.ones(size+1)
    beta = -0.5 * lambda_ * np.ones(size)
    gamma = -0.5 * lambda_ * np.ones(size)
    return alpha, beta, gamma


@nb.njit()
def _g_func(tau, x, k):
    g_vector = np.exp(tau * (k + 1)**2 / 4) * np.maximum(0, np.exp((k - 1)*x/2) - np.exp((k + 1)*x/2))
    return g_vector

@nb.njit()
def _get_crank_symtridiag_matrix(n_points, lambda_):
    Bu = np.empty(n_points-1, dtype=np.float64)
    Bu.fill(lambda_ / 2)
    Bd = np.empty(n_points, dtype=np.float64) 
    Bd.fill(1 - lambda_)
    return Bu, Bd

@nb.njit()
def _dot_symtridiag_matvec(Bu, Bd, w):
    v = np.zeros_like(Bd)
    v[0] = Bd[0]*w[0] + Bu[0]*w[1]
    v[-1] = Bu[-1]*w[-2] + Bd[-1]*w[-1]
    for i in range(1, len(w) - 1):
        v[i] = Bu[i]*w[i-1] + Bd[i]*w[i] + Bu[i]*w[i+1]
    return v

@nb.njit()
def _transform(w_matrix, x_array, tau_array, K, T, r, sigma):
    k = 2 * r / sigma ** 2
    coef = ((k + 1)**2) / 4
    V = np.zeros_like(w_matrix)
    for n in np.arange(len(tau_array)):
        for m in np.arange(len(x_array)):
            V[m, n] = K * np.exp((1 - k) * x_array[m] / 2 - coef * tau_array[n]) * w_matrix[m, n]
    t_array = np.array([T - 2 * t / sigma**2 for t in tau_array])
    t_array[-1] = 0
    S_array = K * np.exp(x_array)
    return V, S_array, t_array


@nb.njit()
def price_american_put_bsm(
        K: float,
        T: float,
        r: float,
        sigma: float,
        S_min: float,
        S_max: float,
        npoints_S: int = 1000,
        npoints_t: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the value of American put option under the Black-Scholes model,
       using the Brennan-Schwartz algorithm.

    Args:
        K: Strike price.
        T: Time to expiry in years.
        r: Risk-free rate.
        sigma: Volatility.
        S_min: Minimum underlying price for the (S, t) grid.
        S_max: Maximum unerlying price for the (S, t) grid.
        npoints_S: Number of underlying price points on the grid.
        npoints_t: Number of time points on the grid.

    Returns:
        1) Values of the option on the (S, t) grid. Shape: (npoints_S, npoints_t).
        2) Underlying price values from the grid. Shape: (npoints_S,).
        3) Time values from the grid. Shape: (npoints_t,).
        Note that the spacing of the grid in variable S is not uniform:
        it increases exponentially. In variable t the spacing is uniform.
    """
    tau_max = T * (sigma ** 2) / 2
    x_min = math.log(S_min / K)
    x_max = math.log(S_max / K)
    delta_tau = tau_max / (npoints_t - 1)
    delta_x = (x_max - x_min) / (npoints_S - 1)
    lambda_ = delta_tau / (delta_x ** 2)
    k = 2 * r / sigma ** 2

    x = np.linspace(x_min, x_max, npoints_S)
    tau_array = np.linspace(0, tau_max, npoints_t)
    w_matrix = np.zeros((npoints_S, npoints_t))
    alpha, beta, gamma = _get_A_diags(npoints_S - 1, lambda_)
    Bu, Bd = _get_crank_symtridiag_matrix(npoints_S, lambda_)
    

    # setting initial and boundary conditions
    w_matrix[:, 0] = _g_func(0, x, k)
    w_matrix[0, :] = _g_func(tau_array, x_min, k)

    for nu in range(npoints_t - 1):
        w = w_matrix[:, nu]
        d = np.zeros_like(w)
        d[0] = lambda_ / 2 * (_g_func(tau_array[nu], x_min, k) + _g_func(tau_array[nu + 1], x_min, k))
        # explicit step
        f = _dot_symtridiag_matvec(Bu, Bd, w) + d
        # implicit step
        w_ = _brennan_schwartz(alpha, beta, gamma, f, _g_func(tau_array[nu + 1], x, k))
        w_matrix[:, nu + 1] = w_

    V, S_array, t_array = _transform(w_matrix, x, tau_array, K, T, r, sigma)
    return V, S_array, t_array


@nb.njit()
def find_early_exercise(V, S_array, t_array, K, tol=1e-5) -> Tuple[List, List]:
    stop_S_values = [K]
    stop_V_values = [0]
    for i in range(1, len(t_array)):
        stop_idx = np.argmax(V[:, i] > np.maximum(K - S_array + tol, 0))
        stop_S_values.append(S_array[stop_idx])
        stop_V_values.append(V[stop_idx, i])
    return stop_V_values, stop_S_values
