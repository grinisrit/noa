import numpy as np
from numba import njit
# TODO: add comments


@njit()
def put_func(q, t, x):
    return np.exp(t * (q + 1) ** 2 / 4) * np.maximum(np.zeros_like(x),
                                                     np.exp((q - 1) * x / 2) - np.exp((q + 1) * x / 2))


@njit()
def call_func(q, t, x):
    return np.exp((q + 1) ** 2 * t / 4) * np.maximum(np.zeros_like(x),
                                                     np.exp((q + 1) * x / 2) - np.exp((q - 1) * x / 2))

@njit()
def scalar_walk(A, B):
    """
    three-dots scalar walk algorythm for solving Linear systems with three-diagonal matrix
    :param A: three-diagonal matrix
    :param B: array
    :return: solution for system A*x=B
    """""
    # scalar walk coefficients (forward walk)
    f = -B
    n = len(A)
    P = np.zeros(n - 1)
    Q = np.zeros(n)
    P[0] = -1 * A[0][1] / A[0][0]
    Q[0] = -1 * f[0] / A[0][0]
    for i in range(1, n):
        a = A[i][i - 1]
        c = -1 * A[i][i]
        if i != n - 1:
            b = A[i][i + 1]
            P[i] = b / (c - a * P[i - 1])
        Q[i] = (f[i] + a * Q[i - 1]) / (c - a * P[i - 1])

    # (backward walk)
    x = np.zeros(n)
    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]
    return x


@njit
def brennan_schwartz(alpha, beta, gamma, b, g):
    """
    Solution to Ax-b >= 0 ; x >= g and (Ax-b)'(x-g)=0 ;

    A - bidiagonal matrix with alpha, beta, gamma coefficients
    alpha: [n x 1] numpy vector  of main diagonal of A
    beta: [(n-1) x 1] numpy vector of upper diagonal
    gamma: [(n-1) x 1] numpy vector of lower diagonal
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
