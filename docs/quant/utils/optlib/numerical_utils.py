import numpy as np
from numba import njit
# TODO: add comments


@njit()
def put_func(q, t, x):
    return np.exp(t * (q + 1) ** 2 / 4) * max(0, np.exp((q - 1) * x / 2) - np.exp((q + 1) * x / 2))


@njit()
def call_func(q, t, x):
    return np.exp((q + 1) ** 2 * t / 4) * max(0, np.exp((q + 1) * x / 2) - np.exp((q - 1) * x / 2))


@njit()
def set_bounds(net, q, t_array, x_array, call: bool):
    if call:
        for i in range(len(t_array)):
            net[-1, i] = call_func(q, t_array[i], x_array[-1])
        for i in range(len(x_array)):
            net[i, 0] = call_func(q, t_array[0], x_array[i])
    else:
        for i in range(len(t_array)):
            net[-1, i] = put_func(q, t_array[i], x_array[-1])
        for i in range(len(x_array)):
            net[i, 0] = put_func(q, t_array[0], x_array[i])
    return net


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
def get_matrices(size, lambda_):
    A = np.diag(np.ones(size) * (1 + lambda_))
    B = np.diag(np.ones(size) * (1 - lambda_))
    size = size - 1
    for i in range(1, size):
        A[i, i - 1] = -lambda_ / 2
        A[i - 1, i] = -lambda_ / 2
        B[i, i - 1] = lambda_ / 2
        B[i - 1, i] = lambda_ / 2
    A[0, 0] = 1
    A[size, size] = 1
    A[0, 1] = 0
    A[size, size-1] = 0
    B[0, 0] = 1
    B[size, size] = 1
    B[0, 1] = 0
    B[size, size-1] = 0
    return A, B


@njit
def crank_nickolson_scheme(net, lambda_):
    size, time_steps = net.shape
    A, B = get_matrices(size, lambda_)
    for i in range(0, time_steps-1):
        # explicit step
        f = np.dot(B, net[:, i])
        # implicit step
        solution = scalar_walk(A, f)
        net[:, i + 1] = solution
    return net


# diagonals of the A matrix (implicit step):
@njit
def get_matrix_diag(size, lambda_):
    alpha = (1 + lambda_) * np.ones(size+1)
    alpha[0] = 1
    alpha[-1] = 1
    beta = -0.5 * lambda_ * np.ones(size)
    beta[0] = 0
    gamma = -0.5 * lambda_ * np.ones(size)
    gamma[-1] = 0
    return alpha, beta, gamma


@njit
def brennan_schwartz_algorithm(alpha, beta, gamma, b, g):
    """
    Solution to Ax-b >= 0 ; x >= g and (Ax-b)'(x-g)=0 ;

    A - bidiagonal matrix with alpha, beta, gamma coefficients
    alpha: [n x 1] numpy vector  of main diagonal of A
    beta: [(n-1) x 1] numpy vector of upper diagonal
    gamma: [(n-1) x 1] numpy vector of lower diagonal
    """""

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


# brennan-shwartz algorythm (only for put option)
@njit
def brennan_schwartz_scheme(net, time_vector, x_vector, lambda_, k):
    size, time_steps = net.shape
    A, B = get_matrices(size, lambda_)
    alpha, beta, gamma = get_matrix_diag(size, lambda_)
    for i in range(0, time_steps-1):
        # setting early execution curve
        time = time_vector[i + 1]
        g = tuple(map(lambda x: put_func(k, time, x), x_vector))
        # explicit step
        f = np.dot(B, net[:, i])
        # implicit step
        solution = brennan_schwartz_algorithm(alpha, beta, gamma, f, g)
        net[:, i + 1] = solution
    return net


@njit()
def get_crank_B_matrix(m, lambda_):
    B = np.zeros((m + 1, m + 1))
    for i in range(m):
        B[i][i] = 1 - lambda_
        B[i][i + 1] = lambda_ / 2
        B[i + 1][i] = lambda_ / 2
    B[m][m] = 1 - lambda_
    return B


@njit()
def get_A_diags(size, lambda_):
    alpha = (1 + lambda_) * np.ones(size+1)
    beta = -0.5 * lambda_ * np.ones(size)
    gamma = -0.5 * lambda_ * np.ones(size)
    return alpha, beta, gamma


@njit()
def g_func(tau, x, k):
    g_vector = np.exp(tau * (k + 1)**2 / 4) * np.maximum(0, np.exp((k - 1)*x/2) - np.exp((k + 1)*x/2))
    return g_vector


@njit()
def price_american_put(T, r, sigma, x_min, x_max, delta_x, delta_tau):
    k = 2 * r / sigma ** 2
    tau_max = T * (sigma ** 2) / 2
    x = np.arange(x_min, x_max + delta_x, delta_x)
    m = int((x_max - x_min) / delta_x)
    lambda_ = delta_tau / (delta_x ** 2)
    tau_array = np.arange(0, tau_max + delta_tau, delta_tau)
    w_matrix = np.zeros((len(x),len(tau_array)))
    alpha, beta, gamma = get_A_diags(len(x)-1, lambda_)
    B = get_crank_B_matrix(m, lambda_)

    # setting initial and boundary conditions
    w_matrix[:,0] = g_func(0, x, k)
    w_matrix[0,:] = g_func(tau_array, x_min, k)

    for nu in range((len(tau_array)) - 1):
        w = w_matrix[:, nu]
        d = np.zeros_like(w)
        d[0] = lambda_ / 2 * (g_func(tau_array[nu], x_min, k) + g_func(tau_array[nu+1], x_min, k))
        # explicit step
        f = np.dot(B, w) + d
        # implicit step
        w_ = brennan_schwartz_algorithm(alpha, beta, gamma, f, g_func(tau_array[nu+1], x, k))
        w_matrix[:, nu + 1] = w_
    return w_matrix, x, tau_array
