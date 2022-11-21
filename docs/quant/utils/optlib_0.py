import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import norm
from numba import njit


def bsm_value(self, currentTime=0):
    d1, d2 = d_calc(self.price, 0, self.strike, self.maturity, self.volatility, self.interest)
    if self.call:
        self.value = - self.strike * np.exp(-self.interest * (self.maturity - currentTime)) * norm.cdf(d2) + \
                     self.price * norm.cdf(d1)
    else:
        self.value = self.strike * np.exp(-self.interest * (self.maturity - currentTime)) * norm.cdf(-d2) - \
                     self.price * norm.cdf(-d1)
    return self.value.copy()


class Grid:
    """
    Assumes, that parameters of the grid (last 4 params) are set as for Heat Equation.
    """""
    def __init__(self,
                 xSteps: int,
                 tSteps: int,
                 xLeft=-3.0,
                 xRight=3.0):

        self.xSteps = xSteps
        self.tSteps = tSteps
        self.xLeft = xLeft
        self.xRight = xRight
        self.option = None

        self._net = np.zeros((self.xSteps + 1, self.tSteps + 1))
        self.net_add = np.zeros_like(self._net)
        self.xGrid = np.linspace(self.xLeft, self.xRight, self.xSteps + 1)
        self.sGrid = None
        self.timeGrid = None
        self.tBorder = None
        self.tGrid = None
        self.dx = None
        self.dt = None
        self.lamda = None
        self.q = None
        self.vega = None

    def _createGrid(self):
        self.sGrid = self.option.strike * np.exp(self.xGrid)
        self.tBorder = self.option.maturity * self.option.Underlying.volatility ** 2 / 2
        self.tGrid = np.linspace(0, self.tBorder, self.tSteps + 1)
        self.timeGrid = revert_time(self.tGrid, self.option.maturity, self.option.volatility)
        self.dx = (self.xRight - self.xLeft) / self.xSteps
        self.dt = self.tBorder / self.tSteps
        self.lamda = self.dt / self.dx ** 2
        self.q = 2 * self.option.Underlying.interest / self.option.Underlying.volatility ** 2

    def addOption(self, option: Option):
        self.option = option
        self._createGrid()

    @property
    def net(self):
        return self._net

    @net.setter
    def net(self, external_net):
        if self._net.shape != (self.xSteps + 1, self.tSteps + 1):
            print(
                f'Warning: unexpected size of net {np.size(self.net)} instead of {(self.xSteps + 1, self.tSteps + 1)}')
        self._net = external_net

    # TODO: add meshgrid
    def bsm(self, unsafe=False):
        p = 1 if self.option.call else -1
        self.net_add[:, 0] = np.maximum(p * (self.sGrid - self.option.strike), np.zeros_like(self.sGrid))
        for i in np.arange(1, len(self.tGrid)):
            d1, d2 = d_calc(self.sGrid, self.timeGrid[i], self.option.strike, self.option.maturity, self.option.volatility, self.option.interest)
            self.net_add[:, i] = - p * self.option.strike * np.exp(-self.option.interest * (self.option.maturity - self.timeGrid[i])) * norm.cdf(p*d2) + p * self.option.price * norm.cdf(p*d1)
        if unsafe:
            self._net = self.net_add

    def setBoundsPut(self):
        q = self.q
        for i in np.arange(self.tSteps + 1):
            self.net[self.xSteps, i] = g_func(q, self.tGrid[i], self.xGrid[-1])
            self.net[0, i] = g_func(q, self.tGrid[i], self.xGrid[0])
        for i in np.arange(self.xSteps + 1):
            self.net[i, 0] = g_func(q, self.tGrid[0], self.xGrid[i])

    def toNormal(self, obj_mutation=False):
        tmpNet = self.net.copy()
        for n in np.arange(len(self.tGrid)):
            for k in np.arange(len(self.xGrid)):
                tmpNet[k, n] = self.option.strike * np.exp((1 - self.q) * self.xGrid[k] / 2 - (((self.q - 1) ** 2) / 4 + self.q) * self.tGrid[n]) * tmpNet[k, n]
        t_array = np.array([self.option.maturity - 2 * t / self.option.Underlying.volatility ** 2 for t in self.tGrid])
        S_array = np.array([self.option.strike * np.exp(x) for x in self.xGrid])
        if obj_mutation:
            self.xGrid = S_array
            self.tGrid = t_array
            self.net = tmpNet
        else:
            return S_array, t_array, tmpNet

    # add backward transformation toDefault

    def plot(self, slice_num=0, cut=False, border=0.15, stoppingline=True):
        """ plotting 3D graph and slices by time """
        K = self.option.strike
        r = self.option.Underlying.interest
        T = self.option.maturity
        tempNet = self.net
        tempXgrid = self.xGrid
        if cut:
            leftBorder = K * border
            rightBorder = K * 3
            leftBorder = np.where(self.xGrid < leftBorder)[0][-1]
            rightBorder = np.where(self.xGrid > rightBorder)[0][0]
            tempNet = self.net[leftBorder:rightBorder, :]
            tempXgrid = self.xGrid[leftBorder:rightBorder]

        if slice_num == 0:
            surface = go.Surface(z=tempNet, x=self.tGrid, y=tempXgrid)
            if stoppingline:
                # curve = go.Scatter3d(z=[(1 - tau ** 1.4) * self.option.strike / 4 for tau in self.tGrid],
                #                      x=self.tGrid,
                #                      y=np.array(tuple(map(lambda tmp : early_exercise(K, r, T, tmp), self.tGrid))),
                #                      mode="markers",
                #                      marker=dict(
                #                          size=3,
                #                          color="green")
                #                      )
                # fig = go.Figure([surface, curve])
                fig = go.Figure([surface])
            else:
                fig = go.Figure([surface])

            fig.update_layout(title='V(S,t)', autosize=False,
                              width=800, height=500,
                              margin=dict(l=65, r=50, b=65, t=90))
            fig.show()
        else:
            plt.style.use("Solarize_Light2")
            plt.figure(figsize=(15, 8))
            for i in np.linspace(0, len(self.tGrid) - 1, slice_num):
                plt.plot(tempXgrid, tempNet[:, int(i)], label=f"t = {self.tGrid[int(i)]}")
                if stoppingline:
                    plt.axvline(early_exercise(K, r, T, self.tGrid[int(i)]), color='green')
            plt.legend()
            plt.show()

    def BrennanSchwartz(self):
        n = self.xSteps
        # matrix for explicit step
        B = np.zeros((n + 1, n + 1))
        for i in range(1, n + 1):
            B[i, i] = 1 - self.lamda
            B[i, i - 1] = self.lamda / 2
            B[i - 1, i] = self.lamda / 2
        # diagonals of the A matrix (implicit step):
        alpha = (1 + self.lamda) * np.ones(n - 1)
        beta = -0.5 * self.lamda * np.ones(n - 2)

        # brennan-shwartz algorythm
        for i in np.arange(0, self.tSteps):
            # setting early execution curve
            time = self.tGrid[i + 1]
            g = np.array(tuple(map(lambda x: g_func(self.q, time, x), self.xGrid)))
            # explicit step
            f_hat = self.net[:, i]
            f = np.dot(B, f_hat).copy()
            f[0] = f[0] + 0.5 * self.lamda * (g_func(self.q, self.tGrid[i], self.xGrid[0]) + g[n])
            #f[n] = f_hat[n] + self.lamda * g[n] / 2
            # implicit step
            solution = brennan_schwartz(alpha, beta, beta, f[1:n], g)
            self.net[1:n, i + 1] = solution


@njit()
def put_func(q, t, x):
    return np.exp(t * (q + 1)**2 / 4) * np.maximum(np.zeros_like(x), np.exp((q - 1)*x/2) - np.exp((q + 1)*x/2))


@njit()
def call_func(q, t, x):
    return np.exp((q + 1)**2 * t/4) * np.maximum(np.zeros_like(x), np.exp((q + 1)*x/2) - np.exp((q - 1)*x/2))


@njit()
def d_calc(S, t, K, T, sigma, r):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)
    return d1, d2

@njit()
def revert_time(t_array, T, volatility):
    return np.array([T - 2 * t / volatility ** 2 for t in t_array])

# @njit
# def g_func(q, t, x):
#     return np.exp(t * (q + 1)**2 / 4) * max(0, np.exp((q - 1)*x/2) - np.exp((q + 1)*x/2))


# @njit
# def call_boundary_condition(q, t, x):
#     return np.exp((q + 1)**2 * t/4) * max(0, np.exp((q + 1)*x/2) - np.exp((q - 1)*x/2))


@njit
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
    P = np.zeros(n-1)
    Q = np.zeros(n)
    P[0] = -1*A[0][1]/A[0][0]
    Q[0] = -1*f[0]/A[0][0]
    for i in range(1, n):
        a = A[i][i-1]
        c = -1*A[i][i]
        if i != n-1:
            b = A[i][i+1]
            P[i] = b/(c-a*P[i-1])
        Q[i] = (f[i]+a*Q[i-1])/(c-a*P[i-1])

    # (backward walk)
    x = np.zeros(n)
    x[n-1] = Q[n-1]
    for i in range(n-2, -1, -1):
        x[i] = P[i]*x[i+1]+Q[i]
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


