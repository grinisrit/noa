import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import norm
from numba import njit


class Underlying:

    def __init__(self,
                 price: float,
                 volatility: float,
                 interest=0.0):
        self.price = price
        self.interest = interest
        self.volatility = volatility
        # add simulation function


class Option:

    def __init__(self,
                 underlying: Underlying,
                 style: str,
                 call: bool,
                 strike: float,
                 maturity=1.0):

        self.Underlying = underlying
        self.style = style
        self.call = call
        self.strike = strike
        self.maturity = maturity
        self.value = None

    def BSM(self, currentTime=0):
        """
        Pricing by BSM only for European style Options
        """
        D1 = (np.log(self.Underlying.price / self.strike) + (
                    self.Underlying.volatility ** 2 / 2 + self.Underlying.interest) * (
                          self.maturity - currentTime)) / (
                         self.Underlying.volatility * np.sqrt(self.maturity - currentTime))
        D2 = (np.log(self.Underlying.price / self.strike) + (
                    -self.Underlying.volatility ** 2 / 2 + self.Underlying.interest) * (
                          self.maturity - currentTime)) / (
                         self.Underlying.volatility * np.sqrt(self.maturity - currentTime))
        if self.call:
            if currentTime == self.maturity:
                V = max(self.Underlying.price - self.strike, 0)
            else:
                V = - self.strike * np.exp(self.Underlying.interest * (currentTime - self.maturity)) * norm.cdf(D2) \
                    + self.Underlying.price * norm.cdf(D1)
        else:
            if currentTime == self.maturity:
                V = max(- self.Underlying.price + self.strike, 0)
            else:
                V = self.strike * np.exp(self.Underlying.interest * (currentTime - self.maturity)) * norm.cdf(-D2) \
                    - self.Underlying.price * norm.cdf(-D1)
        self.value = V


class Grid:
    """
    Assumes, that parameters of the grid (last 4 params) are set as for Heat Equation.
    """

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

        self.net = np.zeros((self.xSteps + 1, self.tSteps + 1))
        self.xGrid = np.linspace(self.xLeft, self.xRight, self.xSteps + 1)
        self.timeBorder = None
        self.tGrid = None
        self.dx = None
        self.dt = None
        self.lamda = None
        self.q = None
        self.vega = None

    def _createGrid(self):
        self.timeBorder = self.option.maturity * self.option.Underlying.volatility ** 2 / 2
        self.tGrid = np.linspace(0, self.timeBorder, self.tSteps + 1)
        self.dx = (self.xRight - self.xLeft) / self.xSteps
        self.dt = self.timeBorder / self.tSteps
        self.lamda = self.dt / self.dx ** 2
        self.q = 2 * self.option.Underlying.interest / self.option.Underlying.volatility ** 2

    def addOption(self, option: Option):
        self.option = option
        self._createGrid()

    def valuateBSM(self):  # Надо разобраться с путом и ненулевой ставкой
        self.xGrid = np.linspace(self.xLeft, self.xRight, self.xSteps + 1)
        self.tGrid = np.linspace(0, self.timeBorder, self.tSteps + 1)
        self.net = np.zeros((self.xSteps + 1, self.tSteps + 1))
        self.toNormal(True)
        K = self.option.strike
        vol = self.option.Underlying.volatility
        r = self.option.Underlying.interest
        T = self.option.maturity

        if self.option.call:
            p = 1
        else:
            p = -1
        for i in range(1, len(self.tGrid)):
            for j in range(len(self.xGrid)):
                t = self.tGrid[i]
                S = self.xGrid[j]
                D1 = (np.log(S / K) + (vol ** 2 / 2 + r) * (T - t)) / (vol * np.sqrt(T - t))
                D2 = (np.log(S / K) + (-vol ** 2 / 2 + r) * (T - t)) / (vol * np.sqrt(T - t))
                self.net[j, i] = p * S * norm.cdf(p * D1) - p * K * np.exp(r * (t - T)) * norm.cdf(p * D2)
        for j in range(len(self.xGrid)):
            self.net[j, 0] = max(p*(self.xGrid[j] - K), 0)

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

    def plot(self, slice_num=0, cut=False, stoppingline=True):
        """ plotting 3D graph and slices by time """
        K = self.option.strike
        r = self.option.Underlying.interest
        T = self.option.maturity
        tempNet = self.net
        tempXgrid = self.xGrid
        if cut:
            leftBorder = K * 0.15
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
            f_hat = self.net[:, i].copy()
            f_hat[0] = f_hat[0] + self.lamda * g[0] / 2
            f_hat[n] = f_hat[n] + self.lamda * g[n] / 2
            f = np.dot(B, f_hat)
            # implicit step
            solution = brennan_schwartz(alpha, beta, beta, f[1:n], g)
            self.net[1:n, i + 1] = solution


@njit
def g_func(q, t, x):
    return np.exp(t * (q + 1)**2 / 4) * max(0, np.exp((q - 1)*x/2) - np.exp((q + 1)*x/2))


@njit
def call_boundary_condition(q, t, x):
    return np.exp((q + 1)**2 * t/4) * max(0, np.exp((q + 1)*x/2) - np.exp((q - 1)*x/2))


@njit
def early_exercise(K, r, T, time):
    return K * np.exp(r*(-T + time))


@njit
def scalar_walk(A, B):
    """
    three-dots scalar walk algorythm for solving Linear systems with bidiagonal matrix
    :param A: bidiagonal matrix
    :param B: array
    :return: solution for system A*x=B
    """
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


