from docs.quant.heat_equation.HeatGrid import *
from docs.quant.utils.optlib.numerical_utils import crank_nickolson_scheme
from numba import njit


@njit()
def g_func(t, x):
    # q = 2*r/sigma**2 == 0 as r = 0
    return np.exp(t/4) * np.maximum(np.zeros_like(x), np.exp(-x/2) - np.exp(x/2))


class HeatSolver(HeatGrid):

    def __init__(self,
                 xSteps: int,
                 tSteps: int,
                 xLeft=-3.0,
                 xRight=3.0,
                 tMax=1.0):

        super().__init__(xSteps, tSteps, xLeft, xRight, tMax)

    def setBounds(self):
        # boundary conditions
        for i in (0, -1):
            self._net[i, :] = g_func(self.tGrid, self.xGrid[i])
        # initial conditions
        self._net[:, 0] = g_func(t=0, x=self.xGrid)
        self._net_mod_map[Mode.MAIN] = self.net

    def CN(self):
        self.setBounds()
        self._net = crank_nickolson_scheme(self.net, self.lamda)
        self._net_mod_map[Mode.MAIN] = self.net
