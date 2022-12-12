from docs.quant.heat_equation.heat_grid import *
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
                 sigma=1,
                 xLeft=-3.0,
                 xRight=3.0,
                 tMax=1.0):

        super().__init__(xSteps, tSteps, xLeft, xRight, tMax)
        self.lambda_ = sigma * self.dt / self.dx ** 2
        # TODO: sigma == sigma(xx, tt)

    def setBounds(self):
        # boundary conditions
        for i in (0, -1):
            self._net[i, :] = g_func(self.tGrid, self.xGrid[i])
        # initial conditions
        self._net[:, 0] = g_func(t=0, x=self.xGrid)
        self._net_mod_map[Mode.MAIN] = self.net

    def CN(self):
        self.setBounds()
        self._net = crank_nickolson_scheme(self.net, self.lambda_)
        self._net_mod_map[Mode.MAIN] = self.net

    # TODO: manage with cashing and if before CN and without backup
    def diff_sigma(self, shift_percent=0.001):
        if self._net_mod_map[Mode.CASH] is None:
            print('Please solve CN and make backup (CASH is empty)')
        else:
            self._net_mod_map[Mode.DIFF_C] = \
                (crank_nickolson_scheme(self.net, self.lambda_*(1 + shift_percent)) - self._net_mod_map[Mode.CASH]) / (shift_percent*self.lambda_)
