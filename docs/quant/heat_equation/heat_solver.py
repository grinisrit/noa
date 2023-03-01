import types
from typing import Union, Callable

from docs.quant.heat_equation.heat_grid import *
from docs.quant.utils.optlib.numerical_utils import crank_nickolson_mod, brennan_schwartz_scheme_mod
from numba import njit
from numpy import ndarray


@njit()
def g_func(t, x):
    # q = 2*r/sigma**2 == 0 as r = 0
    return np.exp(t/4) * np.maximum(np.zeros_like(x), np.exp(-x/2) - np.exp(x/2))


class HeatSolver(HeatGrid):

    def __init__(self,
                 xSteps: int,
                 tSteps: int,
                 sigma=1.0,
                 xLeft=-3.0,
                 xRight=3.0,
                 tMax=1.0):

        super().__init__(xSteps, tSteps, xLeft, xRight, tMax)
        self.sigma_net = sigma * np.ones_like(self._net)
        self.lambda_ = self.sigma_net * self.dt / self.dx ** 2

    def set_sigma(self, sigma: Union[ndarray, Callable[[ndarray, ndarray], ndarray]]):
        if isinstance(sigma, ndarray):
            assert sigma.shape == (self.xSteps + 1, self.tSteps + 1)
            self.sigma_net = sigma
            self.lambda_ = sigma * self.dt / self.dx ** 2
        elif isinstance(sigma, types.FunctionType):
            xx, tt = np.meshgrid(self.xGrid, self.tGrid, indexing='ij')
            self.sigma_net = sigma(xx, tt)
            self.lambda_ = self.sigma_net * self.dt / self.dx ** 2

    def setBounds(self):
        # boundary conditions
        for i in (0, -1):
            self._net[i, :] = g_func(self.tGrid, self.xGrid[i])
        # initial conditions
        self._net[:, 0] = g_func(t=0, x=self.xGrid)
        self._net_mod_map[Mode.MAIN] = self.net

    def CN(self):
        self.setBounds()
        self._net = crank_nickolson_mod(self.net, self.lambda_)
        self._net_mod_map[Mode.MAIN] = self.net

    def BS(self, g_func):
        self.setBounds()
        self._net = brennan_schwartz_scheme_mod(self.net,
                                                self.lambda_,
                                                self.tGrid,
                                                self.xGrid,
                                                g_func=g_func)
        self._net_mod_map[Mode.MAIN] = self.net

    # TODO: manage with cashing
    def diff_sigma(self, shift=10**-5):
        if self._net_mod_map[Mode.CASH] is None:
            print('Please solve CN and make backup (CASH is empty)')
        else:
            sigma_net_shifted = self.sigma_net + shift
            lambda_shifted = sigma_net_shifted * self.dt / self.dx ** 2
            self._net_mod_map[Mode.DIFF_C] = \
                (crank_nickolson_mod(self.net, lambda_shifted) - self._net_mod_map[Mode.CASH]) / shift
