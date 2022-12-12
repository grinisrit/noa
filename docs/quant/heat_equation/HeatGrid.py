from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from enum import Enum
from typing import Dict, Optional


class Mode(Enum):
    MAIN = 'main net'
    CASH = 'additional cashed net'
    DIFF_T = 'sensitivity by time via finite differences'
    DIFF_X = 'sensitivity by x via finite differences'
    DIFF_C = 'sensitivity by sigma coef via finite differences'


# TODO: Add sensitivities by sigma via finite differences
class HeatGrid:
    _net_mod_map: Dict[Mode, Optional[ndarray]] = {
        Mode.MAIN: None,
        Mode.CASH: None,
        Mode.DIFF_T: None,
        Mode.DIFF_X: None,
        Mode.DIFF_C: None,
    }

    def __init__(self,
                 xSteps: int,
                 tSteps: int,
                 xLeft=-3.0,
                 xRight=3.0,
                 tMax=1.0):

        self.xSteps = xSteps
        self.tSteps = tSteps

        self._net = np.zeros((xSteps + 1, tSteps + 1))
        self.xGrid = np.linspace(xLeft, xRight, xSteps + 1)
        self.tGrid = np.linspace(0, tMax, tSteps + 1)
        self.dx = (xRight - xLeft) / xSteps
        self.dt = tMax / tSteps

    @property
    def net(self):
        return self._net

    @net.setter
    def net(self, external_net):
        if self._net.shape != (self.xSteps + 1, self.tSteps + 1):
            print(
                f'Warning: unexpected size of net {np.size(self.net)} instead of {(self.xSteps + 1, self.tSteps + 1)}')
        self._net = external_net

    def backup(self):
        self._net_mod_map[Mode.CASH] = self.net

    def clear(self):
        self._net = np.zeros((self.xSteps + 1, self.tSteps + 1))
        for mod in Mode:
            self._net_mod_map[mod] = None

    def diff_t(self):
        self._net_mod_map[Mode.DIFF_T] = (self._net[:, 1:] - self.net[:, :-1]) / self.dt

    def diff_x(self):
        self._net_mod_map[Mode.DIFF_X] = (self._net[1:, :] - self.net[:-1, :]) / self.dx

    def _get_mod_grid(self, mod):
        match mod:
            case Mode.DIFF_T:
                return self.xGrid, self.tGrid[:-1]
            case Mode.DIFF_X:
                return self.xGrid[:-1], self.tGrid
            case _:
                return self.xGrid, self.tGrid

    def plot(self, slice_num=5, mod=Mode.MAIN):
        net = self._net_mod_map[mod]
        x, t = self._get_mod_grid(mod)
        indexes = np.linspace(0, len(t)-1, slice_num)
        plt.style.use("dark_background")
        plt.figure(figsize=(15, 8))
        for i in indexes:
            plt.plot(x, net[:, int(i)], label=f"t = {round(t[int(i)], 2)}")
        plt.legend()
        plt.show()

    def plot3D(self, mod=Mode.MAIN):
        net = self._net_mod_map[mod]
        x, t = self._get_mod_grid(mod)
        surface = go.Surface(z=net, x=t, y=x)
        fig = go.Figure(surface)
        fig.update_layout(title='V(S,t)', autosize=False, width=1200, height=800,
                          margin=dict(l=65, r=50, b=65, t=90))
        fig.show()
