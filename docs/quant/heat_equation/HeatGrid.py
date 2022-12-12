from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from enum import Enum
from typing import Dict, Optional


class Mode(Enum):
    MAIN = 'main net'
    CASH = 'additional cashed net'
    DIFF = 'sensitivities by finite differences'


class HeatGrid:
    _net_mod_map: Dict[Mode, Optional[ndarray]] = {
        Mode.MAIN: None,
        Mode.CASH: None,
        Mode.DIFF: None,
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
        self.lamda = self.dt / self.dx ** 2

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

    # TODO: Add sensitivities by t, x, sigma via finite differences

    def clear(self):
        self._net = np.zeros((self.xSteps + 1, self.tSteps + 1))
        for mod in Mode:
            self._net_mod_map[mod] = None

    def plot(self, slice_num=5, mod=Mode.MAIN):
        net = self._net_mod_map[mod]
        indexes = np.linspace(0, self.tSteps - 1, slice_num)
        plt.style.use("dark_background")
        plt.figure(figsize=(15, 8))
        for i in indexes:
            plt.plot(self.xGrid, net[:, int(i)], label=f"t = {round(self.tGrid[int(i)], 2)}")
        plt.legend()
        plt.show()

    def plot3D(self, mod=Mode.MAIN):
        net = self._net_mod_map[mod]
        surface = go.Surface(z=net, x=self.tGrid, y=self.xGrid)
        fig = go.Figure(surface)
        fig.update_layout(title='V(S,t)', autosize=False, width=1200, height=800,
                          margin=dict(l=65, r=50, b=65, t=90))
        fig.show()
