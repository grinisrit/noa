from containers import Underlying, Option
from functions import revert_time, transform_to_normal
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from enum import Enum


class Mode(Enum):
    HEAT = 'heat_equation_mode'
    NORM = 'normal_mode'
    BSM = 'black_scholes_mode'
    DIFF_BSM = 'difference with BSM'
    # TODO: add diff with transformed bsm

# TODO: add comments
class Grid:

    def __init__(self,
                 underlying: Underlying,
                 option: Option,
                 xSteps: int,
                 tSteps: int,
                 xLeft=-3.0,
                 xRight=3.0):

        self.underlying = underlying
        self.option = option
        self.xSteps = xSteps
        self.tSteps = tSteps
        self.xLeft = xLeft
        self.xRight = xRight

        self._net = np.zeros((self.xSteps+1, self.tSteps+1))
        self._netNorm = None
        self._netBSM = None
        self.xHeat = np.linspace(self.xLeft, self.xRight, self.xSteps+1)
        self._maturityHeat = self.option.maturity * self.underlying.volatility ** 2 / 2
        self.tHeat = np.linspace(0, self._maturityHeat, self.tSteps + 1)
        self.dxHeat = (self.xRight - self.xLeft) / self.xSteps
        self.dtHeat = self._maturityHeat / self.tSteps
        self.lamda = self.dtHeat / self.dxHeat ** 2
        self.q = 2 * self.underlying.interest / self.underlying.volatility ** 2

        self.xNorm = self.option.strike * np.exp(self.xHeat)
        self.tNorm = revert_time(self.tHeat, self.option.maturity, self.underlying.volatility)

    @property
    def net(self):
        return self._net

    @net.setter
    def net(self, external_net):
        if self._net.shape != (self.xSteps + 1, self.tSteps + 1):
            print(
                f'Warning: unexpected size of net {np.size(self.net)} instead of {(self.xSteps + 1, self.tSteps + 1)}')
        self._net = external_net

    def _make_normal_net(self):
        self._netNorm = transform_to_normal(K=self.option.strike,
                                            T=self.option.maturity,
                                            vol=self.underlying.volatility,
                                            q=self.q,
                                            x_array=self.xHeat,
                                            t_array=self.tHeat,
                                            net=self._net.copy())

    @property
    def netNorm(self):
        return self._net

    @netNorm.getter
    def netNorm(self):
        if self._netNorm is None:
            self._make_normal_net()
        return self._netNorm

    def _get_mod_grid(self, mod):
        global tmpX, tmpT, tmpNet
        match mod:
            case Mode.HEAT:
                tmpNet = self._net
                tmpX = self.xHeat
                tmpT = self.tHeat
            case Mode.NORM:
                if self._netNorm is None:
                    self._make_normal_net()
                tmpNet = self._netNorm
                tmpX = self.xNorm
                tmpT = self.tNorm
            case Mode.BSM:
                # TODO: fill netBSM if it isnan
                tmpNet = self._netBSM
                tmpX = self.xNorm
                tmpT = self.tNorm
            case Mode.DIFF_BSM:
                # TODO: fill netBSM if it isnan and transform backward
                tmpNet = self._netNorm - self._netBSM
                tmpX = self.xNorm
                tmpT = self.tNorm
        return tmpX, tmpT, tmpNet

    def plot(self, cut=True, slice_num=5, mod=Mode.HEAT, stopline=True):
        # TODO: add other functions
        x, t, net = self._get_mod_grid(mod)

        plt.style.use("dark_background")
        plt.figure(figsize=(15, 8))
        for i in np.linspace(0, self.tSteps - 1, slice_num):
            plt.plot(x, net[:, int(i)], label=f"t = {self.net[int(i)]}")
        if cut:
            plt.xlim(0, 2*self.option.strike)
        # TODO: add stopline
        plt.legend()
        plt.show()

    # TODO: add plot3D
    # def plot3D(self, cut=True, stopline=False):
    #     """ plotting 3D graph and slices by time """
    #     K = self.option.strike
    #     r = self.option.Underlying.interest
    #     T = self.option.maturity
    #     tempNet = self.net
    #     tempXgrid = self.xGrid
    #     if cut:
    #         leftBorder = K * 0.15
    #         rightBorder = K * 3
    #         leftBorder = np.where(self.xGrid < leftBorder)[0][-1]
    #         rightBorder = np.where(self.xGrid > rightBorder)[0][0]
    #         tempNet = self.net[leftBorder:rightBorder, :]
    #         tempXgrid = self.xGrid[leftBorder:rightBorder]
    #     if slice_num == 0:
    #         surface = go.Surface(z=tempNet, x=self.tGrid, y=tempXgrid)
    #         if stoppingline:
    #             stop_curve = self.asymptotic_stopping_line()
    #             curve = go.Scatter3d(z=10 * np.ones(len(stop_curve)), x=self.tGrid, y=stop_curve, mode="markers",
    #                                  marker=dict(size=3, color="green"))
    #             fig = go.Figure([surface, curve])
    #         else:
    #             fig = go.Figure([surface])
    #         fig.update_layout(title='V(S,t)', autosize=False, width=800, height=500,
    #                           margin=dict(l=65, r=50, b=65, t=90))
    #         fig.show()
    #     else:


