from docs.quant.utils.optlib.containers import Underlying, Option
from docs.quant.utils.optlib.utils import revert_time, transform_to_normal, find_early_exercise, fill_bsm, fill_bsm_dev
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from enum import Enum
# TODO: add comments


class Mode(Enum):
    HEAT = 'heat_equation_mode'
    NORM = 'normal_mode'
    BSM = 'black_scholes_mode'
    DIFF_BSM = 'difference with BSM'


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

    def _make_bsm_net(self):
        self._netBSM = fill_bsm_dev(self.xNorm,
                                    self.tNorm,
                                    self.option.strike,
                                    self.option.maturity,
                                    self.underlying.volatility,
                                    self.underlying.interest,
                                    call=self.option.call)

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
                if self._netBSM is None:
                    self._make_bsm_net()
                tmpNet = self._netBSM
                tmpX = self.xNorm
                tmpT = self.tNorm
            case Mode.DIFF_BSM:
                if self._netNorm is None:
                    self._make_normal_net()
                if self._netBSM is None:
                    self._make_bsm_net()
                tmpNet = self._netNorm - self._netBSM
                tmpX = self.xNorm
                tmpT = self.tNorm
        return tmpX, tmpT, tmpNet

    def plot(self, cut=True, lcoef=0, rcoef=2.4, slice_num=5, mod=Mode.HEAT, stopline=False):
        x, t, net = self._get_mod_grid(mod)
        indexes = np.linspace(0, self.tSteps - 1, slice_num)
        plt.style.use("dark_background")
        plt.figure(figsize=(15, 8))
        for i in indexes:
            plt.plot(x, net[:, int(i)], label=f"t = {round(t[int(i)], 2)}")
        if stopline:
            stop_V, stop_X = find_early_exercise(net, x, t, self.option.strike, slice_num)
            plt.vlines(stop_X, ymin=0, ymax=stop_V, color='violet')
        if cut:
            plt.ylim(0, (rcoef-1) * self.option.strike)
            plt.xlim(lcoef * self.option.strike, rcoef * self.option.strike)
        plt.legend()
        plt.show()

    # TODO: and refactor cutting for heat mode
    def _cut_net(self, x, net, lcoef=0, rcoef=2.5):
        left = lcoef * self.option.strike
        right = self.option.strike * rcoef
        leftBorder = np.where(x > left)[0][0]
        rightBorder = np.where(x > right)[0][0]
        net = net[leftBorder:rightBorder, :]
        x = x[leftBorder:rightBorder]
        return x, net

    def plot3D(self, cut=True, mod=Mode.HEAT, stopline=False):
        """ plotting 3D graph and slices by time """
        curve = go.Scatter3d(z=list(), x=list(), y=list())
        x, t, net = self._get_mod_grid(mod)
        if cut:
            x, net = self._cut_net(x, net)
        if stopline:
            stop_V, stop_X = find_early_exercise(net, x, t, self.option.strike)
            surface = go.Surface(z=net, x=x, y=t)
            curve = go.Scatter3d(z=stop_V[1:], x=stop_X[1:], y=t[1:], mode="markers", marker=dict(size=2, color="green"))
        surface = go.Surface(z=net, x=t, y=x)
        fig = go.Figure([surface, curve])
        fig.update_layout(title='V(S,t)', autosize=False, width=1200, height=800,
                          margin=dict(l=65, r=50, b=65, t=90))
        fig.show()
