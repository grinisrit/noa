from numpy import ndarray
from docs.quant.utils.optlib.Containers import Underlying, Option
from docs.quant.utils.optlib.utils import *
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from enum import Enum
from typing import Dict, Optional
# TODO: add comments


class Mode(Enum):
    HEAT = 'Heat equation mode'
    NORM = 'Normal mode'
    BSM = 'Black-Scholes mode'
    DIFF_BSM = 'Difference with BSM'
    VEGA = 'Vega by BSM'
    VEGA_HEAT = 'Vega in Heat Equation coordinates'
    # SHIFT_PARAM = 'Grid with shifted params'


class Grid:
    _net_mod_map: Dict[Mode, Optional[ndarray]] = {
        Mode.NORM: None,
        Mode.BSM: None,
        Mode.VEGA: None
        # Mode.SHIFT_PARAM: None
    }

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

        self._net = np.zeros((self.xSteps + 1, self.tSteps + 1))
        self.xHeat = np.linspace(self.xLeft, self.xRight, self.xSteps + 1)
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
        self._net_mod_map[Mode.NORM] = transform_to_normal(K=self.option.strike,
                                                           q=self.q,
                                                           t_heat=self.tHeat,
                                                           x_heat=self.xHeat,
                                                           net=self._net.copy())

    def _make_bsm_net(self):
        if self._net_mod_map[Mode.BSM] is None:
            self._net_mod_map[Mode.BSM] = fill_bsm_dev(self.xNorm,
                                                       self.tNorm,
                                                       self.option.strike,
                                                       self.option.maturity,
                                                       self.underlying.volatility,
                                                       self.underlying.interest,
                                                       call=self.option.call)

    def _make_vega_net(self):
        if self._net_mod_map[Mode.VEGA] is None:
            self._net_mod_map[Mode.VEGA] = fill_vega(self.xNorm,
                                                     self.tNorm,
                                                     self.option.strike,
                                                     self.option.maturity,
                                                     self.underlying.volatility,
                                                     self.underlying.interest)

    def get_mod_net(self, mod=Mode.HEAT):
        match mod:
            case Mode.HEAT:
                return self._net

            case Mode.NORM:
                self._make_normal_net()
                return self._net_mod_map[Mode.NORM]

            case Mode.BSM:
                self._make_bsm_net()
                return self._net_mod_map[Mode.BSM]

            case Mode.DIFF_BSM:
                self._make_normal_net()
                self._make_bsm_net()
                return self._net_mod_map[Mode.NORM] - self._net_mod_map[Mode.BSM]

            case Mode.VEGA:
                self._make_vega_net()
                return self._net_mod_map[Mode.VEGA]

            case Mode.VEGA_HEAT:
                self._make_vega_net()
                return self._net_mod_map[Mode.VEGA]

            # case Mode.SHIFT_PARAM:
            #     if self._net_mod_map[Mode.SHIFT_PARAM] is None:
            #         warnings.warn('NetIsNone: this net has no content')
            #     else:
            #         return self._net_mod_map[Mode.SHIFT_PARAM]

    def get_mod_grid(self, mod=Mode.HEAT):
        if mod == Mode.HEAT or mod == Mode.VEGA_HEAT:
            return self.xHeat, self.tHeat, self.get_mod_net(mod)
        else:
            return self.xNorm, self.tNorm, self.get_mod_net(mod)

    # TODO: and cutting for other modes
    def plot(self, cut=False, lcoef=0, rcoef=2.4, slice_num=5, mod=Mode.HEAT, stopline=False):
        x, t, net = self.get_mod_grid(mod)
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

    def plot3D(self, cut=False, lcoef=0, rcoef=2.4, mod=Mode.HEAT, stopline=False):
        curve = go.Scatter3d(z=list(), x=list(), y=list())
        x, t, net = self.get_mod_grid(mod)
        surface = go.Surface(z=net, x=t, y=x)
        # if cut:
        # x, net = self._cut_net(x, net)
        if stopline:
            stop_V, stop_X = find_early_exercise(net, x, t, self.option.strike)
            curve = go.Scatter3d(z=stop_V[1:], x=t[1:], y=stop_X[1:], mode="markers",
                                 marker=dict(size=2, color="green"))
        fig = go.Figure([surface, curve])
        fig.update_layout(title='V(S,t)', autosize=False, width=1200, height=800,
                          margin=dict(l=65, r=50, b=65, t=90))
        if cut:
            fig.update_layout(
                scene=dict(yaxis=dict(nticks=4, range=[lcoef * self.option.strike, rcoef * self.option.strike])))
        fig.show()
