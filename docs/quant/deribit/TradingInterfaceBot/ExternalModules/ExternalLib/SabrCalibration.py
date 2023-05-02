import datetime
import logging
from threading import Thread
import time

import numpy as np
import numba as nb

from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objs as go

from typing import List, Tuple

from docs.quant.deribit.TradingInterfaceBot.Utils.AvailableInstrumentType import InstrumentType
try:
    from ..AbstractExternal import AbstractExternal
    from ...InstrumentManager import AbstractInstrument
except ImportError:
    from docs.quant.deribit.TradingInterfaceBot.ExternalModules.AbstractExternal import AbstractExternal
    from docs.quant.deribit.TradingInterfaceBot.InstrumentManager.AbstractInstrument import AbstractInstrument
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# BLOCK WITH NUMBA FUNC. Dmitry Bazanov, Roland Grinis, Ivan Novikov.
# https://github.com/grinisrit/noa/blob/feature/quant/docs/quant/vol_calibration.ipynb
@nb.njit()
def normal_cdf(x):
    t = 1 / (1 + 0.2316419 * abs(x))
    summ = 0.319381530 * t - 0.356563782 * t ** 2 + 1.781477937 * t ** 3 - 1.821255978 * t ** 4 + 1.330274429 * t ** 5
    if x >= 0:
        return 1 - summ * np.exp(-abs(x) ** 2 / 2) / np.sqrt(2 * np.pi)
    else:
        return summ * np.exp(-abs(x) ** 2 / 2) / np.sqrt(2 * np.pi)


@nb.njit()
def black_scholes_pv(sigma, S0, K, T, r, is_call=True):
    d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    p = 1 if is_call else -1
    return p * S0 * normal_cdf(p * d1) - p * K * np.exp(-r * T) * normal_cdf(p * d2)


@nb.njit()
def black_scholes_vega(sigma, S0, K, T, r):
    d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    return S0 * np.sqrt(T) * np.exp(-d1 ** 2 / 2) / np.sqrt(2 * np.pi)


@nb.njit()
def black_scholes_delta(sigma, S0, K, T, r, is_call=True):
    d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    p = 1 if is_call else -1
    return p * normal_cdf(p * d1)


@nb.njit()
def black_scholes_gamma(sigma, S0, K, T, r):
    d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-d1 ** 2 / 2) / (np.sqrt(2 * np.pi) * S0 * np.sqrt(T) * sigma)


@nb.njit()
def g_impvol(V_mkt, sigma, T, S0, K, r, is_call=True):
    return V_mkt - black_scholes_pv(sigma, S0, K, T, r, is_call)


@nb.njit()
def g_impvol_prime(sigma, T, S0, K, r):
    return -black_scholes_vega(sigma, S0, K, T, r)


@nb.njit()
def implied_vol(V_mkt, S0, K, T, r, is_call=True, tol=10 ** -4, sigma_l=10 ** -8, sigma_r=2):
    if g_impvol(V_mkt, sigma_l, T, S0, K, r, is_call) * \
            g_impvol(V_mkt, sigma_r, T, S0, K, r, is_call) > 0:
        # print('no zero at the initial interval')
        return 0.
    else:
        sigma = (sigma_l + sigma_r) / 2
        epsilon = g_impvol(V_mkt, sigma, T, S0, K, r, is_call)
        grad = g_impvol_prime(sigma, T, S0, K, r)
        while abs(epsilon) > tol:
            if abs(grad) > 1e-6:
                sigma -= epsilon / grad
                if sigma > sigma_r or sigma < sigma_l:
                    sigma = (sigma_l + sigma_r) / 2
                    if g_impvol(V_mkt, sigma_l, T, S0, K, r, is_call) * epsilon > 0:
                        sigma_l = sigma
                    else:
                        sigma_r = sigma
                    sigma = (sigma_l + sigma_r) / 2
            else:
                if g_impvol(V_mkt, sigma_l, T, S0, K, r, is_call) * epsilon > 0:
                    sigma_l = sigma
                else:
                    sigma_r = sigma
                sigma = (sigma_l + sigma_r) / 2

            epsilon = g_impvol(V_mkt, sigma, T, S0, K, r, is_call)
            grad = g_impvol_prime(sigma, T, S0, K, r)
        return sigma


@nb.jit()
def get_implied_vols(forward, maturity, strikes, forward_values, is_call=True):
    f, T, Ks = forward, maturity, strikes
    n = len(Ks)
    ivols = np.zeros(n, dtype=np.float64)
    for index in range(n):
        K = Ks[index]
        V_mkt = forward_values[index]
        ivols[index] = implied_vol(V_mkt, f, K, T, r=0., is_call=is_call)
    return ivols


@nb.njit()
def jacobian_sabr(forward, maturity, strikes, backbone, params):
    f, T, Ks = forward, maturity, strikes
    n = len(Ks)
    beta = backbone
    alpha, rho, v = params[0], params[1], params[2]
    ddalpha = np.zeros(n, dtype=np.float64)
    ddv = np.zeros(n, dtype=np.float64)
    ddrho = np.zeros(n, dtype=np.float64)
    for index in range(n):
        K = Ks[index]
        x = np.log(f / K)
        I_H = (
                alpha ** 2 * (K * f) ** (beta - 1) * (1 - beta) ** 2 / 24
                + alpha * beta * rho * v * (K * f) ** (beta / 2 - 1 / 2) / 4
                + v ** 2 * (2 - 3 * rho ** 2) / 24
        )

        dI_H_alpha = (
                alpha * (K * f) ** (beta - 1) * (1 - beta) ** 2 / 12
                + beta * rho * v * (K * f) ** (beta / 2 - 1 / 2) / 4
        )

        dI_H_rho = (
                alpha * beta * v * (K * f) ** (beta / 2 + -1 / 2) / 4 - rho * v ** 2 / 4
        )

        dI_h_v = (
                alpha * beta * rho * (K * f) ** (beta / 2 + -1 / 2) / 4
                + v * (2 - 3 * rho ** 2) / 12
        )

        if x == 0.0:
            I_B = alpha * K ** (beta - 1)
            B_alpha = K ** (beta - 1)
            B_v = 0.0
            B_rho = 0.0

        elif v == 0.0:
            I_B = alpha * (1 - beta) * x / (f ** (1 - beta) - (K ** (1 - beta)))
            B_alpha = (beta - 1) * x / (K ** (1 - beta) - f ** (1 - beta))
            B_v = 0.0
            B_rho = 0.0

        elif beta == 1.0:
            z = v * x / alpha
            sqrt = np.sqrt(1 - 2 * rho * z + z ** 2)
            I_B = v * x / (np.log((sqrt + z - rho) / (1 - rho)))
            B_alpha = (
                    v * x * z / (alpha * sqrt * np.log((rho - z - sqrt) / (rho - 1)) ** 2)
            )
            B_v = (
                    x
                    * (alpha * sqrt * np.log((rho - z - sqrt) / (rho - 1)) - v * x)
                    / (alpha * sqrt * np.log((rho - z - sqrt) / (rho - 1)) ** 2)
            )
            B_rho = (
                    v
                    * x
                    * ((rho - 1) * (z + sqrt) + (-rho + z + sqrt) * sqrt)
                    / (
                            (rho - 1)
                            * (-rho + z + sqrt)
                            * sqrt
                            * np.log((rho - z - sqrt) / (rho - 1)) ** 2
                    )
            )

        elif beta < 1.0:
            z = v * (f ** (1 - beta) - K ** (1 - beta)) / (alpha * (1 - beta))
            sqrt = np.sqrt(1 - 2 * rho * z + z ** 2)
            I_B = v * (-(K ** (1 - beta)) + f ** (1 - beta)) / (alpha * (1 - beta))
            B_alpha = (
                    v * x * z / (alpha * sqrt * np.log((rho - z - sqrt) / (rho - 1)) ** 2)
            )
            B_v = -v * x * ((-rho * z / v + z ** 2 / v) / sqrt + z / v) / (
                    (-rho + z + sqrt) * np.log((-rho + z + sqrt) / (1 - rho)) ** 2
            ) + x / np.log((-rho + z + sqrt) / (1 - rho))

            B_rho = (
                    -v
                    * x
                    * (1 - rho)
                    * ((-z / sqrt - 1) / (1 - rho) + (-rho + z + sqrt) / (1 - rho) ** 2)
                    / ((-rho + z + sqrt) * np.log((-rho + z + sqrt) / (1 - rho)) ** 2)
            )

        sig_alpha = B_alpha * (1 + I_H * T) + dI_H_alpha * I_B * T
        sig_v = B_v * (1 + I_H * T) + dI_h_v * I_B * T
        sig_rho = B_rho * (1 + I_H * T) + dI_H_rho * I_B * T

        ddalpha[index] = sig_alpha
        ddv[index] = sig_v
        ddrho[index] = sig_rho

    return ddalpha, ddrho, ddv


@nb.njit()
def vol_sabr(forward, maturity, strikes, backbone, params, compute_risk=False):
    f, T, Ks = forward, maturity, strikes
    n = len(Ks)
    beta = backbone
    alpha, rho, v = params[0], params[1], params[2]

    if compute_risk:
        dsigma_dalphas, dsigma_drhos, dsigma_dvs = jacobian_sabr(
            forward,
            maturity,
            strikes,
            backbone,
            params)

    sigmas = np.zeros(n, dtype=np.float64)

    if compute_risk:
        deltas = np.zeros(n, dtype=np.float64)
        gammas = np.zeros(n, dtype=np.float64)
        vegas = np.zeros(n, dtype=np.float64)
        regas = np.zeros(n, dtype=np.float64)
        segas = np.zeros(n, dtype=np.float64)

    for index in range(n):
        K = Ks[index]
        x = np.log(f / K)
        dIH1dF = alpha ** 2 * (K * f) ** (beta - 1) * (1 - beta) ** 2 * (beta - 1) / (
                24 * f
        ) + alpha * beta * rho * v * (K * f) ** (beta / 2 - 1 / 2) * (
                         beta / 2 - 1 / 2
                 ) / (
                         4 * f
                 )

        I_H_1 = (
                alpha ** 2 * (K * f) ** (beta - 1) * (1 - beta) ** 2 / 24
                + alpha * beta * rho * v * (K * f) ** (beta / 2 + -1 / 2) / 4
                + v ** 2 * (2 - 3 * rho ** 2) / 24
        )

        if x == 0.0:
            I_B_0 = K ** (beta - 1) * alpha
            dI_B_0_dF = 0.0

        elif v == 0.0:
            I_B_0 = alpha * (1 - beta) * x / (-(K ** (1 - beta)) + f ** (1 - beta))
            dI_B_0_dF = (
                    alpha
                    * (beta - 1)
                    * (
                            f * (K ** (1 - beta) - f ** (1 - beta))
                            - f ** (2 - beta) * (beta - 1) * x
                    )
                    / (f ** 2 * (K ** (1 - beta) - f ** (1 - beta)) ** 2)
            )

        else:
            if beta == 1.0:
                z = v * x / alpha
                sqrt = np.sqrt(1 - 2 * rho * z + z ** 2)

                dI_B_0_dF = (
                        v
                        * (alpha * sqrt * np.log((rho - z - sqrt) / (rho - 1)) - v * x)
                        / (alpha * f * sqrt * np.log((rho - z - sqrt) / (rho - 1)) ** 2)
                )

            elif beta < 1.0:
                z = v * (f ** (1 - beta) - K ** (1 - beta)) / (alpha * (1 - beta))
                sqrt = np.sqrt(1 - 2 * rho * z + z ** 2)
                dI_B_0_dF = (
                        v
                        * (
                                alpha
                                * f
                                * (-rho + z + sqrt)
                                * sqrt
                                * np.log((rho - z - sqrt) / (rho - 1))
                                + f ** (2 - beta) * v * x * (rho - z - sqrt)
                        )
                        / (
                                alpha
                                * f ** 2
                                * (-rho + z + sqrt)
                                * sqrt
                                * np.log((rho - z - sqrt) / (rho - 1)) ** 2
                        )
                )

            I_B_0 = v * x / (np.log((sqrt + z - rho) / (1 - rho)))

        sigma = I_B_0 * (1 + I_H_1 * T)
        sigmas[index] = sigma

        if compute_risk:
            dsigma_df = dI_B_0_dF * (1 + I_H_1 * T) + dIH1dF * I_B_0 * T
            vega_bsm = black_scholes_vega(sigma, f, K, T, r=0.)
            delta_bsm = black_scholes_delta(sigma, f, K, T, r=0., is_call=True)

            dsigma_dalpha, dsigma_drho, dsigma_dv = \
                dsigma_dalphas[index], dsigma_drhos[index], dsigma_dvs[index]
            # call delta, for put don't forget to take 1-delta
            deltas[index] = delta_bsm + vega_bsm * (dsigma_df + dsigma_dalpha * rho * v / f ** beta)

            # todo: SABR gamma
            gammas[index] = black_scholes_gamma(sigma, f, K, T, r=0.)

            vegas[index] = vega_bsm * (dsigma_dalpha + dsigma_df * rho * f ** beta / v)
            regas[index] = vega_bsm * dsigma_drho
            segas[index] = vega_bsm * dsigma_dv

    greeks = (deltas, gammas, vegas, regas, segas) if compute_risk else None

    return sigmas, greeks


@nb.njit()
def np_clip(a, a_min, a_max):
    if a < a_min:
        out = a_min
    elif a > a_max:
        out = a_max
    else:
        out = a
    return out


@nb.njit()
def calibrate_sabr(forward, maturity, strikes, implied_vols, backbone, initial_params=np.array([1., -0.1, 0.0])):
    def clip_params(params):
        eps = 1e-4
        alpha, rho, v = params[0], params[1], params[2]
        alpha = np_clip(alpha, eps, 50.0)
        v = np_clip(v, eps, 50.0)
        rho = np_clip(rho, -1.0 + eps, 1.0 - eps)
        sabr_params = np.array([alpha, rho, v])
        return sabr_params

    def get_residuals(params):
        J = np.stack(
            jacobian_sabr(forward, maturity, strikes, backbone, params)
        )
        iv, _ = vol_sabr(
            forward,
            maturity,
            strikes,
            backbone,
            params,
            compute_risk=False
        )
        weights = np.ones_like(strikes)
        weights = weights / np.sum(weights)
        res = iv - implied_vols
        return res * weights, J @ np.diag(weights)

    def levenberg_marquardt(Niter, f, proj, x0):
        x = x0.copy()

        mu = 100.0
        nu1 = 2.0
        nu2 = 2.0

        res, J = f(x)
        F = np.linalg.norm(res)

        result_x = x
        result_error = F
        eps = 1e-5

        for i in range(Niter):
            multipl = J @ J.T
            I = np.diag(np.diag(multipl)) + 1e-5 * np.eye(len(x))
            dx = np.linalg.solve(mu * I + multipl, J @ res)
            x_ = proj(x - dx)
            res_, J_ = f(x_)
            F_ = np.linalg.norm(res_)
            if F_ < F:
                x, F, res, J = x_, F_, res_, J_
                mu /= nu1
                result_error = F
            else:
                i -= 1
                mu *= nu2
                continue
            if F < eps:
                break
            result_x = x
        return result_x, result_error

    return levenberg_marquardt(500, get_residuals, clip_params, initial_params)


# END BLOCK WITH NUMBA FUNC


class SabrCalibration(AbstractExternal, Thread):
    def __init__(self):
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        Thread.__init__(self)

        self.test_x_values = [1, 2, 3, 4]

        self.test_y_values = [0, 0, 0, 1]

        # Thread with Dash App
        self.dash_th = Thread(target=self.main).start()

    def main(self):
        app = Dash(__name__)

        app.layout = html.Div(
            [
                html.Div([
                    html.Div([
                        dcc.Graph(id='live-graph-index', animate=False)],
                        className='six columns'),
                ], className='row'),

                dcc.Interval(
                    id='interval',
                    interval=20  # in millisecond 1*1000= 1 second
                ),
            ]

        )
        global web_app
        web_app = self

        @app.callback(Output('live-graph-index', 'figure'),
                      [Input('interval', 'n_intervals')])

        def update_line_chart(self):
            _real_implied_vols_fig = go.Scatter(x=web_app.test_x_values, y=web_app.test_y_values)
            # print(f"{web_app.test_x_values=}, {web_app.test_y_values=}")
            # _real_implied_vols_fig = go.Scatter(x=np.random.randint(0, 10, 5), y=[1, 2, 3, 4, 5])
            return {'data': [_real_implied_vols_fig],
                    'layout': go.Layout(title='Volatility Surface', )}

        app.run_server(debug=False)

    def update_plotting(self):
        self.test_y_values = [np.random.randint(0, 10, 5)]
        self.test_x_values = [[1, 2, 3, 4, 5]]

    def collect_data_from_instruments(self):
        recorded_instruments: Tuple[AbstractInstrument] = tuple(
            self.strategy.data_provider.instrument_manager.managed_instruments.values())
        instruments_maturities = np.array([instrument.get_raw_instrument_maturity() for instrument in recorded_instruments])
        instruments_maturities = instruments_maturities[instruments_maturities != np.array(None)]
        unique_maturities = np.unique(instruments_maturities)
        # TODO: make better! Made like this only to see result!!!
        if len(unique_maturities) > 1:
            logging.error("Currently unable to process more then 1 maturity")
            return
        for maturity in unique_maturities:
            available_instruments = []
            for instrument in recorded_instruments:
                if instrument.get_raw_instrument_maturity() == maturity:
                    available_instruments.append(instrument)

            extract_future = list(filter(lambda _: _.instrument_type == InstrumentType.FUTURE, available_instruments))[0]
            if extract_future.last_trades[-1] is None:
                logging.error("No trades with underlying future. Go to next maturity")
                continue

            call_options = list(
                filter(lambda _: _.instrument_type == InstrumentType.CALL_OPTION, available_instruments))
            put_options = list(
                filter(lambda _: _.instrument_type == InstrumentType.PUT_OPTION, available_instruments))


            _call_strikes = np.array([item.instrument_strike for item in call_options])
            _put_strikes = np.array([item.instrument_strike for item in put_options])

            _call_prices = np.array([item.last_trades[-1] for item in call_options])
            _put_prices = np.array([item.last_trades[-1] for item in put_options])


            _clean_call_strikes = []
            _clean_call_prices = []
            for batch in zip(_call_prices, _call_strikes):
                # print("Call batch", batch)
                if (batch[0] is not None) and (batch[1] is not None):
                    _clean_call_prices.append(batch[0].trade_price)
                    _clean_call_strikes.append(batch[1])

            _clean_put_strikes = []
            _clean_put_prices = []
            for batch in zip(_put_prices, _put_strikes):
                # print("Put batch", batch)
                if (batch[0] is not None) and (batch[1] is not None):
                    _clean_put_prices.append(batch[0].trade_price)
                    _clean_put_strikes.append(batch[1])

            # Call recalculation for sabr calibration. in new thread
            Thread(target=self.test_calculation, args=(
                _clean_put_strikes,
                _clean_call_strikes,
                extract_future.instrument_maturity,
                extract_future.last_trades[-1].trade_price,
                _clean_put_prices,
                _clean_call_prices
            )).start()

    async def on_order_book_update(self, abstractInstrument: AbstractInstrument):
        print("SABR ORDER BOOK UPDATE")
        self.collect_data_from_instruments()

    async def on_trade_update(self, abstractInstrument: AbstractInstrument):
        print("SABR TRADE UPDATE")
        self.collect_data_from_instruments()

    async def on_tick_update(self, callback: dict):
        print("SABR TICK UPDATE")

    def test_calculation(self, put_strikes, call_strikes, T, forward, puts, calls):
        put_strikes = np.array(put_strikes)
        call_strikes = np.array(call_strikes)
        T = T
        forward = forward
        puts = np.array(puts)
        calls = np.array(calls)

        strikes = np.concatenate((put_strikes, call_strikes))
        implied_vols = np.concatenate(
            (get_implied_vols(forward, T, put_strikes, puts, is_call=False),
             get_implied_vols(forward, T, call_strikes, calls, is_call=True)
        )
        )
        # self.draw_axes.plot(strikes, 100 * implied_vols, '.C3', markersize=20, label='real')
        # self.draw_axes.set_title('Market skew', fontsize=15)
        # self.draw_axes.set_xlabel('Strikes', color='white', fontsize=15)
        # self.draw_axes.set_ylabel('IV(%)', color='white', fontsize=15)
        # self.draw_axes.grid()
        beta = 0.95

        calibrated_params, error = calibrate_sabr(
            forward,
            T,
            strikes,
            implied_vols,
            beta)

        test_strikes = np.linspace(strikes[0], strikes[-1], 100)
        test_iv, _ = vol_sabr(
            forward,
            T,
            test_strikes,
            beta,
            calibrated_params,
            compute_risk=False)

        self.test_x_values = list(test_strikes)
        self.test_y_values = list(test_iv)
        # self.draw_axes.plot(test_strikes, 100 * test_iv, 'C1', label='calibrated')
        #
        # self.draw_axes.legend(loc='lower right')
        # self.draw_object.show()


if __name__ == '__main__':
    sabr = SabrCalibration()
    # sabr.test_calculation()
    # sabr.update_plotting()
    while True:
        time.sleep(1)
        sabr.update_plotting()