import numpy as np
from scipy.stats import norm
from numba import njit
# TODO: add comments


@njit()
def revert_time(t_array, T, volatility):
    return np.array([T - 2 * t / volatility ** 2 for t in t_array])


@njit()
def transform_to_normal(K, q, t_heat, x_heat, net):
    for n in range(len(t_heat)):
        for k in range(len(x_heat)):
            net[k, n] = K * np.exp((1 - q) * x_heat[k] / 2 - (((q + 1) ** 2) / 4) * t_heat[n]) * net[k, n]
    return net


@njit()
def transform_to_heat(K, q, t_heat, x_heat, net):
    for n in range(len(t_heat)):
        for k in range(len(x_heat)):
            net[k, n] = net[k, n] / (K * np.exp((1 - q) * x_heat[k] / 2 - (((q + 1) ** 2) / 4) * t_heat[n]))
    return net


@njit()
def find_early_exercise(V, S_array, t_array, K, slice_num=0, tolerance=10 ** -5):
    stop_line_V = list()
    stop_line_S = list()
    if slice_num == 0:
        for i in range(len(t_array)):
            v_array = V[:, int(i)]
            stop = [(s, v) for v, s in zip(v_array, S_array) if v <= max(K - s + tolerance, 0)]
            stop_line_V.append(stop[-1][1])
            stop_line_S.append(stop[-1][0])
    else:
        for i in np.linspace(0, len(t_array) - 1, slice_num):
            stop = [(s, v) for v, s in zip(V[:, int(i)], S_array) if v <= max(K - s + tolerance, 0)]
            stop_line_V.append(stop[-1][1])
            stop_line_S.append(stop[-1][0])
    return stop_line_V, stop_line_S


def fill_bsm_dev(x_array, t_array, K, T, vol, r, call=True):
    tt, xx = np.meshgrid(t_array[1:], x_array)
    dtt = T - tt
    d1 = np.multiply((np.log(xx / K) + (r + vol ** 2 / 2) * dtt),  1 / (vol * np.sqrt(dtt)))
    d2 = d1 - vol * np.sqrt(dtt)
    p = 1 if call else -1

    V = np.empty((len(x_array), len(t_array)))
    V[:, 1:] = p*np.multiply(xx, norm.cdf(p*d1)) - p*K*np.multiply(np.exp(r*dtt), norm.cdf(p*d2))
    V[:, 0] = np.maximum(p * (np.array(x_array) - K), np.zeros_like(x_array))
    return V


def fill_vega(x_array, t_array, K, T, vol, r):
    tt, xx = np.meshgrid(t_array[1:], x_array)
    dtt = T - tt
    d1 = np.multiply((np.log(xx / K) + (r + vol ** 2 / 2) * dtt), 1 / (vol * np.sqrt(dtt)))

    V = np.zeros((len(x_array), len(t_array)))
    V[:, 1:] = np.multiply(np.multiply(xx, np.sqrt(dtt)), np.exp(-np.square(d1)/2) / np.sqrt(2*np.pi))
    return V

# def fill_vega_heat(x_array, t_array, K, T, vol, r):
#     tt, xx = np.meshgrid(t_array[1:], x_array)
#     d1 = np.multiply((xx + (2*r/vol**2 + 1) * tt), 1 / (np.sqrt(2 * tt)))
#
#     V = np.zeros((len(x_array), len(t_array)))
#     V[:, 1:] = np.multiply(np.multiply(K * np.exp(xx) / vol, np.sqrt(2 * tt)), np.exp(-np.square(d1) / 2) / np.sqrt(2 * np.pi))
#     return V
