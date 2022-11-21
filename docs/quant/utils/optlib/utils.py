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
def find_early_exercise(V, S_array, t_array, K, slice_num=0, tolerance=10**-5):
    stop_line_V = list()
    stop_line_S = list()
    if slice_num == 0:
        for i in range(len(t_array)):
            v_array = V[:, int(i)]
            stop = [(s, v) for v, s in zip(v_array, S_array) if v <= max(K-s+tolerance, 0)]
            stop_line_V.append(stop[-1][1])
            stop_line_S.append(stop[-1][0])
    else:
        for i in np.linspace(0, len(t_array) - 1, slice_num):
            stop = [(s, v) for v, s in zip(V[:, int(i)], S_array) if v <= max(K-s+tolerance, 0)]
            stop_line_V.append(stop[-1][1])
            stop_line_S.append(stop[-1][0])
    return stop_line_V, stop_line_S


def fill_bsm(x_aray, t_array, K, T, sigma, r, call=True):
    xx, tt = np.meshgrid(x_aray, t_array)
    dtt = T - tt
    d1 = (np.log(xx / K) + (r + sigma ** 2 / 2) * dtt) / (sigma * np.sqrt(dtt))
    d2 = d1 - sigma * np.sqrt(dtt)
    p = 1 if call else -1
    return p * xx * norm.cdf(p * d1) - p * K * np.exp(r * dtt) * norm.cdf(p * d2)
