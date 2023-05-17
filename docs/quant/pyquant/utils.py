import numpy as np
import numba as nb


@nb.njit()
def normal_cdf(x: nb.float64) -> nb.float64:
    t = 1 / (1 + 0.2316419 * abs(x))
    summ = (
        0.319381530 * t
        - 0.356563782 * t**2
        + 1.781477937 * t**3
        - 1.821255978 * t**4
        + 1.330274429 * t**5
    )
    if x >= 0:
        return 1 - summ * np.exp(-np.absolute(x) ** 2 / 2) / np.sqrt(2 * np.pi)
    else:
        return summ * np.exp(-np.absolute(x) ** 2 / 2) / np.sqrt(2 * np.pi)


@nb.njit()
def normal_pdf(x: nb.float64) -> nb.float64:
    probability = 1.0 / np.sqrt(2 * np.pi)
    probability *= np.exp(-0.5 * x**2)
    return probability


@nb.njit()
def np_clip(a: nb.float64, a_min: nb.float64, a_max: nb.float64) -> nb.float64:
    if a < a_min:
        out = a_min
    elif a > a_max:
        out = a_max
    else:
        out = a
    return out 

@nb.njit
def is_sorted(a: nb.float64[:]) -> nb.boolean:
    for i in range(a.size-1):
         if a[i+1] < a[i] :
               return False
    return True
