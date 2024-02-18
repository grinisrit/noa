import numpy as np
import numba as nb


# One year in nanoseconds (365 * 24 * 3600 * 1_000_000_000)
YEAR_NANOS: int = 31536000000000000


@nb.njit(nb.float64(nb.float64))
def normal_cdf(x: nb.float64) -> nb.float64:
    t = 1 / (1 + 0.2316419 * np.absolute(x))
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

@nb.vectorize([nb.float64(nb.float64)], nopython=True)
def normal_cdf_vec(x: nb.float64[:]) -> nb.float64[:]:
    return normal_cdf(x)

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

@nb.njit
def mass_weights(t: nb.float64, Ts: nb.float64[:], tol: nb.float64 = 1e-6) -> nb.float64[:]:
    n = len(Ts)
    w = np.zeros_like(Ts)
    flag = False
    for i in range(n):
        wi = t - Ts[i]
        flag = t - Ts[i] <= 0.
        if flag:
            if abs(wi) <= tol:
                w[i] = 1.
                break
            w[i] = 1/(abs(wi) + 1e-12)
            if i-1 >= 0:
                w[i-1] = 1/(abs(t - Ts[i-1]) + 1e-12)
            break
    if np.all(w<=0):
        w[-1] = 1.
    return w / w.sum()


@nb.njit(cache = True, fastmath = True)
def searchsorted(a: nb.float64[:], b: nb.float64) -> nb.int64:
    idx = 0
    pa, pb = 0, 0
    while pb < 1:
        if pa < len(a) and a[pa] < b:
            pa += 1
        else:
            idx = pa
            pb += 1
    return idx


@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class XAxis:
    def __init__(self, x: nb.float64[:]):
        self.data = x


@nb.experimental.jitclass([
    ("data", nb.float64[:])
])
class YAxis:
    def __init__(self, y: nb.float64[:]):
        self.data = y


@nb.experimental.jitclass([
    ("_x0", nb.float64[:]),
    ("_a", nb.float64[:]),
    ("_b", nb.float64[:]),
    ("_c", nb.float64[:]),
    ("_d", nb.float64[:]),
])
class CubicSpline1D:
    def __init__(self, x: XAxis, y: YAxis):
        self._x0 = x.data
        self._calc_spline_params(x.data, y.data)

    def _calc_spline_params(self, x: nb.float64[:], y: nb.float64[:]):
        n = x.size - 1
        a = y.copy()
        h = x[1:] - x[:-1]
        alpha = 3 * ((a[2:] - a[1:-1]) / h[1:] - (a[1:-1] - a[:-2]) / h[:-1])
        c = np.zeros(n+1)
        ell, mu, z = np.ones(n+1), np.zeros(n), np.zeros(n+1)
        for i in range(1, n):
            ell[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
            mu[i] = h[i] / ell[i]
            z[i] = (alpha[i-1] - h[i-1] * z[i-1]) / ell[i]
        for i in range(n-1, -1, -1):
            c[i] = z[i] - mu[i] * c[i+1]
        b = (a[1:] - a[:-1]) / h + (c[:-1] + 2 * c[1:]) * h / 3
        d = np.diff(c) / (3 * h)

        self._a = a[1:]
        self._b = b
        self._c = c[1:]
        self._d = d

    def _func_spline(self, x: nb.float64, ix: nb.int64) -> nb.float64:
        dx = x - self._x0[1:][ix]
        return self._a[ix] + (self._b[ix] + (self._c[ix] + self._d[ix] * dx) * dx) * dx
    
    def apply(self, x: nb.float64) -> nb.float64:
        ix = searchsorted(self._x0[1 : -1], x)
        return self._func_spline(x, ix)