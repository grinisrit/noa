from heston import MarketParameters, ModelParameters, fHes, JacHes, HesIntJac
import numpy as np
# from levenberg_marquardt import Levenberg_Marquardt
from typing import Tuple

karr = np.array(
    [
        0.9371,
        0.8603,
        0.8112,
        0.7760,
        0.7470,
        0.7216,
        0.6699,
        0.6137,
        0.9956,
        0.9868,
        0.9728,
        0.9588,
        0.9464,
        0.9358,
        0.9175,
        0.9025,
        1.0427,
        1.0463,
        1.0499,
        1.0530,
        1.0562,
        1.0593,
        1.0663,
        1.0766,
        1.2287,
        1.2399,
        1.2485,
        1.2659,
        1.2646,
        1.2715,
        1.2859,
        1.3046,
        1.3939,
        1.4102,
        1.4291,
        1.4456,
        1.4603,
        1.4736,
        1.5005,
        1.5328,
    ],
    dtype=np.float64,
)


tarr = np.array(
    [
        0.119047619047619,
        0.238095238095238,
        0.357142857142857,
        0.476190476190476,
        0.595238095238095,
        0.714285714285714,
        1.07142857142857,
        1.42857142857143,
        0.119047619047619,
        0.238095238095238,
        0.357142857142857,
        0.476190476190476,
        0.595238095238095,
        0.714285714285714,
        1.07142857142857,
        1.42857142857143,
        0.119047619047619,
        0.238095238095238,
        0.357142857142857,
        0.476190476190476,
        0.595238095238095,
        0.714285714285714,
        1.07142857142857,
        1.42857142857143,
        0.119047619047619,
        0.238095238095238,
        0.357142857142857,
        0.476190476190476,
        0.595238095238095,
        0.714285714285714,
        1.07142857142857,
        1.42857142857143,
        0.119047619047619,
        0.238095238095238,
        0.357142857142857,
        0.476190476190476,
        0.595238095238095,
        0.714285714285714,
        1.07142857142857,
        1.42857142857143,
    ],
    dtype=np.float64,
)
S_val = np.float64(1.0)
r_val = np.float64(0.02)

market = MarketParameters(K=karr, T=tarr, S=S_val, r=r_val)

a = np.float64(3.0)  # kappa                           |  mean reversion rate
b = np.float64(0.10)  # v_infinity                      |  long term variance
c = np.float64(0.25)  # sigma                           |  variance of volatility
rho = np.float64(
    -0.8
)  # rho                             |  correlation between spot and volatility
v0 = np.float64(0.08)

model = ModelParameters(a=a, b=b, c=c, rho=rho, v0=v0)

# fHes demo
x = fHes(
    model_parameters=model,
    market_parameters=market,
)
print(x)

# HesIntJac demo
res = HesIntJac(
    model_parameters=model, 
    market_parameters=market,
    market_pointer=np.int32(0))
print(res.pa1s)


# JacHes demo
hes = JacHes(
    model_parameters=model,
    market_parameters=market,
)
print(hes)





