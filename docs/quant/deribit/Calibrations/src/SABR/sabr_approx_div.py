import numba as nb
import numpy as np
import pandas as pd
from src.utils import get_tick
from typing import Final, Tuple
from src.levenberg_marquardt import LevenbergMarquardt
import math
from sklearn.linear_model import LinearRegression

_spec_market_params = [
    ("S", nb.float64),
    ("r", nb.float64),
    ("T", nb.float64),
    ("K", nb.float64[:]),
    ("iv", nb.float64[:]),
    ("types", nb.boolean[:]),
]

_spec_model_params = [
    ("alpha", nb.float64),
    ("v", nb.float64),
    ("beta", nb.float64),
    ("rho", nb.float64),
]


@nb.experimental.jitclass(_spec_market_params)
class MarketParameters(object):
    F: nb.float64
    r: nb.float64
    T: nb.float64
    K: nb.float64[:]
    iv: nb.float64[:]
    types: nb.boolean[:]

    def __init__(
        self,
        F: nb.float64,
        r: nb.float64,
        T: nb.float64,
        K: nb.float64[:],
        iv: nb.float64[:],
        types: nb.boolean[:],
    ):
        self.F = F
        self.r = r
        self.T = T
        self.K = K
        self.iv = iv
        self.types = types


@nb.experimental.jitclass(_spec_model_params)
class ModelParameters(object):
    alpha: nb.float64
    v: nb.float64
    beta: nb.float64
    rho: nb.float64

    def __init__(
        self, alpha: nb.float64, v: nb.float64, beta: nb.float64, rho: nb.float64
    ):
        self.alpha = alpha
        self.v = v
        self.beta = beta
        self.rho = rho


@nb.njit()
def get_vol(
    model: ModelParameters,
    market: MarketParameters,
) -> np.array:
    F, Ks, T = market.F, market.K, market.T
    alpha, beta, v, rho = model.alpha, model.beta, model.v, model.rho
    assert beta <= 1.0, "Beta should be from [0,1]"
    n = len(Ks)
    sigmas = np.zeros(n, dtype=np.float64)
    for index in range(n):
        K = Ks[index]
        x = np.log(F / K)
        I_H_1 = (
            alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 / 24
            + alpha * beta * rho * v * (K * F) ** (beta / 2 + -1 / 2) / 4
            + v**2 * (2 - 3 * rho**2) / 24
        )
        if x == 0.0:
            I_B_0 = K ** (beta - 1) * alpha
        elif v == 0.0:
            I_B_0 = alpha * (1 - beta) * x / (-(K ** (1 - beta)) + F ** (1 - beta))
        else:
            if beta == 1.0:
                z = v * x / alpha
            elif beta < 1.0:
                z = v * (F ** (1 - beta) - K ** (1 - beta)) / (alpha * (1 - beta))
            sqrt = np.sqrt(1 - 2 * rho * z + z**2)
            epsilon = np.log((-sqrt + rho - z) / (rho - 1))
            I_B_0 = v * x / epsilon

        sigma = I_B_0 * (1 + I_H_1 * T)
        sigmas[index] = sigma
    return sigmas


# ==================================================================
# =================== Black-Sholes greeks and tools ================
# ==================================================================
# @nb.njit()
# def cdf(x: float) -> float:
#     return (1.0 + math.erf(x / np.sqrt(2.0))) / 2.0


@nb.njit()
def cdf(x: float) -> float:
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
def pdf(x: float) -> float:
    probability = 1.0 / np.sqrt(2 * np.pi)
    probability *= np.exp(-0.5 * x**2)
    return probability


# @nb.njit()
def get_implied_volatility(
    option_type: bool,
    C: float,
    K: float,
    T: float,
    F: float,
    r: float = 0.0,
    error: float = 0.001,
) -> float:
    """
    Function to count implied volatility via given params of option, using Newton-Raphson method :

    Args:
        C (float): Option market price(USD).
        K (float): Strike(USD).
        T (float): Time to expiration in years.
        F (float): Underlying price.
        r (float): Risk-free rate.
        error (float): Given threshhold of error.

    Returns:
        float: Implied volatility in percent.
    """
    vol = 1.0
    dv = error + 1
    while abs(dv) > error:
        d1 = (np.log(F / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        D = np.exp(-r * T)
        if option_type == "call":
            price = F * cdf(d1) - K * cdf(d2) * D
        elif option_type == "put":
            price = -F * cdf(-d1) + K * cdf(-d2) * D
        else:
            raise ValueError(f"Value {option_type} is invalid")
        Vega = F * np.sqrt(T / np.pi / 2) * np.exp(-0.5 * d1**2)
        PriceError = price - C
        dv = PriceError / Vega
        vol = vol - dv
    return vol


@nb.njit()
def get_delta_bsm(
    option_type: str,
    sigma: float,
    K: float,
    T: float,
    F: float,
    r: float = 0.0,
) -> float:
    d1 = get_d1(sigma, K, T, F, r)
    return cdf(d1) if option_type else cdf(d1) - 1.0


@nb.njit()
def get_vega_bsm(
    sigma: float,
    K: float,
    T: float,
    F: float,
    r: float = 0.0,
) -> float:
    d1 = get_d1(sigma, K, T, F, r)
    return F * np.sqrt(T) * pdf(d1)


@nb.njit()
def get_gamma_bsm(
    sigma: float,
    K: float,
    T: float,
    F: float,
    r: float = 0.0,
) -> float:
    d1 = get_d1(sigma, K, T, F, r)
    return pdf(d1) / (F * sigma * np.sqrt(T))


# @nb.njit()
def get_price_bsm(
    model: ModelParameters,
    option_type: str,
    K: float,
    T: float,
    F: float,
    r: float = 0.0,
) -> float:
    market = MarketParameters(
        F=F,
        r=r,
        T=T,
        K=np.array([np.float64(K)]),
        # can lay zero, not needed to calculate volatility
        iv=np.array([np.float64(0.0)]),
        types=np.array([np.bool(option_type)]),
    )
    sigma = get_vol(model, market)[0]
    d1 = get_d1(sigma, K, T, F, r)
    d2 = d1 - sigma * np.sqrt(T)
    p = 1 if option_type else -1
    return p * F * cdf(p * d1) - p * K * np.exp(-r * T) * cdf(p * d2)


# @nb.njit()
def get_vanna_bsm():
    pass


@nb.njit()
def get_d1(
    sigma: float,
    K: float,
    T: float,
    F: float,
    r: float = 0.0,
):
    d1 = (np.log(F / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return d1


# ==================================================================
# ==================================================================
# ==================================================================


# ==================================================================
# ===================== SABR greeks ================================
# ==================================================================
# @nb.njit()
def get_delta(
    model: ModelParameters,
    option_type: bool,
    sigma: float,
    K: float,
    T: float,
    F: float,
    r: float = 0.0,
):
    alpha, beta, v, rho = model.alpha, model.beta, model.v, model.rho
    delta_bsm = get_delta_bsm(option_type, sigma, K, T, F, r)
    vega_bsm = get_vega_bsm(sigma, K, T, F, r)
    market = MarketParameters(
        F=np.float64(F),
        r=np.float64(r),
        T=np.float64(T),
        K=np.array([np.float64(K)]),
        iv=np.array([np.float64(sigma)]),
        types=np.array([np.bool(option_type)]),
    )
    dsigma_dalphas, _, _, _ = jacobian_sabr(model=model, market=market)

    dsigma_dalpha = dsigma_dalphas[0]
    dsigma_df = get_dsigma_df(model=model, K=K, T=T, F=F)
    return delta_bsm + vega_bsm * (dsigma_df + dsigma_dalpha * rho * v / F**beta)
    # no sticky delta variant, which passes finite diffs sanity check
    # return delta_bsm + vega_bsm * dsigma_df
    # BSM delta only
    # return delta_bsm


# @nb.njit()
def get_gamma(
    model: ModelParameters,
    option_type: bool,
    sigma: float,
    K: float,
    T: float,
    F: float,
    r: float = 0.0,
):
    alpha, beta, v, rho = model.alpha, model.beta, model.v, model.rho
    d1 = get_d1(sigma, K, T, F, r)
    dsigma_df = get_dsigma_df(model, K, T, F)
    gamma_bsm = get_gamma_bsm(sigma, K, T, F, r)
    market = MarketParameters(
        F=F,
        r=r,
        T=T,
        K=np.array([np.float64(K)]),
        iv=np.array([np.float64(sigma)]),
        types=np.array([np.bool(option_type)]),
    )
    dsigma_dalphas, _, _, _ = jacobian_sabr(model=model, market=market)
    dsigma_dalpha = dsigma_dalphas[0]
    d2_sigma_df2 = get_d2_sigma_df2(model, K, T, F)
    d2_sigma_dalpha_df = get_d2_sigma_dalpha_df(model, K, T, F)
    last_gamma_component_0 = (
        d2_sigma_df2
        + d2_sigma_dalpha_df * rho * v / F**beta
        - dsigma_dalpha * beta * rho * v / F ** (beta + 1)
    )
    last_gamma_component_0_not_sticky = d2_sigma_df2
    last_gamma_component = last_gamma_component_0 * F * pdf(d1) * np.sqrt(T)
    last_gamma_component_not_sticky = (
        last_gamma_component_0_not_sticky * F * pdf(d1) * np.sqrt(T)
    )
    # sticky
    return (
        gamma_bsm
        + (pdf(d1) - F * d1 * gamma_bsm)
        * np.sqrt(T)
        * (dsigma_df + dsigma_dalpha * rho * v / F**beta)
        + last_gamma_component
    )
    # not sticky
    # return (
    #     gamma_bsm
    #     + (pdf(d1) - F * d1 * gamma_bsm) * np.sqrt(T) * dsigma_df
    #     + last_gamma_component_not_sticky
    # )


# @nb.njit()
def get_vega(
    model: ModelParameters,
    option_type: bool,
    sigma: float,
    K: float,
    T: float,
    F: float,
    r: float = 0.0,
):
    alpha, beta, v, rho = model.alpha, model.beta, model.v, model.rho
    vega_bsm = get_vega_bsm(sigma, K, T, F, r)
    market = MarketParameters(
        F=F,
        r=r,
        T=T,
        K=np.array([np.float64(K)]),
        iv=np.array([np.float64(sigma)]),
        types=np.array([np.bool(option_type)]),
    )
    dsigma_dalphas, _, _, _ = jacobian_sabr(model=model, market=market)
    dsigma_dalpha = dsigma_dalphas[0]
    dsigma_df = get_dsigma_df(model, K, T, F)
    return vega_bsm * (dsigma_dalpha + dsigma_df * rho * F**beta / v)
    # no sticky vega variant
    # return vega_bsm * dsigma_dalpha


# @nb.njit()
def get_rega(
    model: ModelParameters,
    option_type: bool,
    sigma: float,
    K: float,
    T: float,
    F: float,
    r: float = 0.0,
):
    vega_bsm = get_vega_bsm(sigma, K, T, F, r)
    market = MarketParameters(
        F=F,
        r=r,
        T=T,
        K=np.array([np.float64(K)]),
        iv=np.array([np.float64(sigma)]),
        types=np.array([np.bool(option_type)]),
    )
    _, dsigma_dvs, _, _ = jacobian_sabr(model=model, market=market)
    dsigma_dv = dsigma_dvs[0]
    return vega_bsm * dsigma_dv


# @nb.njit()
def get_sega(
    model: ModelParameters,
    option_type: bool,
    sigma: float,
    K: float,
    T: float,
    F: float,
    r: float = 0.0,
):
    vega_bsm = get_vega_bsm(sigma, K, T, F, r)
    market = MarketParameters(
        F=F,
        r=r,
        T=T,
        K=np.array([np.float64(K)]),
        iv=np.array([np.float64(sigma)]),
        types=np.array([np.bool(option_type)]),
    )
    _, _, _, dsigma_drhos = jacobian_sabr(model=model, market=market)
    dsigma_drho = dsigma_drhos[0]
    return vega_bsm * dsigma_drho


@nb.njit()
def get_dsigma_dK(
    model: ModelParameters,
    K: float,
    T: float,
    F: float,
) -> float:
    x = np.log(F / K)
    alpha, beta, v, rho = model.alpha, model.beta, model.v, model.rho
    if x == 0.0:
        I_B_0 = K ** (beta - 1) * alpha
        dI_B_0_dK = K ** (beta - 2) * alpha * (beta - 1)
    elif v == 0.0:
        I_B_0 = alpha * (1 - beta) * x / (-(K ** (1 - beta)) + F ** (1 - beta))

        dI_B_0_dK = (
            alpha
            * (beta - 1)
            * (
                -K * (K ** (1 - beta) - F ** (1 - beta))
                + K ** (2 - beta) * (beta - 1) * np.log(F / K)
            )
            / (K**2 * (K ** (1 - beta) - F ** (1 - beta)) ** 2)
        )
    else:
        if beta == 1.0:
            z = v * x / alpha
            sqrt = np.sqrt(1 - 2 * rho * z + z**2)
            epsilon = np.log((-sqrt + rho - z) / (rho - 1))
            dI_B_0_dK = (
                v
                * (-sqrt * alpha * epsilon + v * x)
                / (K * sqrt * alpha * epsilon**2)
            )
        elif beta < 1.0:
            z = v * (F ** (1 - beta) - K ** (1 - beta)) / (alpha * (1 - beta))
            sqrt = np.sqrt(1 - 2 * rho * z + z**2)
            epsilon = np.log((-sqrt + rho - z) / (rho - 1))

            dI_B_0_dK = (
                v
                * (K * v * x - K**beta * sqrt * alpha * epsilon)
                / (K * K**beta * sqrt * alpha * epsilon**2)
            )
        I_B_0 = v * x / (epsilon)

    I_H_1 = (
        alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 / 24
        + alpha * beta * rho * v * (K * F) ** (beta / 2 + -1 / 2) / 4
        + v**2 * (2 - 3 * rho**2) / 24
    )
    dI_H_1_dK = alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 * (beta - 1) / (
        24 * K
    ) + alpha * beta * rho * v * (K * F) ** (beta / 2 - 1 / 2) * (beta / 2 + -1 / 2) / (
        4 * K
    )

    dsigma_dK = dI_B_0_dK * (1 + I_H_1 * T) + dI_H_1_dK * I_B_0 * T
    return dsigma_dK


# ==================================================================
# ==================================================================
# ==================================================================


# ==================================================================
# =================== Sigma derivative componenets =================
# ==================================================================


@nb.njit()
def get_d2_sigma_dalpha_df(
    model: ModelParameters,
    K: float,
    T: float,
    F: float,
) -> float:
    x = np.log(F / K)
    alpha, beta, v, rho = model.alpha, model.beta, model.v, model.rho
    if x == 0.0:
        d2I_B_0_dalpha_df = 0.0
        B_alpha = K ** (beta - 1)
        I_B_0 = K ** (beta - 1) * alpha
        dI_B_0_dF = 0.0
    elif v == 0.0:
        d2I_B_0_dalpha_df = -(F ** (1 - beta)) * x * (1 - beta) ** 2 / (
            F * (-(K ** (1 - beta)) + F ** (1 - beta)) ** 2
        ) + (1 - beta) / (F * (-(K ** (1 - beta)) + F ** (1 - beta)))
        B_alpha = (beta - 1) * x / (K ** (1 - beta) - F ** (1 - beta))
        I_B_0 = alpha * (1 - beta) * x / (-(K ** (1 - beta)) + F ** (1 - beta))
        dI_B_0_dF = (
            alpha
            * (beta - 1)
            * (
                F * (K ** (1 - beta) - F ** (1 - beta))
                - F ** (2 - beta) * (beta - 1) * x
            )
            / (F**2 * (K ** (1 - beta) - F ** (1 - beta)) ** 2)
        )
    else:
        if beta == 1.0:
            z = v * x / alpha
            sqrt = np.sqrt(1 - 2 * rho * z + z**2)
            epsilon = np.log((-sqrt + rho - z) / (rho - 1))
            d2I_B_0_dalpha_df = (
                v
                * (
                    sqrt**2 * alpha * z * epsilon
                    + sqrt**2 * v * x * epsilon
                    - 2 * sqrt * v * x * z
                    + rho * v * x * z * epsilon
                    - v * x * z**2 * epsilon
                )
                / (sqrt**3 * alpha**2 * F * epsilon**3)
            )
            B_alpha = v * x * z / (alpha * sqrt * epsilon**2)
            dI_B_0_dF = (
                v * (alpha * sqrt * epsilon - v * x) / (alpha * F * sqrt * epsilon**2)
            )
        elif beta < 1.0:
            z = v * (F ** (1 - beta) - K ** (1 - beta)) / (alpha * (1 - beta))
            sqrt = np.sqrt(1 - 2 * rho * z + z**2)
            epsilon = np.log((-sqrt + rho - z) / (rho - 1))
            d2I_B_0_dalpha_df = (
                v
                * (
                    sqrt**2 * alpha * F**beta * z * epsilon
                    + sqrt**2 * F * v * x * epsilon
                    - 2 * sqrt * F * v * x * z
                    + F * rho * v * x * z * epsilon
                    - F * v * x * z**2 * epsilon
                )
                / (sqrt**3 * alpha**2 * F * F**beta * epsilon**3)
            )
            B_alpha = v * x * z / (alpha * sqrt * epsilon**2)
            dI_B_0_dF = (
                v
                * (
                    alpha * F * (-rho + z + sqrt) * sqrt * epsilon
                    + F ** (2 - beta) * v * x * (rho - z - sqrt)
                )
                / (alpha * F**2 * (-rho + z + sqrt) * sqrt * epsilon**2)
            )
        I_B_0 = v * x / (epsilon)

    I_H_1 = (
        alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 / 24
        + alpha * beta * rho * v * (K * F) ** (beta / 2 + -1 / 2) / 4
        + v**2 * (2 - 3 * rho**2) / 24
    )
    dI_H_1_dF = alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 * (beta - 1) / (
        24 * F
    ) + alpha * beta * rho * v * (K * F) ** (beta / 2 - 1 / 2) * (beta / 2 - 1 / 2) / (
        4 * F
    )
    dI_H_1_dalpha = (
        alpha * (K * F) ** (beta - 1) * (1 - beta) ** 2 / 12
        + beta * rho * v * (K * F) ** (beta / 2 - 1 / 2) / 4
    )
    d2I_H_1_dalpha_df = alpha * (K * F) ** (beta - 1) * (1 - beta) ** 2 * (beta - 1) / (
        12 * F
    ) + beta * rho * v * (K * F) ** (beta / 2 - 1 / 2) * (beta / 2 - 1 / 2) / (4 * F)

    d2_sigma_dalpha_df2 = (
        d2I_B_0_dalpha_df * (1 + I_H_1 * T)
        + B_alpha * dI_H_1_dF * T
        + d2I_H_1_dalpha_df * I_B_0 * T
        + dI_H_1_dalpha * dI_B_0_dF * T
    )
    return d2_sigma_dalpha_df2


@nb.njit()
def get_d2_sigma_df2(
    model: ModelParameters,
    K: float,
    T: float,
    F: float,
):
    alpha, beta, v, rho = model.alpha, model.beta, model.v, model.rho
    x = np.log(F / K)
    if x == 0.0:
        I_B_0 = K ** (beta - 1) * alpha
        d2I_B_0_d2f = 0.0
        dI_B_0_dF = 0.0
    elif v == 0.0:
        I_B_0 = alpha * (1 - beta) * x / (-(K ** (1 - beta)) + F ** (1 - beta))
        d2I_B_0_d2f = (
            alpha
            * (beta - 1)
            * (
                -(F**4) * (K ** (1 - beta) - F ** (1 - beta)) ** 2
                + F ** (5 - beta)
                * (K ** (1 - beta) - F ** (1 - beta))
                * (beta - 1)
                * (x * (beta - 1) + x - 2)
                + 2 * F ** (6 - 2 * beta) * x * (beta - 1) ** 2
            )
            / (F**6 * (K ** (1 - beta) - F ** (1 - beta)) ** 3)
        )
        dI_B_0_dF = (
            alpha
            * (beta - 1)
            * (
                F * (K ** (1 - beta) - F ** (1 - beta))
                - F ** (2 - beta) * (beta - 1) * x
            )
            / (F**2 * (K ** (1 - beta) - F ** (1 - beta)) ** 2)
        )
    else:
        if beta == 1.0:
            z = v * x / alpha
            sqrt = np.sqrt(1 - 2 * rho * z + z**2)
            epsilon = np.log((-sqrt + rho - z) / (rho - 1))
            d2I_B_0_d2f = (
                v
                * (
                    -(sqrt**3) * alpha**2 * epsilon**2
                    + sqrt**2 * alpha * v * x * epsilon
                    - 2 * sqrt**2 * alpha * v * epsilon
                    + 2 * sqrt * v**2 * x
                    - rho * v**2 * x * epsilon
                    + v**2 * x * z * epsilon
                )
                / (sqrt**3 * alpha**2 * F**2 * epsilon**3)
            )
            dI_B_0_dF = (
                v * (alpha * sqrt * epsilon - v * x) / (alpha * F * sqrt * epsilon**2)
            )
        elif beta < 1.0:
            z = v * (F ** (1 - beta) - K ** (1 - beta)) / (alpha * (1 - beta))
            sqrt = np.sqrt(1 - 2 * rho * z + z**2)
            epsilon = np.log((-sqrt + rho - z) / (rho - 1))
            d2I_B_0_d2f = (
                v
                * (
                    -(sqrt**3) * alpha**2 * F * (sqrt - rho + z) * epsilon**2
                    - 2
                    * sqrt**2
                    * alpha
                    * F ** (2 - beta)
                    * v
                    * (sqrt - rho + z)
                    * epsilon
                    + sqrt
                    * F ** (3 - 2 * beta)
                    * v**2
                    * x
                    * (sqrt - rho + z)
                    * epsilon
                    + 2 * sqrt * F ** (3 - 2 * beta) * v**2 * x * (sqrt - rho + z)
                    + F
                    * v
                    * x
                    * (
                        sqrt**3 * alpha * beta * F ** (1 - beta)
                        + sqrt**2
                        * (
                            alpha * F ** (1 - beta) * (-rho * (beta - 1) - rho + z)
                            - v
                            * (
                                F ** (1 - beta) * (-(K ** (1 - beta)) + F ** (1 - beta))
                                + F ** (2 - 2 * beta)
                            )
                        )
                        + F ** (2 - 2 * beta) * v * (rho - z) ** 2
                    )
                    * epsilon
                )
                / (sqrt**3 * alpha**2 * F**3 * (sqrt - rho + z) * epsilon**3)
            )
            dI_B_0_dF = (
                v
                * (sqrt * alpha * F**beta * epsilon - F * v * x)
                / (sqrt * alpha * F * F**beta * epsilon**2)
            )
        I_B_0 = v * x / (epsilon)

    I_H_1 = (
        alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 / 24
        + alpha * beta * rho * v * (K * F) ** (beta / 2 + -1 / 2) / 4
        + v**2 * (2 - 3 * rho**2) / 24
    )

    dI_H_1_dF = alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 * (beta - 1) / (
        24 * F
    ) + alpha * beta * rho * v * (K * F) ** (beta / 2 - 1 / 2) * (beta / 2 - 1 / 2) / (
        4 * F
    )
    d2I_H_1_d2f = (
        alpha**2
        * (K * F) ** (beta - 1)
        * (1 - beta) ** 2
        * (beta - 1) ** 2
        / (24 * F**2)
        - alpha**2
        * (K * F) ** (beta - 1)
        * (1 - beta) ** 2
        * (beta - 1)
        / (24 * F**2)
        + alpha
        * beta
        * rho
        * v
        * (K * F) ** (beta / 2 + -1 / 2)
        * (beta / 2 - 1 / 2) ** 2
        / (4 * F**2)
        - alpha
        * beta
        * rho
        * v
        * (K * F) ** (beta / 2 - 1 / 2)
        * (beta / 2 - 1 / 2)
        / (4 * F**2)
    )
    d2_sigma_df2 = d2I_B_0_d2f + T * (
        d2I_B_0_d2f * I_H_1 + d2I_H_1_d2f * I_B_0 + 2 * dI_B_0_dF * dI_H_1_dF
    )
    return d2_sigma_df2


@nb.njit()
def get_dsigma_df(
    model: ModelParameters,
    K: float,
    T: float,
    F: float,
) -> float:
    alpha, beta, v, rho = model.alpha, model.beta, model.v, model.rho
    x = np.log(F / K)
    if x == 0.0:
        dI_B_0_dF = 0.0
        I_B_0 = K ** (beta - 1) * alpha
    elif v == 0.0:
        dI_B_0_dF = (
            alpha
            * (beta - 1)
            * (
                F * (K ** (1 - beta) - F ** (1 - beta))
                - F ** (2 - beta) * (beta - 1) * x
            )
            / (F**2 * (K ** (1 - beta) - F ** (1 - beta)) ** 2)
        )
        I_B_0 = alpha * (1 - beta) * x / (-(K ** (1 - beta)) + F ** (1 - beta))
    else:
        if beta == 1.0:
            z = v * x / alpha
            sqrt = np.sqrt(1 - 2 * rho * z + z**2)
            epsilon = np.log((-sqrt + rho - z) / (rho - 1))
            dI_B_0_dF = (
                v * (alpha * sqrt * epsilon - v * x) / (alpha * F * sqrt * epsilon**2)
            )
        elif beta < 1.0:
            z = v * (F ** (1 - beta) - K ** (1 - beta)) / (alpha * (1 - beta))
            sqrt = np.sqrt(1 - 2 * rho * z + z**2)
            epsilon = np.log((-sqrt + rho - z) / (rho - 1))
            dI_B_0_dF = (
                v
                * (
                    alpha * F * (-rho + z + sqrt) * sqrt * epsilon
                    + F ** (2 - beta) * v * x * (rho - z - sqrt)
                )
                / (alpha * F**2 * (-rho + z + sqrt) * sqrt * epsilon**2)
            )
        I_B_0 = v * x / (epsilon)

    I_H_1 = (
        alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 / 24
        + alpha * beta * rho * v * (K * F) ** (beta / 2 + -1 / 2) / 4
        + v**2 * (2 - 3 * rho**2) / 24
    )
    dI_H_1_dF = alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 * (beta - 1) / (
        24 * F
    ) + alpha * beta * rho * v * (K * F) ** (beta / 2 - 1 / 2) * (beta / 2 - 1 / 2) / (
        4 * F
    )

    dsigma_df = dI_B_0_dF * (1 + I_H_1 * T) + dI_H_1_dF * I_B_0 * T
    return dsigma_df


# ==================================================================
# ==================================================================
# ==================================================================


@nb.njit()
def jacobian_sabr(
    model: ModelParameters,
    market: MarketParameters,
):
    F, Ks, T = market.F, market.K, market.T
    n = len(Ks)
    alpha, beta, v, rho = model.alpha, model.beta, model.v, model.rho
    ddalpha = np.zeros(n, dtype=np.float64)
    ddbeta = np.zeros(n, dtype=np.float64)
    ddv = np.zeros(n, dtype=np.float64)
    ddrho = np.zeros(n, dtype=np.float64)
    # need a cycle cause may be different formula for different strikes
    for index in range(n):
        K = Ks[index]
        x = np.log(F / K)
        I_H = (
            alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 / 24
            + alpha * beta * rho * v * (K * F) ** (beta / 2 - 1 / 2) / 4
            + v**2 * (2 - 3 * rho**2) / 24
        )
        dI_H_1_dalpha = (
            alpha * (K * F) ** (beta - 1) * (1 - beta) ** 2 / 12
            + beta * rho * v * (K * F) ** (beta / 2 - 1 / 2) / 4
        )
        dI_H_beta = (
            alpha**2 * (K * F) ** (beta - 1) * (1 - beta) ** 2 * np.log(K * F) / 24
            + alpha**2 * (K * F) ** (beta - 1) * (2 * beta - 2) / 24
            + alpha
            * beta
            * rho
            * v
            * (K * F) ** (beta / 2 + -1 / 2)
            * np.log(K * F)
            / 8
            + alpha * rho * v * (K * F) ** (beta / 2 - 1 / 2) / 4
        )
        dI_h_v = (
            alpha * beta * rho * (K * F) ** (beta / 2 + -1 / 2) / 4
            + v * (2 - 3 * rho**2) / 12
        )
        dI_H_rho = (
            alpha * beta * v * (K * F) ** (beta / 2 + -1 / 2) / 4 - rho * v**2 / 4
        )

        if x == 0.0:
            I_B = alpha * K ** (beta - 1)
            B_alpha = K ** (beta - 1)
            B_beta = K ** (beta - 1) * alpha * np.log(K)
            B_v = 0.0
            B_rho = 0.0

        elif v == 0.0:
            I_B = alpha * (1 - beta) * x / (F ** (1 - beta) - (K ** (1 - beta)))
            B_alpha = (beta - 1) * x / (K ** (1 - beta) - F ** (1 - beta))
            B_beta = (
                alpha
                * (
                    K ** (1 - beta)
                    - F ** (1 - beta)
                    + (beta - 1)
                    * (K ** (1 - beta) * np.log(K) - F ** (1 - beta) * np.log(F))
                )
                * x
                / (K ** (1 - beta) - F ** (1 - beta)) ** 2
            )
            B_v = 0.0
            B_rho = 0.0

        elif beta == 1.0:
            z = v * x / alpha
            sqrt = np.sqrt(1 - 2 * rho * z + z**2)
            epsilon = np.log((-sqrt + rho - z) / (rho - 1))
            I_B = v * x / epsilon
            B_alpha = v * x * z / (alpha * sqrt * epsilon**2)
            B_beta = 0.0
            B_v = x * (alpha * sqrt * epsilon - v * x) / (alpha * sqrt * epsilon**2)
            B_rho = (
                v
                * x
                * ((rho - 1) * (z + sqrt) + (-rho + z + sqrt) * sqrt)
                / ((rho - 1) * (-rho + z + sqrt) * sqrt * epsilon**2)
            )

        elif beta < 1.0:
            z = v * (F ** (1 - beta) - K ** (1 - beta)) / (alpha * (1 - beta))
            sqrt = np.sqrt(1 - 2 * rho * z + z**2)
            epsilon = np.log((-sqrt + rho - z) / (rho - 1))
            # I_B = v * (-(K ** (1 - beta)) + f ** (1 - beta)) / (alpha * (1 - beta))
            I_B = v * x / epsilon
            B_alpha = v * x * z / (alpha * sqrt * epsilon**2)
            B_beta = (
                v
                * x
                * (
                    K * F**beta * v * np.log(K)
                    + K**beta * alpha * F**beta * z
                    - K**beta * F * v * np.log(F)
                )
                / (K**beta * sqrt * alpha * F**beta * (beta - 1) * epsilon**2)
            )
            B_v = (
                x
                * (
                    sqrt * alpha * (beta - 1) * epsilon
                    - v * (K ** (1 - beta) - F ** (1 - beta))
                )
                / (sqrt * alpha * (beta - 1) * epsilon**2)
            )

            B_rho = (
                v
                * x
                * (sqrt * (sqrt - rho + z) + (sqrt + z) * (rho - 1))
                / (sqrt * (rho - 1) * (sqrt - rho + z) * epsilon**2)
            )

        sig_alpha = B_alpha * (1 + I_H * T) + dI_H_1_dalpha * I_B * T
        sig_beta = B_beta * (1 + I_H * T) + dI_H_beta * I_B * T
        sig_v = B_v * (1 + I_H * T) + dI_h_v * I_B * T
        sig_rho = B_rho * (1 + I_H * T) + dI_H_rho * I_B * T

        ddalpha[index] = sig_alpha
        ddbeta[index] = sig_beta
        ddv[index] = sig_v
        ddrho[index] = sig_rho
    return ddalpha, ddv, ddbeta, ddrho


def get_beta(df: pd.DataFrame, timestamp: int = None):
    if timestamp:
        data = df[df["timestamp"] <= timestamp].copy()
    else:
        data = df.copy()

    def get_closest(given_strikes, underlying_price):
        """Finds between which values the given own lies"""
        # given_strikes sorted
        for index in range(0, len(given_strikes) - 1):
            if (
                given_strikes[index] <= underlying_price
                and given_strikes[index + 1] >= underlying_price
            ):
                return given_strikes[index], given_strikes[index + 1]

    def get_closest_strike(forward: float):
        closest_strikes = get_closest(all_covered_strikes, forward)
        return closest_strikes

    available_strikes = sorted(data.strike_price.unique())
    max_value_of_underlying = data.underlying_price.max()
    min_value_of_underlying = data.underlying_price.min()

    # find between which strikes do max and min forward values are
    right_border = get_closest(available_strikes, max_value_of_underlying)[1]
    left_border = get_closest(available_strikes, min_value_of_underlying)[0]
    all_covered_strikes = [
        strike
        for strike in available_strikes
        if strike >= left_border and strike <= right_border
    ]

    # find between whick strikes one current tick the forward is
    data["closest_strikes"] = data["underlying_price"].apply(get_closest_strike)
    # make borders as columns
    data["left_border"] = data["closest_strikes"].apply(lambda x: x[0])
    data["right_border"] = data["closest_strikes"].apply(lambda x: x[1])

    # ticks of only crossed strikes
    df_of_only_needed_strikes = data[data["strike_price"].isin(all_covered_strikes)]

    df_of_only_needed_strikes["tau"] = (
        (df_of_only_needed_strikes.expiration - df_of_only_needed_strikes.timestamp)
        / 1e6
        / 3600
        / 24
        / 365
    )
    df_of_only_needed_strikes["mark_price_usd"] = (
        df_of_only_needed_strikes["strike_price"]
        * df_of_only_needed_strikes["underlying_price"]
    )
    mark_ivs = []
    for index, row in df_of_only_needed_strikes.iterrows():
        iv = get_implied_volatility(
            option_type=row["type"],
            C=row["mark_price_usd"],
            K=row["strike_price"],
            T=row["tau"],
            F=row["underlying_price"],
            r=0.0,
            error=0.001,
        )
        mark_ivs.append(iv)

    df_of_only_needed_strikes["mark_iv"] = mark_ivs

    df_of_only_needed_strikes = (
        df_of_only_needed_strikes[["timestamp", "strike_price", "mark_iv"]]
        .drop_duplicates()
        .copy()
    )

    # get for all timstamps the values of mark_iv for needed strikes
    df_of_only_needed_strikes_left = df_of_only_needed_strikes.copy()
    df_of_only_needed_strikes_left = df_of_only_needed_strikes_left.rename(
        columns={"mark_iv": "mark_iv_left"}
    )
    data = data.merge(
        df_of_only_needed_strikes_left,
        how="left",
        left_on=["timestamp", "left_border"],
        right_on=["timestamp", "strike_price"],
    )
    # same for right ordered
    df_of_only_needed_strikes_right = df_of_only_needed_strikes.copy()
    df_of_only_needed_strikes_right = df_of_only_needed_strikes_right.rename(
        columns={"mark_iv": "mark_iv_right"}
    )
    data = data.merge(
        df_of_only_needed_strikes_right,
        how="left",
        left_on=["timestamp", "right_border"],
        right_on=["timestamp", "strike_price"],
    )
    # drop useless cols
    data = data.drop(columns=["strike_price_y", "strike_price"])
    data = data.rename(columns={"strike_price_x": "strike_price"})
    # if there are no info on mark_iv of given strike for this tick, fill with previous value
    data["mark_iv_left"] = data["mark_iv_left"].fillna(method="ffill")
    data["mark_iv_right"] = data["mark_iv_right"].fillna(method="ffill")

    # get no similar by this cols ticks
    only_needed = data[
        [
            "timestamp",
            "underlying_price",
            "right_border",
            "left_border",
            "mark_iv_left",
            "mark_iv_right",
            "strike_price",
        ]
    ].copy()
    only_needed = only_needed.dropna()
    #     only_needed = only_needed[only_needed["strike_price"].isin(all_covered_strikes)]

    # weight needed vols by distance between forward and closest strikes
    only_needed["ATM_iv"] = only_needed["mark_iv_left"] * (
        only_needed["right_border"] - only_needed["underlying_price"]
    ) / (only_needed["right_border"] - only_needed["left_border"]) + only_needed[
        "mark_iv_right"
    ] * (
        only_needed["underlying_price"] - only_needed["left_border"]
    ) / (
        only_needed["right_border"] - only_needed["left_border"]
    )

    # get logs for regression
    only_needed["ln_underlying_price"] = np.log(only_needed["underlying_price"])
    only_needed["ln_ATM_iv"] = np.log(only_needed["ATM_iv"])
    to_fit = only_needed[["ln_underlying_price", "ln_ATM_iv"]].drop_duplicates().copy()

    lr = LinearRegression()
    lr.fit(to_fit["ln_underlying_price"].values.reshape(-1, 1), to_fit["ln_ATM_iv"])
    k = lr.coef_[0]
    # from formula
    beta = k + 1
    beta = max(min(beta, 1.0), 0.0)
    return beta


def calibrate_sabr(
    df: pd.DataFrame,
    start_params: np.array,
    timestamp: int = None,
    calibration_type: str = "all",
    beta: float = None,
):
    """
    Function to calibrate SABR model.
    Attributes:
        @param df (pd.DataFrame): Dataframe with history
        [
            timestamp(ns),
            type(put or call),
            strike_price(usd),
            expiration(ns),
            mark_price(etc/btc),
            underlying_price(usd)
        ]
        @param start_params (np.array): Params to start calibration via LM from
        @param timestamp (int): On which timestamp to calibrate the model.
            Should be in range of df timestamps.
        @param calibration_type(str): Type of calibration. Should be one of: ["all", "beta"]
        @param beta(float): Fix it to needed value if you don't want to calibrate it

    Return:
        calibrated_params (np.array): Array of optimal params on timestamp tick.
        error (float): Value of error on calibration.
    """
    available_calibration_types = ["all", "beta"]
    if calibration_type not in available_calibration_types:
        raise ValueError(
            f"calibration_type should be from {available_calibration_types}"
        )
    timestamp = timestamp if timestamp else df.sample(1)["dt"].iloc[0]
    tick = df[df["dt"] == timestamp].copy()
    assert len(tick) != 0, f"No data on timestamp {timestamp}"
    iv = []
    # count ivs
    for index, t in tick.iterrows():
        market_iv = get_implied_volatility(
            option_type=t["type"],
            C=t["mark_price_usd"],
            K=t["strike_price"],
            T=t["tau"],
            F=t["underlying_price"],
            r=0.0,
            error=0.001,
        )
        iv.append(market_iv)

    # drop if iv is None
    tick["iv"] = iv
    tick = tick[~tick["iv"].isin([None, np.nan])]
    karr = tick.strike_price.to_numpy(dtype=np.float64)
    # carr = tick.mark_price_usd.to_numpy(dtype=np.float64)
    iv = tick.iv.to_numpy(dtype=np.float64)
    T = np.float64(tick.tau.mean())
    types = np.where(tick["type"] == "call", True, False)
    # take it zero as on deribit
    r_val = np.float64(0.0)
    S_val = np.float64(tick.underlying_price.iloc[0])
    market = MarketParameters(K=karr, T=T, F=S_val, r=r_val, types=types, iv=iv)

    def clip_params(sabr_params: np.ndarray) -> np.ndarray:
        """
        This funciton project sabr parameters into valid range
        Attributes:
            sabr_params(np.ndarray): model parameters
        Returns:
            sabr_params(np.ndarray): clipped parameters
        """
        eps = 1e-4

        def clip_all(params):
            alpha, v, beta, rho = params
            # alpha = np.clip(alpha, eps, 50.0)
            # alpha = np.clip(alpha, eps, 0.8)
            alpha = np.clip(alpha, eps, 50.0)
            v = np.clip(v, eps, 50.0)
            # no need there cause by approx formula it can be 1.0
            beta = np.clip(beta, eps, 1.0)
            rho = np.clip(rho, -1.0 + eps, 1.0 - eps)
            sabr_params = np.array([alpha, v, beta, rho])
            return sabr_params

        if calibration_type == "all":
            sabr_params = clip_all(sabr_params)

        elif calibration_type == "beta":
            params = np.concatenate(
                [
                    sabr_params[0:2],
                    np.array([beta]),
                    sabr_params[2:],
                ]
            )
            sabr_params = clip_all(params)
            sabr_params = np.concatenate([sabr_params[0:2], sabr_params[3:]])

        return sabr_params

    def get_residuals(sabr_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function calculates residuals of market ivd and calibrated ones
        and Jacobian matrix
        Args:
            sabr_params(np.ndarray): model params
        Returns:
            res(np.ndarray) : vector or residuals
            J(np.ndarray)   : Jacobian
        """
        # function supports several calibration types
        if calibration_type == "all":
            model_parameters = ModelParameters(
                sabr_params[0],
                sabr_params[1],
                sabr_params[2],
                sabr_params[3],
            )
            J = jacobian_sabr(model=model_parameters, market=market)

        elif calibration_type == "beta":
            model_parameters = ModelParameters(
                sabr_params[0], sabr_params[1], beta, sabr_params[2]
            )
            J_tmp = jacobian_sabr(model=model_parameters, market=market)
            J = np.concatenate([J_tmp[0:2], J_tmp[3:]])

        iv = get_vol(model=model_parameters, market=market)
        weights = np.ones_like(market.K)
        weights = weights / np.sum(weights)
        res = iv - market.iv
        return res * weights, J @ np.diag(weights)

    if calibration_type == "all":
        res = LevenbergMarquardt(500, get_residuals, clip_params, start_params)
        calibrated_params = np.array(res["x"], dtype=np.float64)

    elif calibration_type == "beta":
        beta = beta if beta else get_beta(df, timestamp)
        res = LevenbergMarquardt(
            500,
            get_residuals,
            clip_params,
            np.concatenate([start_params[0:2], start_params[3:]]),
        )
        calibrated_params = np.array(res["x"], dtype=np.float64)
        calibrated_params = np.concatenate(
            [calibrated_params[0:2], np.array([beta]), calibrated_params[2:]]
        )
    error = res["objective"][-1]
    final_params = ModelParameters(
        calibrated_params[0],
        calibrated_params[1],
        calibrated_params[2],
        calibrated_params[3],
    )
    final_vols = get_vol(model=final_params, market=market)
    tick["calibrated_iv"] = final_vols
    deltas, vegas, gammas, segas, regas, prices = [], [], [], [], [], []
    for index, row in tick.iterrows():
        K = row["strike_price"]
        op_type = True if row["type"] == "call" else False
        vol = row["calibrated_iv"]
        deltas.append(get_delta(final_params, op_type, vol, K, T, S_val))
        vegas.append(get_vega(final_params, op_type, vol, K, T, S_val))
        regas.append(get_rega(final_params, op_type, vol, K, T, S_val))
        segas.append(get_sega(final_params, op_type, vol, K, T, S_val))
        gammas.append(get_gamma(final_params, op_type, vol, K, T, S_val))
        prices.append(get_price_bsm(final_params, op_type, vol, K, T, S_val))

    tick["delta"] = deltas
    tick["vega"] = vegas
    tick["gamma"] = gammas
    tick["sega"] = segas
    tick["rega"] = regas
    tick["rho"] = final_params.rho
    tick["volvol"] = final_params.v
    tick["alpha"] = final_params.alpha
    tick["beta"] = final_params.beta
    tick["calibrated_mark_price"] = prices

    result = tick[
        [
            "type",
            "strike_price",
            "expiration",
            "underlying_price",
            "iv",
            "calibrated_iv",
            "calibrated_mark_price",
            "delta",
            "vega",
            "gamma",
            "sega",
            "rega",
            "mark_price_usd",
            "rho",
            "volvol",
            "alpha",
            "beta",
            "tau"
        ]
    ]
    result["iv"] = 100 * result["iv"]
    result["calibrated_iv"] = 100 * result["calibrated_iv"]
    return calibrated_params, error, result
