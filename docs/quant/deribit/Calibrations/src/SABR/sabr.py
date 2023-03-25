import numba as nb
import numpy as np
import pandas as pd
from src.utils import get_tick, get_implied_volatility
from typing import Final, Tuple
from src.levenberg_marquardt import LevenbergMarquardt
from scipy import stats as sps


_spec_market_params = [
    ("S", nb.float64),
    ("r", nb.float64),
    ("T", nb.float64),
    ("K", nb.float64[:]),
    ("C", nb.float64[:]),
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
    S: nb.float64
    r: nb.float64
    T: nb.float64
    K: nb.float64[:]
    C: nb.float64[:]
    iv: nb.float64[:]
    types: nb.boolean[:]

    def __init__(
        self,
        S: nb.float64,
        r: nb.float64,
        T: nb.float64,
        K: nb.float64[:],
        C: nb.float64[:],
        iv: nb.float64[:],
        types: nb.boolean[:],
    ):
        self.S = S
        self.r = r
        self.T = T
        self.K = K
        self.C = C
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


_tmp_values_vol_sabr = {
    "Fms": nb.types.Array(nb.float64, 1, "A"),
    "Fm": nb.types.Array(nb.float64, 1, "A"),
    "q1": nb.types.Array(nb.float64, 1, "A"),
    "q2": nb.types.Array(nb.float64, 1, "A"),
    "q3": nb.float64,
    "S": nb.types.Array(nb.float64, 1, "A"),
    "zeta": nb.types.Array(nb.float64, 1, "A"),
    "sqrt": nb.types.Array(nb.float64, 1, "A"),
    "X": nb.types.Array(nb.float64, 1, "A"),
    "D": nb.types.Array(nb.float64, 1, "A"),
    "sig": nb.types.Array(nb.float64, 1, "A"),
}

# @nb.njit(locals=_tmp_values_vol_sabr)
def vol_sabr(
    model: ModelParameters,
    market: MarketParameters,
) -> np.array:

    Fm = np.multiply(market.S, market.K)
    # Fm = market.K
    # Fm = np.sqrt(Fms)

    q1 = (model.beta - 1) ** 2 * model.alpha**2 * Fm ** (2 * model.beta - 2) / 24
    q2 = model.rho * model.beta * model.alpha * model.v * Fm ** (model.beta - 1) / 4
    q3 = (2 - 3 * model.rho**2) / 24 * model.v**2

    S = 1 + market.T * (q1 + q2 + q3)

    zeta = model.v / model.alpha * Fm ** (1 - model.beta) * np.log(market.S / market.K)
    sqrt = np.sqrt(1 - 2 * model.rho * zeta + zeta**2)
    X = np.log((sqrt + zeta - model.rho) / (1 - model.rho))

    D = Fm ** (1 - model.beta) * (
        1
        + (model.beta - 1) ** 2 / 24 * (np.log(market.S / market.K)) ** 2
        + (model.beta - 1) ** 4 / 1920 * (np.log(market.S / market.K)) ** 4
    )

    sig = model.alpha * S * zeta / D / X

    # C = black_scholes(K, F, T, r, sig)
    # return C, sig
    return sig


_tmp_values_jacobian_sabr = {}

# @nb.njit(_signature_jacobian_sabr, locals=_tmp_values_jacobian_sabr)
def jacobian_sabr(
    model: ModelParameters,
    market: MarketParameters,
):
    Fm = np.sqrt(market.S * market.K)

    q1 = (model.beta - 1) ** 2 * model.alpha**2 * Fm ** (2 * model.beta - 2) / 24
    q2 = model.rho * model.beta * model.alpha * model.v * Fm ** (model.beta - 1) / 4
    q3 = (2 - 3 * model.rho**2) / 24 * model.v**2

    S = 1 + market.T * (q1 + q2 + q3)

    zeta = model.v / model.alpha * Fm ** (1 - model.beta) * np.log(market.S / market.K)
    sqrt = np.sqrt(1 - 2 * model.rho * zeta + zeta**2)
    X = np.log((sqrt + zeta - model.rho) / (1 - model.rho))

    D = Fm ** (1 - model.beta) * (
        1
        + (model.beta - 1) ** 2 / 24 * (np.log(market.S / market.K)) ** 2
        + (model.beta - 1) ** 4 / 1920 * (np.log(market.S / market.K)) ** 4
    )

    sig = model.alpha * S * zeta / D / X

    X_zeta = 1 / sqrt

    S_alpha = market.T * (2 * q1 + q2) / model.alpha
    zeta_alpha = -zeta / model.alpha
    X_alpha = X_zeta * zeta_alpha

    S_rho = (
        market.T
        * model.v
        * (model.beta * model.alpha * Fm ** (model.beta - 1) - model.rho * model.v)
        / 4
    )
    X_rho = 1 / (1 - model.rho) - 1 / sqrt * (sqrt + zeta) / (sqrt + zeta - model.rho)

    S_v = market.T / model.v * (q2 + 2 * q3)
    zeta_v = zeta / model.v
    X_v = X_zeta * zeta_v

    zeta_beta = -np.log(Fm) * zeta
    X_beta = X_zeta * zeta_beta
    S_beta = market.T * (
        2 * q1 * (1 / (model.beta - 1) + np.log(Fm))
        + q2 * (1 / model.beta + np.log(Fm))
    )
    D_beta = -np.log(Fm) * D + Fm ** (1 - model.beta) * (
        (model.beta - 1) / 12 * (np.log(market.S / market.K)) ** 2
        + (model.beta - 1) ** 3 / 480 * (np.log(market.S / market.K)) ** 4
    )

    logs_alpha = 1 / model.alpha + S_alpha / S + zeta_alpha / zeta - X_alpha / X
    logs_v = S_v / S + zeta_v / zeta - X_v / X
    logs_beta = S_beta / S - D_beta / D + zeta_beta / zeta - X_beta / X
    logs_rho = S_rho / S - X_rho / X

    sig_alpha = sig * logs_alpha
    sig_v = sig * logs_v
    sig_beta = sig * logs_beta
    sig_rho = sig * logs_rho

    # C, vega = black_scholes_vega(K, F, T, r, sig)
    # return C, vega, sig, sig_alpha, sig_v, sig_beta, sig_rho
    return sig_alpha, sig_v, sig_beta, sig_rho


def calibrate_sabr(
    df: pd.DataFrame,
    start_params: np.array,
    timestamp: int = None,
    calibration_type: str = "all",
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

    Return:
        calibrated_params (np.array): Array of optimal params on timestamp tick.
        error (float): Value of error on calibration.
    """
    available_calibration_types = ["all", "beta"]
    if calibration_type not in available_calibration_types:
        raise ValueError(
            f"calibration_type should be from {available_calibration_types}"
        )
    tick = get_tick(df=df, timestamp=timestamp)
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
    carr = tick.mark_price_usd.to_numpy(dtype=np.float64)
    iv = tick.iv.to_numpy(dtype=np.float64)
    T = np.float64(tick.tau.mean())
    types = np.where(tick["type"] == "call", True, False)
    # take it zero as on deribit
    r_val = np.float64(0.0)
    # tick dataframes may have not similar timestamps -->
    # not equal value if underlying --> take mean
    S_val = np.float64(tick.underlying_price.mean())
    market = MarketParameters(K=karr, T=T, S=S_val, r=r_val, C=carr, types=types, iv = iv)

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
            alpha = np.clip(alpha, eps, 50.0)
            v = np.clip(v, eps, 100.0)
            beta = np.clip(beta, eps, 1.0 - eps)
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

        iv = vol_sabr(model=model_parameters, market=market)
        weights = np.ones_like(market.K)
        weights = weights / np.sum(weights)
        res = iv - market.iv
        return res * weights, J @ np.diag(weights)

    if calibration_type == "all":
        res = LevenbergMarquardt(500, get_residuals, clip_params, start_params)
        calibrated_params = np.array(res["x"], dtype=np.float64)

    elif calibration_type == "beta":
        beta = 0.9999
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
    final_vols = vol_sabr(model=final_params, market=market)
    tick["calibrated_iv"] = final_vols
    result = tick[
        [
            "type",
            "strike_price",
            "expiration",
            "underlying_price",
            "iv",
            "calibrated_iv",
        ]
    ]
    result["iv"] = 100*result["iv"]
    result["calibrated_iv"] = 100*result["calibrated_iv"]
    return calibrated_params, error, result
