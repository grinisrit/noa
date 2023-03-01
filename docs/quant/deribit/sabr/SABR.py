import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
from loguru import logger


class SABR:
    def __init__(self, data: pd.DataFrame, beta: float = 0.5) -> None:
        """Class to model the volatility smile"""
        self.data = data
        # let it be fixed as in the artice
        self.beta = beta
        self.T = data.iloc[0].tau
        # Change by milliseconds, but we need similar
        self.underlying_price = data.iloc[0].underlying_price
        # start params for optimization
        self.x0 = np.array([0.99, 0.00, 0.99])
        self.bounds = [(0.0001, 1000.0), (-0.9999, 0.9999), (0.0001, 1000.0)]

    def _sigmaB(
        self, f: float, K: float, T: float, alpha: float, rho: float, v: float
    ) -> float:
        """Function to count modeled volatility"""
        first_part_of_numerator = (
            (1 - self.beta) ** 2 / 24 * alpha**2 / (f * K) ** (1 - self.beta)
        )
        second_part_of_numerator = (
            (rho * self.beta * v * alpha) / 4 * (f * K) ** ((1 - self.beta) / 2)
        )
        third_part_of_numerator = (2 - 3 * rho**2) * v**2 / 24
        numerator = alpha * (
            1
            + self.T
            * (
                first_part_of_numerator
                + second_part_of_numerator
                + third_part_of_numerator
            )
        )

        first_part_of_denominator = (1 - self.beta) ** 2 / 24 * (np.log(f / K)) ** 2
        second_part_of_denominator = (1 - self.beta) ** 4 / 1920 * (np.log(f / K)) ** 4
        denominator = (f * K) ** ((1 - self.beta) / 2) * (
            1 + first_part_of_denominator + second_part_of_denominator
        )

        z = v / alpha * (f * K) ** ((1 - self.beta) / 2) * np.log(f / K)

        xi = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

        return numerator / denominator * z / xi

    def _plot_results(self) -> None:
        """Function to plot results"""

        def get_sigmas_for_set_of_params(alpha: float, rho: float, v: float) -> None:
            """Inner function to count sigmas for current optimization method"""
            self.volatilities = []
            self.test_strikes = np.linspace(500, 5000, 100)
            for strike in self.test_strikes:
                sigma_modeled = self._sigmaB(
                    self.underlying_price, strike, self.T, alpha, rho, v
                )
                self.volatilities.append(sigma_modeled)

        get_sigmas_for_set_of_params(self.alpha_scipy, self.rho_scipy, self.v_scipy)

        fig, ax = plt.subplots(figsize=(20, 7))
        ax = sns.scatterplot(
            x="strike_price",
            y="mark_iv",
            data=self.data,
            color="black",
            label="scipy optimizer",
        )

        ax1 = sns.lineplot(
            x=self.test_strikes,
            y=self.volatilities,
            label="market volatilities",
            color="blue",
        ).set_title(
            f"""
        T = {int(self.T * 365)} days
        alpha: {round(self.alpha_scipy, 2)}
        beta: {round(self.beta, 2)}
        rho: {round(self.rho_scipy, 2)}
        volvol: {round(self.v_scipy, 2)}
        """
        )

    def _vol_square_error(self, x: np.ndarray) -> np.float64:
        """Function to get the argmin function we want to optimize"""
        # init market volatiliteis
        smile = self.data.mark_iv.to_numpy()
        vols = []
        for index, row in self.data.iterrows():
            vols.append(
                self._sigmaB(
                    self.underlying_price, row["strike_price"], self.T, x[0], x[1], x[2]
                )
            )
        return sum((vols - smile) ** 2)

    def _minimize_scipy(self) -> np.float64:
        """Optimization with scipy optimizer"""
        return minimize(self._vol_square_error, x0=self.x0, bounds=self.bounds)

    def run(self) -> None:
        """Run optimization and plot results"""
        # optimization via scipy
        optimum_scipy = self._minimize_scipy()
        self.alpha_scipy, self.rho_scipy, self.v_scipy = optimum_scipy.x
        logger.info(
            f"""Optimal params for T = {int(self.T*365)} days: 
            alpha = {round(self.alpha_scipy, 2)}, 
            rho = {round(self.rho_scipy, 2)}, 
            volvol = {round(self.v_scipy, 2)}
            beta = {round(self.beta, 2)}"""
        )
        # plot market values and modeled function
        self._plot_results()
        return (
            self.volatilities,
            self.alpha_scipy,
            self.beta,
            self.rho_scipy,
            self.v_scipy,
            self.T,
        )
