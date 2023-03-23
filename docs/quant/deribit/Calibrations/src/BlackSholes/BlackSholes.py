import numpy as np
from scipy import stats as sps

class BlackSholes:
    def __init__(self, r: float, K: float, F: float, T: float, vol: float):
        """Returns the price of option with given params
        Args:
            K(float): strike,
            F(float): underlying price,
            T(float): Time to expiration in years,
            r(float): risk-free rate,
            vol(float): volatility
        Returns:
            C/P: option price 
        """
        self.r = r
        self.K = K
        self.F = F
        self.T = T
        self.vol = vol
        d1 = (np.log(F / K) + 0.5 * vol ** 2 * T) \
                / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        D = np.exp(-r * T)
        self.call_price = F * sps.norm.cdf(d1) - K * sps.norm.cdf(d2) * D
        self.put_price = K * sps.norm.cdf(- d2) * D - F * sps.norm.cdf(- d1)