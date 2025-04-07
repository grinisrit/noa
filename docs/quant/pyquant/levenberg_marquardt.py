import numba as nb
import numpy as np


class LevenbergMarquardtOptimizer:
    def __init__(self, num_iter=100, tol=1e-6, min_mu=1e-8, max_mu=1e8):
        self.num_iter = num_iter
        self.tol = tol
        self.min_mu = min_mu
        self.max_mu = max_mu

    def optimize(self, f: callable, proj: callable, x0: np.array, n_points: int):
        """
        Parameters:
        -----------
        f: callable
            Returns (residuals, jacobian) of the function we calibrate
        proj: callable
            Clipping params to needed range
        x0: np.ndarray
            Zero dots
        n_points: int
            On


        Returns:
        --------
            tuple: (optimal params, final error value)
        """
        x = x0.copy()

        mu = 1e-2
        nu1 = 2.0
        nu2 = 2.0

        res, J = f(x)
        F = res.T @ res

        result_x = x
        result_error = F / n_points

        for i in range(self.num_iter):
            if result_error < self.tol:
                break
            multipl = J @ J.T
            I = np.diag(np.diag(multipl)) + 1e-5 * np.eye(len(x))
            dx = np.linalg.solve(mu * I + multipl, J @ res)
            x_ = proj(x - dx)
            res_, J_ = f(x_)
            F_ = res_.T @ res_
            if F_ < F:
                x, F, res, J = x_, F_, res_, J_
                mu = max(self.min_mu, mu / nu1)
                result_error = F / n_points
            else:
                i -= 1
                mu = min(self.max_mu, mu * nu2)
                continue
            result_x = x
        return result_x, result_error
