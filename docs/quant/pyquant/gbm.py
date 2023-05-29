import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def generate_gbm(
        n_paths: int,
        n_steps: int,
        dt: float,
        S0: torch.Tensor,
        sigma: torch.Tensor,
        drift: torch.Tensor
) -> torch.Tensor:
    """Generates one or multiple geometric Brownian motion processes.

    Args:
        n_paths: Number of paths to simulate for each process.
        n_steps: Number of time steps for each process.
        dt: Time step for each process.
        S0: Initial value for each process.
        sigma: Positive-definite covariance matrix for the processes
            (or the value of variance in case of a single process).
        drift: Drift for each process.

    Shape:
        S0: (N, ) where N is the number of processes to generate.
        sigma: (1, ) if N == 1, otherwise (N, N).
        drift: (N, )

    Returns:
        Tensor with generated paths of N processes.
        Shape: (n_paths, n_steps + 1) if N == 1, otherwise (N, n_paths, n_steps + 1).
    """
    if len(S0.shape) != 1:
        raise ValueError('`S0` must be 1D tensor')
    if len(drift.shape) != 1:
        raise ValueError('`drift` must be 1D tensor')
    if len(S0.shape) != len(drift.shape):
        raise ValueError('`S0` and `drift` must have the same length')
    N_VARS = len(S0)
    if N_VARS == 1 and sigma.squeeze().shape != torch.Size([]):
        raise ValueError('For single process, `sigma` must contain a single number')
    if N_VARS > 1 and sigma.shape != torch.Size([N_VARS, N_VARS]):
        raise ValueError('For multiple processes, the shape of `sigma` must be '
                         '(N, N) where N is the length of `S0`')

    distr = MultivariateNormal(loc=torch.zeros(N_VARS), covariance_matrix=sigma)
    sample = distr.sample(torch.Size((n_paths, n_steps)))
    volats = torch.diag(sigma) ** 0.5
    paths = S0 * torch.cumprod(1 + drift * dt + volats * (dt ** 0.5) * sample, dim=1)
    paths = torch.cat((S0.expand(n_paths, 1, N_VARS), paths), dim=1)
    paths = torch.movedim(paths, 2, 0)
    return paths.squeeze(0)
