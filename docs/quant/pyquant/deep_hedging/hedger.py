from typing import Optional, Tuple, List
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
import torch

from derivatives import Portfolio


class BaseHedger(ABC):
    """Base class for hedging model.
    Attributes:
        portfolio:
            Portfolio to hedge.
        under_indexes:
            A list of indexes of underlyings to use for hedging. Indexes must
            be unique integers in range [0, self.portfolio.n_unders-1].
            If `None`, all underlyings are used.
        device:
            Torch device to work on.
    """

    def __init__(
            self,
            portfolio: Portfolio,
            under_indexes: Optional[List] = None,
            device='cpu'
    ):
        self.portfolio = portfolio
        self.device = device

        if under_indexes is not None:
            if len(set(under_indexes)) != len(under_indexes):
                raise ValueError('`under_indexes` must contain unique integers')
            if min(under_indexes) < 0 or max(under_indexes) > self.portfolio.n_unders - 1:
                raise ValueError('`under_indexes` must contain integers in range [1, self.portfolio.n_unders]')
            self.under_indexes = under_indexes
        else:
            self.under_indexes = list(range(0, self.portfolio.n_unders))

    @abstractmethod
    def get_hedge(
            self,
            time_idx: int,
            prev_hedge: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Calculates the hedge amount for the portfolio at given time.
        Args:
            time_idx:
                An index of time point. Must be less than `self.portfolio.n_steps + 1`.
            prev_hedge:
                Hedge amount at previous moment of time.
        Returns:
            A tensor with hedge amount in all or some of the underlyings
            of the portfolio for each path.
            Shape: (N, len(under_indexes)), where N is the number of paths.
        """
        pass

    def compute_pnl(self, cost: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes PnL and costs distribution after hedging a portfolio.

        Args:
            cost: Transaction cost.
        Returns:
            1) PnL. Shape: (self.portfolio.n_paths, ).
            2) Costs. Shape: (self.portfolio.n_paths, ).
        """
        pnl = torch.zeros(self.portfolio.n_paths).to(self.device)
        costs = torch.zeros(self.portfolio.n_paths).to(self.device)
        prev_hedge = torch.zeros(len(self.under_indexes), self.portfolio.n_paths).to(self.device)

        for j in range(0, self.portfolio.n_steps + 1):
            if j != 0:
                pnl += torch.sum(
                    prev_hedge * (self.portfolio.paths[self.under_indexes, :, j] -
                                  self.portfolio.paths[self.under_indexes, :, j-1]).to(self.device),
                    dim=0
                )

            hedge = self.get_hedge(
                time_idx=j, prev_hedge=torch.swapaxes(prev_hedge, 0, 1))

            current_cost = torch.sum(cost * torch.abs(hedge - prev_hedge), dim=0)
            costs -= current_cost
            pnl -= current_cost
            prev_hedge = hedge
        portf_payoff = self.portfolio.get_payoff().to(self.device)
        pnl -= portf_payoff
        return pnl, costs

    def compute_pnl_nbatches(self, cost, n_batches):
        pnl = torch.empty(0)
        costs = torch.empty(0)
        for _ in tqdm(range(n_batches)):
            self.portfolio.simulate()
            pnl_batch, costs_batch = self.compute_pnl(cost)
            pnl = torch.cat((pnl, pnl_batch.to('cpu')))
            costs = torch.cat((costs, costs_batch.to('cpu')))
        return pnl, costs
    