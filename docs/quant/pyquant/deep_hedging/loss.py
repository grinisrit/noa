import math
from abc import ABC, abstractmethod
import torch


class BaseLoss(torch.nn.Module, ABC):
    """Base class for hedging criteria."""

    @abstractmethod
    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        """Returns the loss of the PnL distribution.
        Args:
            pnl: The PnL distribution. Shape: (N, ).
        Returns:
            Loss. Shape: (1, ).
        """
        pass


class EntropicLoss(BaseLoss):
    def __init__(self, risk_aversion):
        super().__init__()
        self.a = risk_aversion

    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        return -torch.mean(-torch.exp(-self.a * pnl))


class EntropicRiskMeasure(BaseLoss):
    def __init__(self, risk_aversion):
        super().__init__()
        self.a = risk_aversion

    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        return (1/self.a) * torch.log(-torch.mean(-torch.exp(-self.a * pnl)))


class ExpectedShortfall(BaseLoss):
    def __init__(self, quantile):
        super().__init__()
        self.q = quantile

    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        return -pnl.topk(math.ceil(self.q * pnl.numel()), largest=False).values.mean()
    