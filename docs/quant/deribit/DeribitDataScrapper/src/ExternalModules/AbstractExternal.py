from abc import ABC, abstractmethod
from InstrumentManager import AbstractInstrument
from typing import TYPE_CHECKING

strategyType = object
if TYPE_CHECKING:
    from Strategy import AbstractStrategy
    strategyType = AbstractStrategy


class AbstractExternal(ABC):
    strategy: strategyType

    @abstractmethod
    async def on_order_book_update(self, abstractInstrument: AbstractInstrument):
        pass

    @abstractmethod
    async def on_trade_update(self, abstractInstrument: AbstractInstrument):
        pass

    @abstractmethod
    async def on_tick_update(self, callback: dict):
        pass

    def connect_strategy(self, strategy: strategyType):
        self.strategy = strategy
