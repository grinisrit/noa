from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from docs.quant.deribit.TradingInterfaceBot.Strategy.Utils.Order import OrderStructure

if TYPE_CHECKING:
    from docs.quant.deribit.TradingInterfaceBot.Scrapper.TradingInterface import DeribitClient
    scrapper_type = DeribitClient
else:
    scrapper_type = object


class AbstractStrategy(ABC):
    data_provider: scrapper_type
    open_orders: dict[int, OrderStructure] = dict()
    all_orders: dict[int, OrderStructure] = dict()

    def connect_data_provider(self, data_provider: scrapper_type):
        self.data_provider = data_provider

    @abstractmethod
    async def on_order_book_update(self, callback: dict):
        pass

    @abstractmethod
    async def on_trade_update(self, callback: dict):
        pass

    @abstractmethod
    async def on_order_update(self, callback: dict):
        pass

    @abstractmethod
    async def on_tick_update(self, callback: dict):
        pass
