from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional
from docs.quant.deribit.TradingInterfaceBotJulyVersion.Utils import OrderStructure
from docs.quant.deribit.TradingInterfaceBotJulyVersion.InstrumentManager import AbstractInstrument
from docs.quant.deribit.TradingInterfaceBotJulyVersion.ExternalModules import AbstractExternal

if TYPE_CHECKING:
    from docs.quant.deribit.TradingInterfaceBotJulyVersion.Scrapper.TradingInterface import DeribitClient
    scrapper_type = DeribitClient
else:
    scrapper_type = object


class AbstractStrategy(ABC):
    data_provider: scrapper_type
    open_orders: dict[int, OrderStructure] = dict()
    all_orders: dict[int, OrderStructure] = dict()

    connected_externals: Optional[Dict[str, AbstractExternal]]

    def __init__(self):
        for external in self.connected_externals.values():
            external.connect_strategy(self)

    def connect_client(self, data_provider: scrapper_type):
        self.data_provider = data_provider

    async def on_order_book_update(self, abstractInstrument: AbstractInstrument):
        await self._on_order_book_update(abstractInstrument)
        for external in self.connected_externals.values():
            await external.on_order_book_update(abstractInstrument)

    async def on_trade_update(self, abstractInstrument: AbstractInstrument):
        await self._on_trade_update(abstractInstrument)
        for external in self.connected_externals.values():
            await external.on_trade_update(abstractInstrument)

    async def on_order_update(self, updatedOrder: OrderStructure):
        await self._on_order_update(updatedOrder)

    async def on_tick_update(self, callback: dict):
        await self._on_tick_update(callback)
        for external in self.connected_externals.values():
            await external.on_tick_update(callback)

    async def on_position_miss_match(self):
        await self._on_position_miss_match()

    async def on_not_enough_fund(self, callback: dict):
        await self._on_not_enough_fund(callback)

    async def on_order_creation(self, createdOrder: OrderStructure):
        await self._on_order_creation(createdOrder)

    async def price_too_high(self, callback: dict):
        await self._price_too_high(callback)

    async def on_api_external_order(self, callback: dict):
        await self._on_api_external_order(callback)

    # IMPLEMENT PART
    @abstractmethod
    async def _on_order_book_update(self, abstractInstrument: AbstractInstrument):
        pass

    @abstractmethod
    async def _on_trade_update(self, abstractInstrument: AbstractInstrument):
        pass

    @abstractmethod
    async def _on_order_update(self, updatedOrder: OrderStructure):
        pass

    @abstractmethod
    async def _on_tick_update(self, callback: dict):
        pass

    @abstractmethod
    async def _on_position_miss_match(self):
        pass

    @abstractmethod
    async def _on_not_enough_fund(self, callback: dict):
        pass

    @abstractmethod
    async def _on_order_creation(self, createdOrder: OrderStructure):
        pass

    @abstractmethod
    async def _price_too_high(self, callback: dict):
        pass

    @abstractmethod
    async def _on_api_external_order(self, callback: dict):
        pass