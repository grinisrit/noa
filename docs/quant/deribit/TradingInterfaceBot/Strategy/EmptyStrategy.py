from typing import Optional

from docs.quant.deribit.TradingInterfaceBot.Strategy.AbstractStrategy import AbstractStrategy
from docs.quant.deribit.TradingInterfaceBot.Strategy.Utils.Order import OrderStructure, OrderType


class EmptyStrategy(AbstractStrategy):
    order_border_time: int = 5  # SEC
    order_border_time *= 1_000

    # off course this is really bad. Do not do like this, but it is a really fast solution to validate anything.
    order_pipeline: dict[str, Optional[OrderStructure]] = {'LimitOrder': None,
                                                           'MarketOrder': None}

    async def on_order_book_update(self, callback: dict):
        pass

    async def on_trade_update(self, callback: dict):
        pass

    async def on_order_update(self, callback: dict):
        pass

    async def on_tick_update(self, callback: dict):
        pass

    def place_order(self, order_side: str, instrument_name: str, amount: int,
                    order_type: OrderType, order_tag: str = 'defaultTag', order_price=None):
        pass

    def cancel_order(self, order_id: int):
        pass