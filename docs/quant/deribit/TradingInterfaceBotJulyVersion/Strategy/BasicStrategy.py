import logging
import time
from typing import Optional

from docs.quant.deribit.TradingInterfaceBotJulyVersion.Strategy.AbstractStrategy import AbstractStrategy
from docs.quant.deribit.TradingInterfaceBotJulyVersion.Utils import OrderStructure, OrderType, \
    convert_deribit_order_type_to_structure, convert_deribit_order_status_to_structure, OrderState
from docs.quant.deribit.TradingInterfaceBotJulyVersion.Utils import MSG_LIST


class BaseStrategy(AbstractStrategy):
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
        # pprint.pprint(callback)
        data = callback['params']['data']
        if int(data['order_id']) not in self.open_orders:
            # print("Add to map new order")
            _new_order = OrderStructure(order_id=data['order_id'],
                                        open_time=data['creation_timestamp'],
                                        price=data['price'],
                                        executed_price=data['average_price'],
                                        total_commission=data['commission'],
                                        direction=data['direction'],
                                        order_amount=data['amount'],
                                        filled_amount=data['filled_amount'],
                                        last_update_time=data['last_update_timestamp'],
                                        order_exist_time=0,
                                        order_border_time=self.order_border_time,
                                        instrument=data['instrument_name'],
                                        order_type=convert_deribit_order_type_to_structure(data["order_type"]),
                                        order_state=convert_deribit_order_status_to_structure(data["order_state"]))

            if (_new_order.order_type == OrderType.MARKET) and (_new_order.order_state != OrderState.REJECTED):
                logging.info("MARKET ORDER HAS BEEN OPENED")
                self.order_pipeline['MarketOrder'] = _new_order
                self.all_orders[int(data['order_id'])] = _new_order
                self.open_orders[int(data['order_id'])] = _new_order
            if _new_order.order_type == OrderType.LIMIT and (_new_order.order_state != OrderState.REJECTED):
                logging.info("LIMIT ORDER HAS BEEN OPENED")
                self.order_pipeline['LimitOrder'] = _new_order
                self.all_orders[int(data['order_id'])] = _new_order
                self.open_orders[int(data['order_id'])] = _new_order
        else:
            # print("Process existed order")
            order = self.open_orders[int(data['order_id'])]
            order.order_id = data['order_id']
            order.open_time = data['creation_timestamp']
            order.price = data['price']
            order.executed_price = data['average_price']
            order.total_commission = data['commission']
            order.direction = data['direction']
            order.order_amount = data['amount']
            order.filled_amount = data['filled_amount']
            order.last_update_time = data['last_update_timestamp']
            order.order_exist_time = time.time() * 1_000 - order.open_time
            order.order_border_time = self.order_border_time
            order.instrument = data['instrument_name']
            order.order_type = convert_deribit_order_type_to_structure(data["order_type"])
            order.order_state = convert_deribit_order_status_to_structure(data["order_state"])

            if (order.order_type == OrderType.LIMIT) and (order.order_state == OrderState.FILLED):
                logging.info("LIMIT ORDER HAS BEEN FILLED")
                self.order_pipeline['LimitOrder'] = None
                self.open_orders.pop(int(data['order_id']))
            if (order.order_type == OrderType.MARKET) and (order.order_state == OrderState.FILLED):
                logging.info("MARKET ORDER HAS BEEN FILLED")
                self.order_pipeline['MarketOrder'] = None
                self.open_orders.pop(int(data['order_id']))

            # Process cancel
            if (order.order_type == OrderType.MARKET) and (order.order_state == OrderState.CANCELLED):
                logging.info("MARKET ORDER HAS BEEN CANCELLED BY STRATEGY")
                self.order_pipeline['MarketOrder'] = None
                self.open_orders.pop(int(data['order_id']))

            if (order.order_type == OrderType.LIMIT) and (order.order_state == OrderState.CANCELLED):
                logging.info("LIMIT ORDER HAS BEEN CANCELLED BY STRATEGY")
                self.order_pipeline['LimitOrder'] = None
                self.open_orders.pop(int(data['order_id']))

    async def on_tick_update(self, callback: dict):
        # print("===" * 20)
        # pprint.pprint(self.open_orders)
        # print("***" * 20)
        # pprint.pprint(self.all_orders)
        # print("===" * 20)
        # Process filled market (take a sleep)
        if self.order_pipeline["MarketOrder"] is not None:
            if (time.time() * 1_000 - self.order_pipeline["MarketOrder"].open_time) > self.order_pipeline["MarketOrder"].order_border_time:
                # print("Deleting blocker market order")
                logging.info("MARKET ORDER TIME BLOCKER HAS BEEN DELETED")
                _open_key = int(self.order_pipeline["MarketOrder"].order_id)
                self.order_pipeline['MarketOrder'] = None
                self.open_orders.pop(_open_key)

        for order in self.open_orders.values():
            order.order_exist_time = time.time() * 1_000 - order.open_time

            # print("++++++" * 30)
            # print("SEND CANCELL CAUSED BY TIME FOR")
            # pprint.pprint(order)
            if order.order_exist_time > order.order_border_time:
                logging.info(f"{order.order_type.deribit_name} ORDER NEED TO BE CANCELED CAUSE LIFE EXPIRATION")
                self.cancel_order(order_id=order.order_id)
                pass

        # If no market
        if self.order_pipeline['MarketOrder'] is None:
            # Open Market
            self.place_order(order_side="buy", instrument_name='BTC-PERPETUAL', amount=10,
                             order_type=OrderType.MARKET, order_tag='none')
            pass
        # If no limit
        if self.order_pipeline['LimitOrder'] is None:
            # Open Limit
            self.place_order(order_side="buy", instrument_name='BTC-PERPETUAL', amount=10,
                             order_type=OrderType.LIMIT, order_tag='none', order_price=21000)
            pass

    def place_order(self, order_side: str, instrument_name: str, amount: int,
                    order_type: OrderType, order_tag: str = 'defaultTag', order_price=None):

        self.data_provider.send_new_request(
            request=MSG_LIST.order_request(order_side, instrument_name, amount, order_type.deribit_name,
                                           order_tag, order_price))

    def cancel_order(self, order_id: int):
        self.data_provider.send_new_request(
            request=MSG_LIST.cancel_order_request(order_id=order_id))