
from docs.quant.deribit.TradingInterfaceBot.Subsciption.AbstractSubscription import AbstractSubscription, flatten, RequestTypo

from docs.quant.deribit.TradingInterfaceBot.Utils import *
from docs.quant.deribit.TradingInterfaceBot.Utils import REQUEST_TO_CREATE_LIMITED_ORDER_BOOK_CONTENT


from numpy import ndarray
from functools import partial
from pandas import DataFrame
from typing import List, TYPE_CHECKING
import logging
import numpy as np

if TYPE_CHECKING:
    from docs.quant.deribit.TradingInterfaceBot.Scrapper.TradingInterface import DeribitClient

    scrapper_typing = DeribitClient
else:
    scrapper_typing = object


class OrderBookSubscriptionCONSTANT(AbstractSubscription):
    tables_names = ["TABLE_DEPTH_{}"]

    def __init__(self, scrapper: scrapper_typing, order_book_depth: int):
        self.depth: int = order_book_depth
        self.tables_names = [f"TABLE_DEPTH_{self.depth}"]
        self.tables_names_creation = list(map(partial(REQUEST_TO_CREATE_LIMITED_ORDER_BOOK_CONTENT,
                                                      depth_size=self.depth), self.tables_names))

        super(OrderBookSubscriptionCONSTANT, self).__init__(scrapper=scrapper, request_typo=RequestTypo.PUBLIC)
        self.number_of_columns = self.depth * 4 + 3

        self.instrument_name_instrument_id_map = self.scrapper.instrument_name_instrument_id_map

    def _place_here_tables_names_and_creation_requests(self):
        self.tables_names = [f"TABLE_DEPTH_{self.depth}"]
        self.tables_names_creation = list(map(partial(REQUEST_TO_CREATE_LIMITED_ORDER_BOOK_CONTENT,
                                                      depth_size=self.depth), self.tables_names))

    def create_columns_list(self) -> List[str]:
        if self.depth == 0:
            raise NotImplementedError
        else:
            columns = ["CHANGE_ID", "NAME_INSTRUMENT", "TIMESTAMP_VALUE"]
            columns.extend(map(lambda x: [f"BID_{x}_PRICE", f"BID_{x}_AMOUNT"], range(self.depth)))
            columns.extend(map(lambda x: [f"ASK_{x}_PRICE", f"ASK_{x}_AMOUNT"], range(self.depth)))

            columns = flatten(columns)
            return columns

    async def _process_response(self, response: dict):
        # SUBSCRIPTION processing
        if response['method'] == "subscription":
            # ORDER BOOK processing. For constant book depth
            if 'change' and 'type' not in response['params']['data']:
                if self.scrapper.connected_strategy is not None:
                    await self.scrapper.connected_strategy.on_order_book_update(callback=response)
                if self.database:
                    await self.database.add_data(
                        update_line=self.extract_data_from_response(input_response=response)
                    )
                return 1

    def extract_data_from_response(self, input_response: dict) -> ndarray:
        _change_id = input_response['params']['data']['change_id']
        _timestamp = input_response['params']['data']['timestamp']
        _instrument_name = self.instrument_name_instrument_id_map[
            input_response['params']['data']['instrument_name']]
        _bids = sorted(input_response['params']['data']['bids'], key=lambda x: x[0], reverse=True)
        _asks = sorted(input_response['params']['data']['asks'], key=lambda x: x[0], reverse=False)
        _bids_insert_array = [[-1.0, -1.0] for _i in range(self.depth)]
        _asks_insert_array = [[-1.0, -1.0] for _i in range(self.depth)]

        _pointer = self.depth - 1
        for i, bid in enumerate(_bids[:self.depth]):
            _bids_insert_array[_pointer] = bid
            _pointer -= 1

        _pointer = self.depth - 1
        for i, ask in enumerate(_asks[:self.depth]):
            _asks_insert_array[i] = ask
            _pointer -= 1
        _bids_insert_array.extend(_asks_insert_array)
        _update_line = [_instrument_name, _timestamp]
        _update_line.extend(_bids_insert_array)
        _update_line.insert(0, 0)
        _update_line = np.array(flatten(_update_line))
        del _bids, _asks, _bids_insert_array, _asks_insert_array, _pointer
        return _update_line

    def make_new_subscribe_constant_depth_book(self, instrument_name: str,
                                               type_of_data="book",
                                               interval="100ms",
                                               depth=None,
                                               group=None):
        if instrument_name not in self.scrapper.instrument_requested:
            subscription_message = MSG_LIST.make_subscription_constant_book_depth(instrument_name,
                                                                                  type_of_data=type_of_data,
                                                                                  interval=interval,
                                                                                  depth=depth,
                                                                                  group=group)

            self.scrapper.send_new_request(request=subscription_message)
            self.scrapper.instrument_requested.add(instrument_name)
        else:
            logging.warning(f"Instrument {instrument_name} already subscribed")

    def _create_subscription_request(self):
        # Send all subscriptions
        for _instrument_name in self.scrapper.instruments_list:
            self.make_new_subscribe_constant_depth_book(instrument_name=_instrument_name,
                                                        depth=self.scrapper.configuration["orderBookScrapper"]["depth"],
                                                        group=self.scrapper.configuration["orderBookScrapper"][
                                                            "group_in_limited_order_book"])

        # Extra like BTC-PERPETUAL
        for _instrument_name in self.scrapper.configuration["orderBookScrapper"]["add_extra_instruments"]:
            print("Extra:", _instrument_name)
            self.make_new_subscribe_constant_depth_book(instrument_name=_instrument_name,
                                                        depth=self.scrapper.configuration["orderBookScrapper"]["depth"],
                                                        group=self.scrapper.configuration["orderBookScrapper"][
                                                            "group_in_limited_order_book"])

    def _record_to_daemon_database_pipeline(self, record_dataframe: DataFrame, tag_of_data: str) -> DataFrame:
        if 'CHANGE_ID' in record_dataframe.columns:
            return record_dataframe.iloc[:, 1:]
        return record_dataframe
