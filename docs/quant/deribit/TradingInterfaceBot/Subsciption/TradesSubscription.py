import asyncio
import pprint

from docs.quant.deribit.TradingInterfaceBot.Subsciption.AbstractSubscription import AbstractSubscription, flatten
from docs.quant.deribit.TradingInterfaceBot.Utils import MSG_LIST
from docs.quant.deribit.TradingInterfaceBot.DataBase.mysqlRecording.cleanUpRequestsLimited import REQUEST_TO_CREATE_TRADES_TABLE

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


class TradesSubscription(AbstractSubscription):
    tables_names = ["Trades_table_{}"]

    def __init__(self, scrapper: scrapper_typing):
        self.tables_names = [f"Trades_table_test"]
        self.tables_names_creation = list(map(REQUEST_TO_CREATE_TRADES_TABLE, self.tables_names))

        super(TradesSubscription, self).__init__(scrapper=scrapper)
        self.number_of_columns = 7
        self.instrument_name_instrument_id_map = self.scrapper.instrument_name_instrument_id_map

    def _place_here_tables_names_and_creation_requests(self):
        self.tables_names = [f"Trades_table_test"]
        self.tables_names_creation = list(map(REQUEST_TO_CREATE_TRADES_TABLE, self.tables_names))

    def create_columns_list(self) -> List[str]:
        columns = ["CHANGE_ID", "TIMESTAMP_VALUE", "TRADE_ID", "PRICE", "NAME_INSTRUMENT", "DIRECTION", "AMOUNT"]
        columns = flatten(columns)
        return columns

    async def _process_response(self, response: dict):
        # SUBSCRIPTION processing
        if response['method'] == "subscription":
            # ORDER BOOK processing. For constant book depth
            if 'params' in response:
                if 'channel' in response['params']:
                    if 'trades' in response['params']['channel']:
                        if self.scrapper.connected_strategy is not None:
                            await self.scrapper.connected_strategy.on_trade_update(callback=response)

                        if self.database:
                            await self.database.add_data(
                                update_line=self.extract_data_from_response(input_response=response)
                            )
                            return 1

    def extract_data_from_response(self, input_response: dict) -> ndarray:
        _full_ndarray = []
        for data_object in input_response['params']['data']:
            _change_id = 666

            _timestamp = data_object['timestamp']
            _instrument_name = self.instrument_name_instrument_id_map[
                data_object['instrument_name']]
            _trade_id = data_object['trade_id']
            _price = data_object["price"]
            _direction = 1 if data_object["direction"] == "buy" else -1
            _amount = data_object["amount"]
            _full_ndarray.append(
                [_change_id, _timestamp, _trade_id, _price, _instrument_name, _direction, _amount]
            )
        return np.array(_full_ndarray)


    def create_subscription_request(self):
        self._trades_subscription_request()

    def _record_to_daemon_database_pipeline(self, record_dataframe: DataFrame, tag_of_data: str) -> DataFrame:
        if 'CHANGE_ID' in record_dataframe.columns:
            return record_dataframe.iloc[:, 1:]
        return record_dataframe

    def _trades_subscription_request(self):
        # TODO for _instrument_name in self.scrapper.instruments_list:
        for _instrument_name in self.scrapper.instruments_list:
            subscription_message = \
                MSG_LIST.make_trades_subscription_request_by_instrument(
                    instrument_name=_instrument_name,
                )
            self.scrapper.send_new_request(request=subscription_message)

        # Extra like BTC-PERPETUAL
        for _instrument_name in self.scrapper.configuration["orderBookScrapper"]["add_extra_instruments"]:
            subscription_message = \
                MSG_LIST.make_trades_subscription_request_by_instrument(
                    instrument_name=_instrument_name,
                )
            self.scrapper.send_new_request(request=subscription_message)
