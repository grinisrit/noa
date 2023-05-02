
from docs.quant.deribit.TradingInterfaceBot.Subsciption.AbstractSubscription import AbstractSubscription, flatten, RequestTypo
from docs.quant.deribit.TradingInterfaceBot.Utils import *

from numpy import ndarray
from pandas import DataFrame
from typing import List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from docs.quant.deribit.TradingInterfaceBot.Scrapper.TradingInterface import DeribitClient

    scrapper_typing = DeribitClient
else:
    scrapper_typing = object


class OwnOrdersSubscription(AbstractSubscription):
    tables_names = ["Trades_table_{}"]

    def __init__(self, scrapper: scrapper_typing):
        self.tables_names = [f"User_orders_test"]
        self.tables_names_creation = list(map(REQUEST_TO_CREATE_OWN_ORDERS_TABLE, self.tables_names))

        super(OwnOrdersSubscription, self).__init__(scrapper=scrapper, request_typo=RequestTypo.PRIVATE)
        self.number_of_columns = 13
        self.instrument_name_instrument_id_map = self.scrapper.instrument_name_instrument_id_map

    def _place_here_tables_names_and_creation_requests(self):
        self.tables_names = [f"User_orders_test"]
        self.tables_names_creation = list(map(REQUEST_TO_CREATE_OWN_ORDERS_TABLE, self.tables_names))

    def create_columns_list(self) -> List[str]:
        columns = ["CHANGE_ID", "CREATION_TIMESTAMP", "LAST_UPDATE_TIMESTAMP", "NAME_INSTRUMENT", "ORDER_TYPE",
                   "ORDER_STATE", "ORDER_ID", "FILLED_AMOUNT", "COMMISSION", "AVERAGE_PRICE", "PRICE",
                   "DIRECTION", "AMOUNT"]
        columns = flatten(columns)
        return columns

    async def _process_response(self, response: dict):
        if 'result' in response:
            if 'order' in response['result']:
                if self.scrapper.order_manager is not None:
                    await self.scrapper.order_manager.process_order_callback(callback=response)

        # SUBSCRIPTION processing
        if response['method'] == "subscription":
            # ORDER BOOK processing. For constant book depth
            if 'params' in response:
                if 'channel' in response['params']:
                    if 'orders' in response['params']['channel']:
                        if self.scrapper.order_manager is not None:
                            await self.scrapper.order_manager.process_order_callback(callback=response)

                        if self.database:
                            await self.database.add_data(
                                update_line=self.extract_data_from_response(input_response=response)
                            )
                            return 1

    def extract_data_from_response(self, input_response: dict) -> ndarray:
        _full_ndarray = []
        data_object = input_response['params']['data']

        _change_id = 666
        _creation_time = data_object["creation_timestamp"]
        _last_update = data_object["last_update_timestamp"]
        _instrument_name = self.instrument_name_instrument_id_map[
            data_object['instrument_name']]
        _order_type = data_object["order_type"]
        _order_state = data_object["order_state"]
        _order_id = data_object["order_id"]
        _filled_amount = data_object["filled_amount"]
        _commission = data_object["commission"]
        _average_price = data_object["average_price"]
        _price = data_object["price"]
        _direction = 1 if data_object["direction"] == "buy" else -1
        _amount = data_object["amount"]

        _full_ndarray = np.array(
            [_change_id, _creation_time, _last_update, _instrument_name, _order_type, _order_state, _order_id,
             _filled_amount, _commission, _average_price, _price, _direction, _amount]
        )
        return np.array(_full_ndarray)

    def _create_subscription_request(self):
        self.scrapper.send_new_request(MSG_LIST.auth_message(client_id=self.client_id,
                                                             client_secret=self.client_secret))

        self._user_orders_change_subscription_request()

    def _record_to_daemon_database_pipeline(self, record_dataframe: DataFrame, tag_of_data: str) -> DataFrame:
        if 'CHANGE_ID' in record_dataframe.columns:
            return record_dataframe.iloc[:, 1:]
        return record_dataframe

    def _user_orders_change_subscription_request(self):
        for _instrument_name in self.scrapper.instruments_list:
            subscription_message = \
                MSG_LIST.make_user_orders_subscription_request_by_instrument(
                    instrument_name=_instrument_name,
                )
            self.scrapper.send_new_request(request=subscription_message)

        # Extra like BTC-PERPETUAL
        for _instrument_name in self.scrapper.configuration["orderBookScrapper"]["add_extra_instruments"]:
            subscription_message = \
                MSG_LIST.make_user_orders_subscription_request_by_instrument(
                    instrument_name=_instrument_name,
                )
            self.scrapper.send_new_request(request=subscription_message)
