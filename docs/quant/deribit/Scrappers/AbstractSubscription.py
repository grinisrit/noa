import logging
from abc import ABC, abstractmethod
from functools import partial
from typing import List, TYPE_CHECKING
import numpy as np
from numpy import ndarray
from pandas import DataFrame

from docs.quant.deribit.DataBase.mysqlRecording.cleanUpRequestsLimited import \
    REQUEST_TO_CREATE_LIMITED_ORDER_BOOK_CONTENT
from docs.quant.deribit.Utils import MSG_LIST

if TYPE_CHECKING:
    from docs.quant.deribit.Scrappers.DeribitClient import DeribitClient

    scrapper_typing = DeribitClient
else:
    scrapper_typing = object

# Block with developing module | START
import yaml
import sys

with open(sys.path[1] + "/docs/quant/deribit/developerConfiguration.yaml", "r") as _file:
    developConfiguration = yaml.load(_file, Loader=yaml.FullLoader)
del _file
# Block with developing module | END


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


class AbstractSubscription(ABC):
    tables_names: List[str]
    tables_names_creation: List[str]

    def __init__(self, scrapper: scrapper_typing):
        self.scrapper = scrapper
        self._place_here_tables_names_and_creation_requests()

    @abstractmethod
    def _place_here_tables_names_and_creation_requests(self):
        pass

    @abstractmethod
    def create_columns_list(self) -> list[str]:
        pass

    @abstractmethod
    def create_subscription_request(self) -> str:
        pass

    @abstractmethod
    def _process_response(self, response: dict):
        pass

    def process_response_from_server(self, response: dict):
        return self._process_response(response=response)

    @abstractmethod
    def extract_data_from_response(self, input_response: dict) -> ndarray:
        pass

    @abstractmethod
    def _record_to_daemon_database_pipeline(self, record_dataframe: DataFrame, tag_of_data: str) -> DataFrame:
        """
        Need to be implemented. Creates request for database daemon (uses it methods | convert data | e.t.c).
        For example for unlimited depth transfer
        :param record_dataframe:
        :param tag_of_data: Can be used for example for logical if for unlimited depth
        :return:
        """
        pass

    def record_to_database(self, record_dataframe: DataFrame, tag_of_data: str = None) -> DataFrame:
        """
        Interface for creation pipeline for recording to database. for example preprocessing or validation data.
        Also make a copy of data to solve problems that can be caused with mutations.
        :param record_dataframe:
        :param tag_of_data:
        :return:
        """
        return self._record_to_daemon_database_pipeline(record_dataframe=record_dataframe.copy(),
                                                        tag_of_data=tag_of_data)


class OrderBookSubscriptionCONSTANT(AbstractSubscription):
    tables_names = ["TABLE_DEPTH_{}"]

    def __init__(self, scrapper: scrapper_typing, order_book_depth: int):
        self.depth: int = order_book_depth
        self.tables_names = [f"TABLE_DEPTH_{self.depth}"]
        self.tables_names_creation = list(map(partial(REQUEST_TO_CREATE_LIMITED_ORDER_BOOK_CONTENT,
                                                      depth_size=self.depth), self.tables_names))

        super(OrderBookSubscriptionCONSTANT, self).__init__(scrapper=scrapper)

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

    def _process_response(self, response: dict):
        # SUBSCRIPTION processing
        if response['method'] == "subscription":

            # ORDER BOOK processing. For constant book depth
            if 'change' and 'type' not in response['params']['data']:
                if self.scrapper.database:
                    self.scrapper.database.add_data(
                        update_line=self.extract_data_from_response(input_response=response)
                    )
                return

    def extract_data_from_response(self, input_response: dict) -> ndarray:
        _change_id = input_response['params']['data']['change_id']
        _timestamp = input_response['params']['data']['timestamp']
        _instrument_name = self.scrapper.database.instrument_name_instrument_id_map[input_response['params']['data']['instrument_name']]
        _bids = sorted(input_response['params']['data']['bids'], key=lambda x: x[0], reverse=True)
        _asks = sorted(input_response['params']['data']['asks'], key=lambda x: x[0], reverse=False)
        _bids_insert_array = [[-1.0, -1.0] for _i in range(self.scrapper.database.depth_size)]
        _asks_insert_array = [[-1.0, -1.0] for _i in range(self.scrapper.database.depth_size)]

        _pointer = self.scrapper.database.depth_size - 1
        for i, bid in enumerate(_bids[:self.scrapper.database.depth_size]):
            _bids_insert_array[_pointer] = bid
            _pointer -= 1

        _pointer = self.scrapper.database.depth_size - 1
        for i, ask in enumerate(_asks[:self.scrapper.database.depth_size]):
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

    def create_subscription_request(self):
        print("Start")
        # Set heartbeat
        self.scrapper.send_new_request(MSG_LIST.set_heartbeat(
            self.scrapper.configuration["orderBookScrapper"]["hearth_beat_time"]))
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


class NullSub(AbstractSubscription):

    def _place_here_tables_names_and_creation_requests(self):
        self.tables_names = ["TEST_TABLE_NULL"]
        self.tables_names_creation = list(map(partial(REQUEST_TO_CREATE_LIMITED_ORDER_BOOK_CONTENT,
                                                      depth_size=2), self.tables_names))

    def create_columns_list(self) -> list[str]:
        columns = ["CHANGE_ID", "NAME_INSTRUMENT", "TIMESTAMP_VALUE"]
        columns.extend(map(lambda x: [f"BID_{x}_PRICE", f"BID_{x}_AMOUNT"], range(2)))
        columns.extend(map(lambda x: [f"ASK_{x}_PRICE", f"ASK_{x}_AMOUNT"], range(2)))

        columns = flatten(columns)
        columns[0] = "TEST_UNIT_TABLE"
        columns[1] = "UNKNOWN_TABLE"
        return columns

    def create_subscription_request(self) -> str:
        pass

    def _process_response(self, response: dict):
        pass

    def extract_data_from_response(self, input_response: dict) -> ndarray:
        pass

    def _record_to_daemon_database_pipeline(self, record_dataframe: DataFrame, tag_of_data: str) -> DataFrame:
        pass


