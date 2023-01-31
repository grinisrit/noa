import asyncio
from abc import ABC, abstractmethod
import random
from typing import List, TYPE_CHECKING
from numpy import ndarray
from pandas import DataFrame


if TYPE_CHECKING:
    from docs.quant.deribit.OrderBookScrapper.Scrapper.DeribitClient import DeribitClient

    scrapper_typing = DeribitClient
else:
    scrapper_typing = object

# Block with developing module | START
import yaml
import sys

with open(sys.path[1] + "/docs/quant/deribit/OrderBookScrapper/developerConfiguration.yaml", "r") as _file:
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
    async def create_subscription_request(self) -> str:
        pass

    @abstractmethod
    async def _process_response(self, response: dict):
        pass

    async def process_response_from_server(self, response: dict):
        # TODO: remove it after test
        _r_sl = random.randint(0, 4)
        if _r_sl != 0:
            await asyncio.sleep(_r_sl)
        _res = await self._process_response(response=response)
        return _res

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
