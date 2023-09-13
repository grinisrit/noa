from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING
from enum import Enum

from numpy import ndarray
from pandas import DataFrame

from Utils.MSG_LIST import auth_message
if TYPE_CHECKING:
    from Scrapper.TradingInterface import DeribitClient
    from DataBase.AbstractDataSaverManager import AbstractDataManager

    scrapper_typing = DeribitClient
    database_typing = AbstractDataManager
else:
    scrapper_typing = object
    database_typing = object


class RequestTypo(Enum):
    PUBLIC = 1
    PRIVATE = 2


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


class AbstractSubscription(ABC):
    """
    Класс абстрактной подписки Deribit. (см. https://docs.deribit.com/#subscriptions)
    При имплементации необходимо определить названия таблиц где будут храниться данные.
    Также следует определить запрос который отправляется на Deribit для запроса подписки.
    Следует определить поведение подписки на приходящий response от сервера
    (note: src не выполняет предварительной фильтрации ответов,
    и все пришедшие данные крутит через все подписки)
    """
    tables_names: List[str]
    tables_names_creation: List[str]
    number_of_columns: int
    database: database_typing

    request_typo: RequestTypo = None

    def __init__(self, scrapper: scrapper_typing, request_typo: RequestTypo):
        self.scrapper = scrapper
        self._place_here_tables_names_and_creation_requests()
        self.developConfiguration = scrapper.developConfiguration

        self.client_id = \
            self.scrapper.configuration["user_data"]["test_net"]["client_id"] \
                if self.scrapper.configuration["orderBookScrapper"]["test_net"] else \
                self.scrapper.configuration["user_data"]["production"]["client_id"]

        self.client_secret = \
            self.scrapper.configuration["user_data"]["test_net"]["client_secret"] \
                if self.scrapper.configuration["orderBookScrapper"]["test_net"] else \
                self.scrapper.configuration["user_data"]["production"]["client_secret"]

        self.request_typo = request_typo

    def plug_in_record_system(self, database: database_typing):
        self.database = database

    @abstractmethod
    def _place_here_tables_names_and_creation_requests(self):
        pass

    @abstractmethod
    def create_columns_list(self) -> list[str]:
        """
        Возвращает список из колонок в БД.
        :return:
        """
        pass

    @abstractmethod
    async def _create_subscription_request(self) -> str:
        """
        Запрос для "заказа" подписки
        :return:
        """
        pass

    def create_subscription_request(self) -> str:
        if (not self.scrapper.auth_complete) and (self.request_typo == RequestTypo.PRIVATE):
            self.send_auth_message()
            self.scrapper.auth_complete = True

        self._create_subscription_request()

    @abstractmethod
    async def _process_response(self, response: dict):
        """
        Определение реакции подписки на пришедший ответ от deribit сервера. Должен фильтровать запрос
        (на соответствие response рассматриваемой подписки)
        :param response:
        :return:
        """
        pass

    async def process_response_from_server(self, response: dict):
        # # TODO: remove it after test
        # _r_sl = random.randint(10, 20)
        # if _r_sl != 0:
        #     await asyncio.sleep(_r_sl)
        _res = await self._process_response(response=response)
        return _res

    @abstractmethod
    def extract_data_from_response(self, input_response: dict) -> ndarray:
        """
        Конвертация ответа от сервера в ndarray со значениями для записи в БД.
        :param input_response:
        :return:
        """
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

    def send_auth_message(self):
        self.scrapper.send_new_request(auth_message(client_id=self.client_id,
                                                    client_secret=self.client_secret))
        # TODO: make validation. Work solution
        self.scrapper.auth_complete = True
