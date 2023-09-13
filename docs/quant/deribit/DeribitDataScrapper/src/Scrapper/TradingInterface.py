from __future__ import annotations

import os
import sys
import threading
import requests

# import nest_asyncio
# nest_asyncio.apply()

import asyncio
import time
from typing import Optional, Union

from InstrumentManager import AbstractInstrumentInfo
from DataBase import *
from OrderManager import OrderManager
from Utils import *
from Subsciption import *
from Strategy import *
from InstrumentManager import InstrumentManager

from SyncLib.AvailableRequests import get_ticker_by_instrument_request
from Scrapper.ScrapperWithPreSelectedMaturities import scrap_available_instruments_by_extended_config

from websocket import WebSocketApp, enableTrace, ABNF
from threading import Thread

from datetime import datetime
import logging
import json
import yaml


async def scrap_available_instruments(currency: Currency, cfg):
    """
    Функция для получения всех доступных опционов для какой-либо валюты.
    Предлагается ввод пользователем конкретного maturity
    :param currency: Валюта. BTC | ETH | SOL
    :param cfg: файл конфигурации бота
    :return: LIST[Instrument-name]
    """
    from SyncLib.AvailableRequests import get_instruments_by_currency_request
    from Utils.AvailableInstrumentType import InstrumentType
    from SyncLib.Scrapper import send_request
    import pandas as pd
    import numpy as np
    make_subscriptions_list = await send_request(get_instruments_by_currency_request(currency=currency,
                                                                                     kind=InstrumentType.OPTION,
                                                                                     expired=False), URL_TO_SCRAP=cfg)

    # Take only the result of answer. Now we have list of json contains information of option dotes.
    answer = make_subscriptions_list['result']
    available_maturities = pd.DataFrame(np.unique(list(map(lambda x: x["instrument_name"].split('-')[1], answer))))
    available_maturities.columns = ['DeribitNaming']
    available_maturities['RealNaming'] = pd.to_datetime(available_maturities['DeribitNaming'], format='%d%b%y')
    available_maturities = available_maturities.sort_values(by='RealNaming').reset_index(drop=True)
    print("Available maturities: \n", available_maturities)

    # TODO: uncomment
    selected_maturity = int(input("Select number of interested maturity "))
    # selected_maturity = -1
    if selected_maturity == -1:
        warnings.warn("Selected list of instruments is empty")
        return []
    selected_maturity = available_maturities.iloc[selected_maturity]['DeribitNaming']
    print('\nYou select:', selected_maturity)

    selected = list(map(lambda x: x["instrument_name"],
                        list(filter(lambda x: (selected_maturity in x["instrument_name"]) and (
                                x["option_type"] == "call" or "put"), answer))))

    get_underlying = await send_request(get_ticker_by_instrument_request(selected[0]),
                                        show_answer=False, URL_TO_SCRAP=cfg)
    get_underlying = get_underlying['result']['underlying_index']
    if 'SYN' not in get_underlying:
        selected.append(get_underlying)
    else:
        if cfg["raise_error_at_synthetic"]:
            raise ValueError("Cannot subscribe to order book for synthetic underlying")
        else:
            warnings.warn("Underlying is synthetic: {}".format(get_underlying))
    print("Selected Instruments")
    print(selected)

    return selected


def validate_configuration_file(configuration_path: str) -> dict:
    """
    Валидация файла конфигуратора для избегания нереализованных/некорректных конфигураций бота.
    :param configuration_path: Путь к файлу конфигурации
    :return:
    """
    with open(configuration_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    cfg["hdf5"]["hdf5_database_directory"] = f'{os.getcwd()}/{cfg["hdf5"]["hdf5_database_directory"]}'
    if type(cfg["orderBookScrapper"]["depth"]) != int:
        raise TypeError("Invalid type for scrapper configuration")
    if type(cfg["orderBookScrapper"]["test_net"]) != bool:
        raise TypeError("Invalid type for scrapper configuration")
    if type(cfg["orderBookScrapper"]["enable_database_record"]) != bool:
        raise TypeError("Invalid type for scrapper configuration")
    if type(cfg["orderBookScrapper"]["clean_database"]) != bool:
        raise TypeError("Invalid type for scrapper configuration")
    if type(cfg["orderBookScrapper"]["hearth_beat_time"]) != int:
        raise TypeError("Invalid type for scrapper configuration")
    if cfg["orderBookScrapper"]["database_daemon"] != "hdf5" and cfg["orderBookScrapper"]["database_daemon"] != "mysql":
        raise TypeError("Invalid type for scrapper configuration")
    if type(cfg["orderBookScrapper"]["add_extra_instruments"]) != list:
        raise TypeError("Invalid type for scrapper configuration")
    print(cfg["orderBookScrapper"]["scrapper_body"])
    available_subs = list(map(
        lambda x: 1 if (x != "OrderBook") and (x != "Trades") and (x != "Portfolio") and (x != "OwnOrderChange") else 0,
                              cfg["orderBookScrapper"]["scrapper_body"]))
    if sum(available_subs) != 0:
        logging.warning("Unknown subscriptions at scrapper_body")
        raise NotImplementedError
    #
    if type(cfg["record_system"]["use_batches_to_record"]) != bool:
        raise TypeError("Invalid type for record system configuration")
    if type(cfg["record_system"]["number_of_tmp_tables"]) != int:
        raise TypeError("Invalid type for record system configuration")
    if type(cfg["record_system"]["size_of_tmp_batch_table"]) != int:
        raise TypeError("Invalid type for record system configuration")

    return cfg


def subscription_map(scrapper, conf: dict) -> dict[str, AbstractSubscription]:
    """
    Creation of subscriptions objects (connect sub to db system).
    При добавлении новых подписок необходимо определить поведение в этой функции.
    :param scrapper: DeribitClient
    :param conf: configuration json object
    :return: dict sub-name <-> sub object (no connection between sub and db system)
    """
    res_dict: dict[str, AbstractSubscription] = dict()
    for sub in conf["orderBookScrapper"]["scrapper_body"]:
        if sub == "OrderBook":
            res_dict["OrderBook"]: OrderBookSubscriptionCONSTANT = \
                OrderBookSubscriptionCONSTANT(scrapper=scrapper, order_book_depth=conf["orderBookScrapper"]["depth"])
        elif sub == "Trades":
            res_dict["Trades"]: TradesSubscription = TradesSubscription(scrapper=scrapper)
        elif sub == "Portfolio":
            res_dict["Portfolio"]: UserPortfolioSubscription = UserPortfolioSubscription(scrapper=scrapper)
        elif sub == "OwnOrderChange":
            res_dict["OwnOrderChange"]: OwnOrdersSubscription = OwnOrdersSubscription(scrapper=scrapper)
    return res_dict


def net_databases_to_subscriptions(scrapper: DeribitClient) -> dict[AbstractSubscription, AbstractDataManager]:
    """
    initialize db system for every subscription in scrapper
    :param scrapper: DeribitClient
    :return: dict sub-object <-> sub db system (connection between sub and db system)
    """
    result_netting = {}
    match scrapper.configuration['orderBookScrapper']["database_daemon"]:
        case 'mysql':
            for action, subscription_type in scrapper.subscriptions_objects.items():
                if action == "OrderBook":
                    if type(scrapper.configuration['orderBookScrapper']["depth"]) == int:
                        database = MySqlDaemon(configuration_path=scrapper.configuration_path,
                                               subscription_type=subscription_type,
                                               loop=scrapper.loop)

                    elif scrapper.configuration['orderBookScrapper']["depth"] is False:
                        database = MySqlDaemon(configuration_path=scrapper.configuration_path,
                                               subscription_type=subscription_type,
                                               loop=scrapper.loop)
                    else:
                        raise ValueError('Unavailable value of depth order book mode')

                    result_netting[subscription_type] = database
                    subscription_type.plug_in_record_system(database=database)
                    time.sleep(1)
                elif action == "Trades":
                    database = MySqlDaemon(configuration_path=scrapper.configuration_path,
                                           subscription_type=subscription_type,
                                           loop=scrapper.loop)
                    result_netting[subscription_type] = database
                    subscription_type.plug_in_record_system(database=database)
                elif action == "OwnOrderChange":
                    database = MySqlDaemon(configuration_path=scrapper.configuration_path,
                                           subscription_type=subscription_type,
                                           loop=scrapper.loop)
                    result_netting[subscription_type] = database
                    subscription_type.plug_in_record_system(database=database)

                elif action == "Portfolio":
                    database = MySqlDaemon(configuration_path=scrapper.configuration_path,
                                           subscription_type=subscription_type,
                                           loop=scrapper.loop)
                    result_netting[subscription_type] = database
                    subscription_type.plug_in_record_system(database=database)

        case "hdf5":
            for action, subscription_type in scrapper.subscriptions_objects.items():
                if action == "OrderBook":
                    if type(scrapper.configuration['orderBookScrapper']["depth"]) == int:
                        database = HDF5Daemon(configuration_path=scrapper.configuration_path,
                                              subscription_type=subscription_type,
                                              loop=scrapper.loop)
                    elif scrapper.configuration['orderBookScrapper']["depth"] is False:
                        database = HDF5Daemon(configuration_path=scrapper.configuration_path,
                                              subscription_type=subscription_type,
                                              loop=scrapper.loop)
                    else:
                        raise ValueError('Unavailable value of depth order book mode')

                    result_netting[subscription_type] = database
                    subscription_type.plug_in_record_system(database=database)
                    time.sleep(1)
                elif action == "Trades":
                    database = HDF5Daemon(configuration_path=scrapper.configuration_path,
                                           subscription_type=subscription_type,
                                           loop=scrapper.loop)
                    result_netting[subscription_type] = database
                    subscription_type.plug_in_record_system(database=database)
                elif action == "OwnOrderChange":
                    database = HDF5Daemon(configuration_path=scrapper.configuration_path,
                                           subscription_type=subscription_type,
                                           loop=scrapper.loop)
                    result_netting[subscription_type] = database
                    subscription_type.plug_in_record_system(database=database)

                elif action == "Portfolio":
                    database = HDF5Daemon(configuration_path=scrapper.configuration_path,
                                           subscription_type=subscription_type,
                                           loop=scrapper.loop)
                    result_netting[subscription_type] = database
                    subscription_type.plug_in_record_system(database=database)
        case _:
            logging.warning("Unknown database daemon selected")
            scrapper.database = None

    return result_netting


class DeribitClient(Thread, WebSocketApp):
    """
    Главный клиент обеспечивающий взаимодействие всех модулей между друг другом. Также выполняет роль
    прослойки между компонентами бота и deribit. По большей части включает в себя все необходимые для работы
    атрибуты. Также открывает соединение websocket с Deribit и перенаправляет ответы от сервера всем необходимым
    методам.
    NOTE: Изначально писался таким образом, чтобы при расширении функционала не было необходимости вносить изменения
    в большую часть компонентов системы, если для расширений необходимо вносить изменения в DeribitClient, значит
    что-то не так.
    """

    websocket: Optional[WebSocketApp]
    database: Optional[Union[MySqlDaemon, HDF5Daemon]] = None
    loop: asyncio.unix_events.SelectorEventLoop
    instrument_name_instrument_id_map: AutoIncrementDict[str, AbstractInstrumentInfo] = None
    only_API_orders: bool = None
    instrument_manager: InstrumentManager = None
    order_manager: OrderManager = None
    connected_strategy: Optional[AbstractStrategy] = None
    client_currency: Optional[Currency] = None

    def __init__(self, cfg, cfg_path: dict, dev_cfg: dict, loopB, client_currency: Currency,
                 instruments_listed: list = None):

        if instruments_listed is None:
            instruments_listed = []
        self.developConfiguration = dev_cfg

        # Load configuration
        self.configuration_path = cfg_path
        self.configuration = cfg
        test_mode = self.configuration['orderBookScrapper']["test_net"]
        enable_traceback = self.configuration['orderBookScrapper']["enable_traceback"]
        enable_database_record = bool(self.configuration['orderBookScrapper']["enable_database_record"])

        self.only_API_orders = self.configuration["orderBookScrapper"]["only_api_orders_processing"]
        # Extra like BTC-PERPETUAL
        for _instrument_name in self.configuration["orderBookScrapper"]["add_extra_instruments"]:
            if _instrument_name not in instruments_listed:
                instruments_listed.append(_instrument_name)

        # Download instrument mapping
        self.instrument_name_instrument_id_map: AutoIncrementDict[str, AbstractInstrumentInfo] = AutoIncrementDict(path_to_file=
                                                                   self.configuration["record_system"][
                                                                       "instrumentNameToIdMapFile"])

        # Initialize all loops and Threads
        Thread.__init__(self)
        self.loop = loopB
        asyncio.set_event_loop(self.loop)

        # Set client currency
        self.client_currency = client_currency

        # Make subscriptions map (OrderBook, Trades, Portfolio changes e.t.c)
        self.subscriptions_objects = subscription_map(scrapper=self, conf=self.configuration)

        # Make list of instruments
        self.instruments_list = instruments_listed

        # Set exchange mode
        self.testMode = test_mode
        self.exchange_version = self._set_exchange()

        # Place initial local time
        self.time = datetime.now()

        self.websocket = None
        self.enable_traceback = enable_traceback
        # Set storages for requested data
        self.instrument_requested = set()
        if enable_database_record:
            self.subscription_type = net_databases_to_subscriptions(scrapper=self)

        # Set flag for authentication validation
        self.auth_complete: bool = False

    def add_instrument_manager(self):
        if self.configuration["externalModules"]["add_instrument_manager"]:
            self.instrument_manager = InstrumentManager(self, self.configuration,
                                                        self.loop,
                                                        ConfigRoot.DIRECTORY)
        else:
            logging.warning("No instrument manager selected by configuration")

    def add_order_manager(self):
        if self.configuration["externalModules"]["add_order_manager"]:
            self.order_manager = OrderManager()
            self.order_manager.connect_client(client=self)
        else:
            logging.warning("No order manager selected by configuration")

    def _set_exchange(self):
        """
        Don't touch me!
        Выбор адреса websocket. Тест режим / Prod режим.
        Настоятельно рекомендуется использовать testMode при работе с ордерами и подключении "непустых" стратегий.
        :return:
        """
        if self.testMode:
            print("Initialized TEST NET mode")
            return 'wss://test.deribit.com/ws/api/v2'
        else:
            print("Initialized REAL MARKET mode")
            return 'wss://www.deribit.com/ws/api/v2'

    def run(self):
        """
        Don't touch me!
        :return:
        """
        self.websocket = WebSocketApp(self.exchange_version,
                                      on_message=self._on_message, on_open=self._on_open, on_error=self._on_error,)
        if self.enable_traceback:
            enableTrace(True)
        # Run forever loop
        # self.websocket.last_ping_tm = 32
        # self.websocket.last_pong_tm = 1
        while True:
            try:
                self.websocket.run_forever(ping_interval=None, reconnect=2, skip_utf8_validation=True)
            except Exception as e:
                print(e)
                logging.error("Error at run_forever loop")
                # TODO: place here notificator
                continue

    def _on_error(self, websocket, error):
        # TODO: send Telegram notification
        logging.error(error)
        self.instrument_requested.clear()
        # import os, signal
        # os.kill(os.getpid(), signal.SIGUSR1)

    def _on_message(self, websocket, message):
        """
        Логика реакции на ответ сервера.
        :param websocket:
        :param message:
        :return:
        """
        response = json.loads(message)
        self._process_callback(response)
        # print(response)
        # Process initial order placement
        if 'result' in response:
            if 'order' in response['result']:
                if 'OwnOrderChange' in self.subscriptions_objects:
                    asyncio.run_coroutine_threadsafe(
                self.subscriptions_objects["OwnOrderChange"].process_response_from_server(response=response),
                        loop=self.loop)
        # subscriptions
        if 'method' in response:
            # Answer to heartbeat request
            if response['method'] == 'heartbeat':
                # Send test message to approve that connection is still alive
                self.send_new_request(MSG_LIST.test_message())
                return
            # TODO
            for action, sub in self.subscriptions_objects.items():
                asyncio.run_coroutine_threadsafe(sub.process_response_from_server(response=response),
                                                 loop=self.loop)

        if self.instrument_manager is not None:
            # validation requests for instrument manager
            if 'result' in response:
                # Cut subs
                if 'method' not in response:
                    # Cut auth messages
                    if 'token_type' not in response['result']:
                        if response['result'] != 'ok':
                            for dict_obj in response['result']:
                                asyncio.run_coroutine_threadsafe(self.instrument_manager.process_validation(dict_obj), self.loop)

        # Process errors
        if 'error' in response:
            self._process_error_callbacks(response=response)

    def _process_error_callbacks(self, response: dict):
        if 'message' in response['error']:
            if 'not_enough_funds' in response['error']['message']:
                logging.warning(f"{response}")
                asyncio.run_coroutine_threadsafe(self.order_manager.not_enough_funds(callback=response),
                                                 loop=self.loop)
            elif 'price_too_high' in response['error']['message']:
                logging.warning(f"{response}")
                asyncio.run_coroutine_threadsafe(self.order_manager.price_too_high(callback=response),
                                                 loop=self.loop)
            else:
                logging.error(f"Unknown error callback: | {response}")

    def _process_callback(self, response):
        logging.info(response)
        pass

    def _on_open(self, websocket):
        """
        Don't touch me!
        Создает необходимые initial запросы на подписки
        :param websocket:
        :return:
        """
        # Add Instrument Manager
        self.add_instrument_manager()
        # Set heartbeat
        self.send_new_request(MSG_LIST.set_heartbeat(
            self.configuration["orderBookScrapper"]["hearth_beat_time"]))

        logging.info("Client start his work")
        # Execute initial subscription's request pipelines
        for action, sub in self.subscriptions_objects.items():
            sub.create_subscription_request()

    def send_new_request(self, request: dict):
        """
        Don't touch me.
        Отправляет запрос на сервер. Блокирующий time-sleep вызван спецификой запросов к deribit.
        :param request:
        :return:
        """
        self.websocket.send(json.dumps(request), ABNF.OPCODE_TEXT)
        # TODO: do it better. Unsync.
        time.sleep(0.05)

    def send_block_sync_request(self, params: dict, method="get_position", _private='private') -> dict:
        """
        Послать блокирующий запрос. Реализована аутентификация.
        Пример: Используется для инициализации начальных позиций в инструментах. Блокируется до получения ответа
        :param params:
        :param method
        :param _private
        :return: dict
        """
        if self.testMode:
            _hist = 'test.'
        else:
            _hist = ''
        session = requests.Session()
        # В случае приватного запроса сначала необходимо добавить в header токен.
        if (_private == 'private') and (not self.testMode):
            import os, signal
            logging.error('Cannot pass auth with prod mode right now')
            os.kill(os.getpid(), signal.SIGUSR1)
            raise NotImplementedError
        if _private == 'private':
            client_id = \
                self.configuration["user_data"]["test_net"]["client_id"] \
                    if self.configuration["orderBookScrapper"]["test_net"] else \
                    self.configuration["user_data"]["production"]["client_id"]

            client_secret = \
                self.configuration["user_data"]["test_net"]["client_secret"] \
                    if self.configuration["orderBookScrapper"]["test_net"] else \
                    self.configuration["user_data"]["production"]["client_secret"]

            auth_params = {
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            }
            auth = session.get(f"https://{_hist}deribit.com/api/v2/public/auth", params=auth_params)
            session.headers.update(
                {"Authorization": f"Bearer {auth.json()['result']['access_token']}"}
            )

        response = session.get(f"https://{_hist}deribit.com/api/v2/{_private}/{method}", params=params)
        return response.json()

    def add_strategy(self, strategy: AbstractStrategy):
        """
        Подключение стратегии в интерфейс
        :param strategy:
        :return:
        """
        strategy.connect_client(data_provider=self)
        self.connected_strategy = strategy


