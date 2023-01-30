import threading

import nest_asyncio
nest_asyncio.apply()
import asyncio
import random

from tqdm import tqdm
import time
import warnings
from typing import Optional, Union

from docs.quant.deribit.OrderBookScrapper.DataBase.HDF5NewDaemon import HDF5Daemon

from docs.quant.deribit.OrderBookScrapper.DataBase.MySQLNewDaemon import MySqlDaemon
from docs.quant.deribit.OrderBookScrapper.Utils import MSG_LIST
from docs.quant.deribit.OrderBookScrapper.Utils.AvailableCurrencies import Currency
from docs.quant.deribit.OrderBookScrapper.SyncLib.AvailableRequests import get_ticker_by_instrument_request
from docs.quant.deribit.OrderBookScrapper.Subsciption.AbstractSubscription import AbstractSubscription
from docs.quant.deribit.OrderBookScrapper.Subsciption.OrderBookSubscriptionLimitedDepth import OrderBookSubscriptionCONSTANT

from websocket import WebSocketApp, enableTrace, ABNF
from threading import Thread

from datetime import datetime
import logging
import json
import yaml


async def scrap_available_instruments(currency: Currency, cfg):
    from docs.quant.deribit.OrderBookScrapper.SyncLib.AvailableRequests import get_instruments_by_currency_request
    from docs.quant.deribit.OrderBookScrapper.Utils.AvailableInstrumentType import InstrumentType
    from docs.quant.deribit.OrderBookScrapper.SyncLib.Scrapper import send_request
    import pandas as pd
    import numpy as np
    make_subscriptions_list = await send_request(get_instruments_by_currency_request(currency=currency,
                                                                                     kind=InstrumentType.OPTION,
                                                                                     expired=False))

    # Take only the result of answer. Now we have list of json contains information of option dotes.
    answer = make_subscriptions_list['result']
    available_maturities = pd.DataFrame(np.unique(list(map(lambda x: x["instrument_name"].split('-')[1], answer))))
    available_maturities.columns = ['DeribitNaming']
    available_maturities['RealNaming'] = pd.to_datetime(available_maturities['DeribitNaming'], format='%d%b%y')
    available_maturities = available_maturities.sort_values(by='RealNaming').reset_index(drop=True)
    print("Available maturities: \n", available_maturities)

    # TODO: uncomment
    # selected_maturity = int(input("Select number of interested maturity "))
    selected_maturity = -1
    if selected_maturity == -1:
        warnings.warn("Selected list of instruments is empty")
        return []
    # selected_maturity = 3
    selected_maturity = available_maturities.iloc[selected_maturity]['DeribitNaming']
    print('\nYou select:', selected_maturity)

    selected = list(map(lambda x: x["instrument_name"],
                        list(filter(lambda x: (selected_maturity in x["instrument_name"]) and (
                                x["option_type"] == "call" or "put"), answer))))

    get_underlying = send_request(get_ticker_by_instrument_request(selected[0]),
                                  show_answer=False)['result']['underlying_index']
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
    with open(configuration_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
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
    if cfg["orderBookScrapper"]["scrapper_body"] != "OrderBook":
        raise NotImplementedError
    #
    if type(cfg["record_system"]["use_batches_to_record"]) != bool:
        raise TypeError("Invalid type for record system configuration")
    if type(cfg["record_system"]["number_of_tmp_tables"]) != int:
        raise TypeError("Invalid type for record system configuration")
    if type(cfg["record_system"]["size_of_tmp_batch_table"]) != int:
        raise TypeError("Invalid type for record system configuration")

    return cfg


def subscription_map(scrapper, conf: dict) -> AbstractSubscription:
    match conf["orderBookScrapper"]["scrapper_body"]:
        case "OrderBook":
            return OrderBookSubscriptionCONSTANT(scrapper=scrapper, order_book_depth=conf["orderBookScrapper"]["depth"])
        case _:
            raise NotImplementedError


class DeribitClient(Thread, WebSocketApp):
    websocket: Optional[WebSocketApp]
    database: Optional[Union[MySqlDaemon, HDF5Daemon]] = None
    loop: asyncio.unix_events.SelectorEventLoop

    def __init__(self, cfg, cfg_path: str, loopB, instruments_listed: list = []):

        self.configuration_path = cfg_path
        self.configuration = cfg
        test_mode = self.configuration['orderBookScrapper']["test_net"]
        enable_traceback = self.configuration['orderBookScrapper']["enable_traceback"]
        enable_database_record = bool(self.configuration['orderBookScrapper']["enable_database_record"])

        clean_database = self.configuration['orderBookScrapper']["clean_database"]
        constant_depth_order_book = self.configuration['orderBookScrapper']["depth"]
        instruments_listed = instruments_listed

        Thread.__init__(self)
        self.loop = loopB
        asyncio.set_event_loop(self.loop)

        self.subscription_type = subscription_map(scrapper=self, conf=self.configuration)
        self.instruments_list = instruments_listed
        self.testMode = test_mode
        self.exchange_version = self._set_exchange()
        self.time = datetime.now()

        self.websocket = None
        self.enable_traceback = enable_traceback
        # Set logger settings
        logging.basicConfig(
            level=self.configuration['orderBookScrapper']["logger_level"],
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # Set storages for requested data
        self.instrument_requested = set()
        if enable_database_record:
            match self.configuration['orderBookScrapper']["database_daemon"]:
                case 'mysql':
                    if type(constant_depth_order_book) == int:
                        self.database = MySqlDaemon(configuration_path=self.configuration_path,
                                                    subscription_type=self.subscription_type,
                                                    loop=self.loop)
                    elif constant_depth_order_book is False:
                        self.database = MySqlDaemon(configuration_path=self.configuration_path,
                                                    subscription_type=self.subscription_type,
                                                    loop=self.loop)
                    else:
                        raise ValueError('Unavailable value of depth order book mode')
                    time.sleep(1)

                case "hdf5":
                    if type(constant_depth_order_book) == int:
                        self.database = HDF5Daemon(configuration_path=self.configuration_path,
                                                   subscription_type=self.subscription_type,
                                                   loop=self.loop)
                    elif constant_depth_order_book is False:
                        self.database = HDF5Daemon(configuration_path=self.configuration_path,
                                                   subscription_type=self.subscription_type,
                                                   loop=self.loop)
                    else:
                        raise ValueError('Unavailable value of depth order book mode')
                    time.sleep(1)
                case _:
                    logging.warning("Unknown database daemon selected")
                    self.database = None


    def _set_exchange(self):
        if self.testMode:
            print("Initialized TEST NET mode")
            return 'wss://test.deribit.com/ws/api/v2'
        else:
            print("Initialized REAL MARKET mode")
            return 'wss://www.deribit.com/ws/api/v2'

    def run(self):
        self.websocket = WebSocketApp(self.exchange_version,
                                      on_message=self._on_message, on_open=self._on_open, on_error=self._on_error)
        if self.enable_traceback:
            enableTrace(True)
        # Run forever loop
        while True:
            try:
                self.websocket.run_forever()
            except Exception as e:
                print(e)
                logging.error("Error at run_forever loop")
                # TODO: place here notificator
                continue

    def _on_error(self, websocket, error):
        # TODO: send Telegram notification
        logging.error(error)
        print("ERROR:", error)
        self.instrument_requested.clear()

    def _on_message(self, websocket, message):
        """
        Логика реакции на ответ сервера.
        :param websocket:
        :param message:
        :return:
        """
        response = json.loads(message)
        self._process_callback(response)
        # TODO: Create executor function to make code more readable.
        if 'method' in response:
            # Answer to heartbeat request
            if response['method'] == 'heartbeat':
                # Send test message to approve that connection is still alive
                self.send_new_request(MSG_LIST.test_message())
                return
            # TODO
            asyncio.run_coroutine_threadsafe(self.subscription_type.process_response_from_server(response=response),
                                             loop=self.loop)

    def _process_callback(self, response):
        logging.info(response)
        pass

    def _on_open(self, websocket):
        logging.info("Client start his work")
        self.subscription_type.create_subscription_request()

    def send_new_request(self, request: dict):
        print("Send message")
        self.websocket.send(json.dumps(request), ABNF.OPCODE_TEXT)
        # TODO: do it better. Unsync.
        time.sleep(.1)

    # def make_new_subscribe_all_book(self, instrument_name: str, type_of_data="book", interval="100ms"):
    #     if instrument_name not in self.instrument_requested:
    #         subscription_message = MSG_LIST.make_subscription_all_book(instrument_name, type_of_data=type_of_data,
    #                                                                    interval=interval, )
    #
    #         self.send_new_request(request=subscription_message)
    #         self.instrument_requested.add(instrument_name)
    #     else:
    #         logging.warning(f"Instrument {instrument_name} already subscribed")

    # def make_new_subscribe_constant_depth_book(self, instrument_name: str,
    #                                            type_of_data="book",
    #                                            interval="100ms",
    #                                            depth=None,
    #                                            group=None):
    #     if instrument_name not in self.instrument_requested:
    #         subscription_message = MSG_LIST.make_subscription_constant_book_depth(instrument_name,
    #                                                                               type_of_data=type_of_data,
    #                                                                               interval=interval,
    #                                                                               depth=depth,
    #                                                                               group=group)
    #
    #         self.send_new_request(request=subscription_message)
    #         self.instrument_requested.add(instrument_name)
    #     else:
    #         logging.warning(f"Instrument {instrument_name} already subscribed")


# def request_pipeline(websocket: DeribitClient, cfg):
#     print("Start")
#     # Set heartbeat
#     websocket.send_new_request(MSG_LIST.set_heartbeat(cfg["hearth_beat_time"]))
#     # Send all subscriptions
#     for _instrument_name in websocket.instruments_list:
#         if cfg["depth"] is False:
#             websocket.make_new_subscribe_all_book(instrument_name=_instrument_name)
#         else:
#             websocket.make_new_subscribe_constant_depth_book(instrument_name=_instrument_name,
#                                                              depth=cfg["depth"],
#                                                              group=cfg["group_in_limited_order_book"])
#
#     # Extra like BTC-PERPETUAL
#     for _instrument_name in cfg["add_extra_instruments"]:
#         print("Extra:", _instrument_name)
#         if cfg["depth"] is False:
#             websocket.make_new_subscribe_all_book(instrument_name=_instrument_name)
#         else:
#             websocket.make_new_subscribe_constant_depth_book(instrument_name=_instrument_name,
#                                                              depth=cfg["depth"],
#                                                              group=cfg["group_in_limited_order_book"])

async def f():
    configuration = validate_configuration_file("../configuration.yaml")
    match configuration['orderBookScrapper']["currency"]:
        case "BTC":
            _currency = Currency.BITCOIN
        case "ETH":
            _currency = Currency.ETHER
        case _:
            raise ValueError("Unknown currency")

    derLoop = asyncio.new_event_loop()
    instruments_list = await scrap_available_instruments(currency=_currency, cfg=configuration['orderBookScrapper'])
    deribitWorker = DeribitClient(cfg=configuration, cfg_path="../configuration.yaml",
                                  instruments_listed=instruments_list, loopB=derLoop)
    deribitWorker.start()
    # deribitWorker.send_new_request(MSG_LIST.set_heartbeat(10))
    # derLoop.run_forever()
    # await derLoop.run_in_executor(None, deribitWorker.start)
    # derLoop.create_task(derLoop.run_in_executor(None, deribitWorker.start))

    th = threading.Thread(target=derLoop.run_forever)
    print("Thread started")
    th.start()

    # js = "{'jsonrpc': '2.0', 'method': 'subscription', 'params': {'channel': 'book.BTC-PERPETUAL.none.10.100ms', 'data': {'timestamp': 1670796989478, 'instrument_name': 'BTC-PERPETUAL', 'change_id': 52016142177, 'bids': [[17132.0, 35530.0], [17131.5, 64020.0], [17131.0, 20000.0], [17130.5, 1510.0], [17130.0, 30.0], [17129.0, 6000.0], [17128.5, 5250.0], [17127.5, 480.0], [17127.0, 200.0], [17126.5, 4990.0]], 'asks': [[17132.5, 52250.0], [17133.0, 12950.0], [17133.5, 2780.0], [17134.0, 21710.0], [17134.5, 18580.0], [17135.0, 20000.0], [17135.5, 109300.0], [17136.0, 1060.0], [17136.5, 77790.0], [17137.0, 34440.0]]}}}"
    # js = js.replace("'", "\"")
    # js = json.loads(js)
    # # # deribitWorker.database.add_data(deribitWorker.subscription_type.extract_data_from_response(input_response=js))
    # # js = "{'jsonrpc': '2.0', 'method': 'subscription', 'params': {'channel': 'book.BTC-PERPETUAL.none.10.100ms', 'data': {'timestamp': 1670796989666, 'instrument_name': 'ETH-PERPETUAL', 'change_id': 52016142666, 'bids': [[17666.0, 35530.0], [17131.5, 64020.0], [17131.0, 20000.0], [17130.5, 1510.0], [17130.0, 30.0], [17129.0, 6000.0], [17128.5, 5250.0], [17127.5, 480.0], [17127.0, 200.0], [17126.5, 4990.0]], 'asks': [[17132.5, 52250.0], [17133.0, 12950.0], [17133.5, 2780.0], [17134.0, 21710.0], [17134.5, 18580.0], [17135.0, 20000.0], [17135.5, 109300.0], [17136.0, 1060.0], [17136.5, 77790.0], [17137.0, 34440.0]]}}}"
    # # js = js.replace("'", "\"")
    # # js = json.loads(js)
    # for _ in tqdm(range(10_000)):
    #     js['params']['data']['timestamp'] = _
    #     deribitWorker._on_message(deribitWorker.websocket, message=json.dumps(js))
    #     # deribitWorker._on_message(deribitWorker.websocket, message=json.dumps(js))
    #     # deribitWorker._on_message(deribitWorker.websocket, message=json.dumps(js))
    #     # deribitWorker._on_message(deribitWorker.websocket, message=json.dumps(js))
    #     # deribitWorker.database.add_data(deribitWorker.subscription_type.extract_data_from_response(input_response=js))
    # # deribitWorker.start()
    # # Very important time sleep. I spend smth around 3 hours to understand why my connection
    # # is closed when i try to place new request :(

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(f())
    loop.run_forever()
    time.sleep(1)
