import pprint
import time
import warnings

from Orderbook.DataBase.MySQLDaemon import MySqlDaemon
from Orderbook.Utils import MSG_LIST
from Orderbook.Utils.AvailableCurrencies import Currency
from Orderbook.SyncLib.AvailableRequests import get_ticker_by_instrument_request

from websocket import WebSocketApp, enableTrace, ABNF
from threading import Thread

from datetime import datetime
import logging
import json
import yaml

with open("../configuration.yaml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)["orderBookScrapper"]
# TODO: make available to select TEST_NET inside Scrapper


# TODO: Add here + index_future for time_maturity
def scrap_available_instruments(currency: Currency):
    from Orderbook.SyncLib.AvailableRequests import get_instruments_by_currency_request
    from Orderbook.Utils.AvailableInstrumentType import InstrumentType
    from Orderbook.SyncLib.Scrapper import send_request
    import pandas as pd
    import numpy as np

    make_subscriptions_list = send_request(get_instruments_by_currency_request(currency=currency,
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
    selected_maturity = 3
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


class DeribitClient(Thread, WebSocketApp):
    websocket: WebSocketApp | None
    database: MySqlDaemon | None

    def __init__(self, test_mode: bool = False, enable_traceback: bool = True, enable_database_record: bool = True,
                 clean_database=False, constant_depth_order_book: bool | int = False):
        Thread.__init__(self)
        self.testMode = test_mode
        self.exchange_version = self._set_exchange()
        self.time = datetime.now()

        self.websocket = None
        self.enable_traceback = enable_traceback
        # Set logger settings
        logging.basicConfig(
            level=cfg["logger_level"],
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # Set storages for requested data
        self.instrument_requested = set()
        if enable_database_record:
            if type(constant_depth_order_book) == int:
                self.database = MySqlDaemon(constant_depth_mode=constant_depth_order_book, clean_tables=clean_database)
            elif constant_depth_order_book is False:
                self.database = MySqlDaemon(constant_depth_mode=constant_depth_order_book, clean_tables=clean_database)
            else:
                raise ValueError('Unavailable value of depth order book mode')
            time.sleep(1)
        else:
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
            except:
                logging.error("Error at run_forever loop")
                # TODO: place here notificator
                continue

    def _on_error(self, websocket, error):
        # TODO: send Telegram notification
        logging.error(error)
        print(error)
        pass

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

            # SUBSCRIPTION processing
            if response['method'] == "subscription":

                # ORDER BOOK processing. For constant book depth
                if 'change' and 'type' not in response['params']['data']:

                    if self.database:
                        self.database.add_order_book_content_limited_depth(
                            change_id=response['params']['data']['change_id'],
                            timestamp=response['params']['data']['timestamp'],
                            bids=response['params']['data']['bids'],
                            asks=response['params']['data']['asks'],
                            instrument_name=response['params']['data']['instrument_name']
                        )
                    # raise ValueError("STOP ALL")
                    return

                # INITIAL SNAPSHOT processing. For unlimited book depth
                if response['params']['data']['type'] == 'snapshot':
                    if self.database:
                        self.database.add_instrument_init_snapshot(
                            instrument_name=response['params']['data']['instrument_name'],
                            start_instrument_scrap_time=response['params']['data']['timestamp'],
                            request_change_id=response['params']['data']['change_id'],
                            bids_list=response['params']['data']['bids'],
                            asks_list=response['params']['data']['asks'],
                        )
                        return
                # CHANGE ORDER BOOK processing. For unlimited book depth
                if response['params']['data']['type'] == 'change':
                    if self.database:
                        self.database.add_instrument_change_order_book_unlimited_depth(
                            request_change_id=response['params']['data']['change_id'],
                            request_previous_change_id=response['params']['data']['prev_change_id'],
                            change_timestamp=response['params']['data']['timestamp'],
                            bids_list=response['params']['data']['bids'],
                            asks_list=response['params']['data']['asks'],
                        )
                        return

    def _process_callback(self, response):
        logging.info(response)
        pass

    def _on_open(self, websocket):
        logging.info("Client start his work")
        self.websocket.send(json.dumps(MSG_LIST.hello_message()))

    def send_new_request(self, request: dict):
        self.websocket.send(json.dumps(request), ABNF.OPCODE_TEXT)

    def make_new_subscribe_all_book(self, instrument_name: str, type_of_data="book", interval="100ms"):
        if instrument_name not in self.instrument_requested:
            subscription_message = MSG_LIST.make_subscription_all_book(instrument_name, type_of_data=type_of_data,
                                                                       interval=interval,)

            self.send_new_request(request=subscription_message)
            self.instrument_requested.add(instrument_name)
        else:
            logging.warning(f"Instrument {instrument_name} already subscribed")

    def make_new_subscribe_constant_depth_book(self, instrument_name: str,
                                               type_of_data="book",
                                               interval="100ms",
                                               depth=None,
                                               group=None):
        if instrument_name not in self.instrument_requested:
            subscription_message = MSG_LIST.make_subscription_constant_book_depth(instrument_name,
                                                                                  type_of_data=type_of_data,
                                                                                  interval=interval,
                                                                                  depth=depth,
                                                                                  group=group)

            self.send_new_request(request=subscription_message)
            self.instrument_requested.add(instrument_name)
        else:
            logging.warning(f"Instrument {instrument_name} already subscribed")


if __name__ == '__main__':
    if cfg["currency"] == "BTC":
        _currency = Currency.BITCOIN
    elif cfg["currency"] == "ETH":
        _currency = Currency.ETHER
    else:
        raise ValueError("Unknown currency")

    instruments_list = scrap_available_instruments(currency=_currency)

    deribitWorker = DeribitClient(test_mode=cfg["test_net"],
                                  enable_traceback=cfg["enable_traceback"],
                                  enable_database_record=cfg["enable_database_record"],
                                  clean_database=cfg["clean_database"],
                                  constant_depth_order_book=cfg["depth"])
    deribitWorker.start()
    # Very important time sleep. I spend smth around 3 hours to understand why my connection
    # is closed when i try to place new request :(
    time.sleep(1)
    # Send Hello Message
    deribitWorker.send_new_request(MSG_LIST.hello_message())
    # Set heartbeat
    deribitWorker.send_new_request(MSG_LIST.set_heartbeat(cfg["hearth_beat_time"]))
    # Send all subscriptions
    for _instrument_name in instruments_list:
        if cfg["depth"] is False:
            deribitWorker.make_new_subscribe_all_book(instrument_name=_instrument_name)
        else:
            deribitWorker.make_new_subscribe_constant_depth_book(instrument_name=_instrument_name,
                                                                 depth=cfg["depth"],
                                                                 group=cfg["group_in_limited_order_book"])