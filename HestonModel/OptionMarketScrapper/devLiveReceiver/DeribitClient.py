import time
from MySQLDaemon import MySqlDaemon
from AvailableCurrencies import Currency

from websocket import WebSocketApp, enableTrace, ABNF
from threading import Thread

from datetime import datetime
import logging
import json
# TODO: make available to select TEST_NET inside Scrapper
from HestonModel.OptionMarketScrapper.Scrapper import TEST_NET
import MSG_LIST


# TODO: Add here + index_future for time_maturity
def scrap_available_instruments(currency: Currency):
    from HestonModel.OptionMarketScrapper.AvailableRequests import get_instruments_by_currency_request
    from AvailableInstrumentType import InstrumentType
    from HestonModel.OptionMarketScrapper.Scrapper import send_request
    import pandas as pd
    import numpy as np

    make_subscriptions_list = send_request(get_instruments_by_currency_request(currency=currency,
                                                                               kind=InstrumentType.OPTION,
                                                                               expired=False))

    # answer_id = make_subscriptions_list['id']
    # Take only the result of answer. Now we have list of json contains information of option dotes.
    answer = make_subscriptions_list['result']
    available_maturities = pd.DataFrame(np.unique(list(map(lambda x: x["instrument_name"].split('-')[1], answer))))
    available_maturities.columns = ['DeribitNaming']
    available_maturities['RealNaming'] = pd.to_datetime(available_maturities['DeribitNaming'], format='%d%b%y')
    available_maturities = available_maturities.sort_values(by='RealNaming').reset_index(drop=True)
    print("Available maturities: \n", available_maturities)

    # TODO: uncomment
    selected_maturity = int(input("Select number of interested maturity "))
    # selected_maturity = 2
    selected_maturity = available_maturities.iloc[selected_maturity]['DeribitNaming']
    print('\nYou select:', selected_maturity)

    selected = list(map(lambda x: x["instrument_name"],
                        list(filter(lambda x: (selected_maturity in x["instrument_name"]) and (
                                x["option_type"] == "call"), answer))))

    print("Selected Instruments")
    print(selected)
    return selected


class DeribitClient(Thread, WebSocketApp):
    websocket: WebSocketApp | None
    database: MySqlDaemon | None

    def __init__(self, test_mode: bool = False, enable_traceback: bool = True, enable_database_record: bool = True,
                 clean_database=False):
        Thread.__init__(self)
        self.testMode = test_mode
        self.exchange_version = self._set_exchange()
        self.time = datetime.now()

        self.websocket = None
        self.enable_traceback = enable_traceback
        # Set logger settings
        logging.basicConfig(
            level='INFO',
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        # Set storages for requested data
        self.instrument_requested = set()
        if enable_database_record:
            self.database = MySqlDaemon(clean_tables=clean_database)
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
                # INITIAL SNAPSHOT processing
                if response['params']['data']['type'] == 'snapshot':
                    if self.database:
                        self.database.add_instrument_init_snapshot(
                            instrument_name=response['params']['data']['instrument_name'],
                            start_instrument_scrap_time=response['params']['data']['timestamp'],
                            request_change_id=response['params']['data']['change_id'],
                            bids_list=response['params']['data']['bids'],
                            asks_list=response['params']['data']['asks'],
                        )
                # CHANGE ORDER BOOK processing
                if response['params']['data']['type'] == 'change':
                    if self.database:
                        self.database.add_instrument_change_order_book(
                            request_change_id=response['params']['data']['change_id'],
                            request_previous_change_id=response['params']['data']['prev_change_id'],
                            change_timestamp=response['params']['data']['timestamp'],
                            bids_list=response['params']['data']['bids'],
                            asks_list=response['params']['data']['asks'],
                        )

    def _process_callback(self, response):
        logging.info(response)
        pass

    def _on_open(self, websocket):
        logging.info("Client start his work")
        self.websocket.send(json.dumps(MSG_LIST.hello_message()))

    def send_new_request(self, request: dict):
        self.websocket.send(json.dumps(request), ABNF.OPCODE_TEXT)

    def make_new_subscribe(self, instrument_name: str, type_of_data="book", interval="100ms", depth=None, group=None):
        if instrument_name not in self.instrument_requested:
            subscription_message = MSG_LIST.make_subscription(instrument_name, type_of_data=type_of_data,
                                                              interval=interval, depth=depth, group=group)

            self.send_new_request(request=subscription_message)
            self.instrument_requested.add(instrument_name)
        else:
            logging.warning(f"Instrument {instrument_name} already subscribed")


if __name__ == '__main__':
    instruments_list = scrap_available_instruments(currency=Currency.BITCOIN)

    deribitWorker = DeribitClient(test_mode=TEST_NET,
                                  enable_traceback=False,
                                  enable_database_record=True,
                                  clean_database=True)
    deribitWorker.start()
    # Very important time sleep. I spend smth around 3 hours to understand why my connection
    # is closed when i try to place new request :(
    time.sleep(1)
    # Send Hello Message
    deribitWorker.send_new_request(MSG_LIST.hello_message())
    # Set heartbeat
    deribitWorker.send_new_request(MSG_LIST.set_heartbeat(10))
    # Send one subscription
    # deribitWorker.make_new_subscribe("ETH-PERPETUAL")
    # Send all subscriptions
    for instrument_name in instruments_list:
        deribitWorker.make_new_subscribe(instrument_name=instrument_name)
