import pandas as pd

from HestonModel.OptionMarketScrapper.AvailableRequests import get_instruments_by_currency_request
from AvailableCurrencies import Currency
from AvailableInstrumentType import InstrumentType
from HestonModel.OptionMarketScrapper.Scrapper import send_request, URL_TO_SCRAP

import asyncio
import json
import logging
from typing import Dict
from mysql.connector import connect, Error

import numpy as np
import websockets


class MAIN:
    def __init__(self, ws_connection_url: str, client_id: str, client_secret: str) -> None:
        # Async Event Loop
        self.loop = None

        # Instance Variables
        self.ws_connection_url: str = ws_connection_url
        self.client_id: str = client_id
        self.client_secret: str = client_secret
        self.websocket_client: websockets.WebSocketClientProtocol = None
        # Block with storages
        self.subscribed_instruments = set()

        self.pairs_instruments_id = dict()

        try:
            self.connection = connect(
                    host="localhost",
                    user="root",
                    database="DeribitOrderBook"
            )
            print(self.connection)
        except Error as e:
            print(e)

    def run(self, list_of_subscription):
        self.loop = asyncio.get_event_loop()
        # Start Primary Coroutine
        self.loop.run_until_complete(self.ws_manager(list_of_subscription=list_of_subscription))
        self.loop.close()

    async def ws_manager(self, list_of_subscription) -> None:
        async with websockets.connect(self.ws_connection_url, ping_interval=None, compression=None, close_timeout=60) \
                as self.websocket_client:
            # Subscribe to the specified WebSocket Channel
            [self.loop.create_task(subscribe_request) for subscribe_request in list_of_subscription]

            while self.websocket_client.open:
                message: bytes = await self.websocket_client.recv()
                message: Dict = json.loads(message)
                self.process_answer(message=message)

    async def send_subscribe_request(self, operation: str, ws_channel: str) -> None:
        """
        Requests `public/subscribe` or `public/unsubscribe`
        to DBT's API for the specific WebSocket Channel.
        """
        await asyncio.sleep(1)

        msg: Dict = {
            "jsonrpc": "2.0",
            "method": f"public/{operation}",
            "id": 42,
            "params": {
                "channels": [ws_channel]
            }
        }
        print(msg)
        await self.websocket_client.send(json.dumps(msg))

    def process_answer(self, message: json):
        logging.info(message)

        # Process initial
        if 'result' in message:
            if len(message['result']) != 0:
                self.subscribed_instruments.add(message['result'][0])


        # Process initial snapshot. Place line at:
        # Initial snapshot pair | Initial snapshots data
        if 'params' in message:
            message = message['params']
        if 'data' in message:
            message = message['data']
            insert_query = ("INSERT INTO DUMB_TABLE "
                            "(INSTRUMENT_NAME, TIMESTAMP, OPERATION, WAY, PRICE, AMOUNT, CHANGE_ID, PREV_CHANGE_ID)"
                            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)")

            prev_change = 0
            if message['type'] == 'snapshot':
                prev_change = 0
            if 'prev_change_id' in message:
                prev_change = message['prev_change_id']

            current_id = message['change_id']
            timestamp = message['timestamp']
            instrument = message['instrument_name']

            # PLACE INTO CHANGE_ID <-> TIMESTAMP
            # place_db(change_id, timestamp)

            # FILL INITIAL

            bids = message['bids']
            for bid in bids:
                operation = bid[0]
                price = bid[1]
                amount = bid[2]
                insert = (instrument, timestamp, operation, 'BID', price, amount, current_id, prev_change)
                print(insert)
                with self.connection.cursor() as cursor:
                    cursor.execute(insert_query, insert)
                    self.connection.commit()

                # place_db(new_pair_id, "BID", amount, price, current_id)
            asks = message['asks']
            for ask in asks:
                operation = ask[0]
                price = ask[1]
                amount = ask[2]
                insert = (instrument, timestamp, operation, 'ASK', price, amount, current_id, prev_change)
                print(insert)
                with self.connection.cursor() as cursor:
                    cursor.execute(insert_query, insert)
                    self.connection.commit()

    def process_answer_to_do(self, message: json):
        logging.info(message)
        # Process initial
        if 'result' in message:
            if len(message['result']) != 0:
                self.subscribed_instruments.add(message['result'][0])

        # Process initial snapshot. Place line at:
        # Initial snapshot pair | Initial snapshots data
        if 'data' in message:
            if message['data']['type'] == "snapshot":
                current_id = message['data']['change_id']
                timestamp = message['timestamp']
                instrument = message['instrument_name']

                # Select last filled ID PAIR
                select_last_id = 0
                new_pair_id = 1

                self.pairs_instruments_id[instrument] = new_pair_id

                # PLACE INTO CHANGE_ID <-> TIMESTAMP
                # place_db(change_id, timestamp)

                # FILL INITIAL

                bids = message['data']['bids']
                for bid in bids:
                    operation = bid[0]
                    price = bid[1]
                    amount = bid[2]

                    # place_db(new_pair_id, "BID", amount, price, current_id)
                asks = message['data']['asks']
                for ask in asks:
                    operation = ask[0]
                    price = ask[1]
                    amount = ask[2]

                    # place_db(new_pair_id, "ASK", amount, price, current_id)

        # Process updates. Place line at:
        # Change_db
        if 'data' in message:
            if message['data']['type'] == "change":
                prev_id = message['data']['prev_change_id']
                current_id = message['data']['change_id']
                timestamp = message['timestamp']
                instrument = message['instrument_name']

                # Select last filled ID PAIR
                select_last_id = 0
                new_pair_id = 1

                ID_initial = self.pairs_instruments_id[instrument]

                # PLACE INTO CHANGE_ID <-> TIMESTAMP
                # place_db(change_id, timestamp)

                # FILL
                bids = message['data']['bids']
                for bid in bids:
                    operation = bid[0]
                    price = bid[1]
                    amount = bid[2]
                    # place_db(new_pair_id, "BID", amount, price, current_id)

                asks = message['data']['asks']
                for ask in asks:
                    operation = ask[0]
                    price = ask[1]
                    amount = ask[2]
                    # place_db(new_pair_id, "ASK", amount, price, current_id)



if __name__ == "__main__":
    # Logging
    logging.basicConfig(
        level='INFO',
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # DBT LIVE WebSocket Connection URL
    # ws_connection_url: str = 'wss://www.deribit.com/ws/api/v2'
    # DBT TEST WebSocket Connection URL
    ws_connection_url: str = URL_TO_SCRAP

    # DBT Client ID
    client_id: str = '<client-id>'
    # DBT Client Secret
    client_secret: str = '<client_secret>'
    connection = MAIN(ws_connection_url=ws_connection_url, client_id=client_id, client_secret=client_secret)

    currency = Currency.BITCOIN
    make_subscriptions_list = send_request(get_instruments_by_currency_request(currency=currency,
                                                                               kind=InstrumentType.OPTION,
                                                                               expired=False))

    answer_id = make_subscriptions_list['id']
    # Take only the result of answer. Now we have list of json contains information of option dotes.
    answer = make_subscriptions_list['result']
    available_maturities = pd.DataFrame(np.unique(list(map(lambda x: x["instrument_name"].split('-')[1], answer))))
    available_maturities.columns = ['DeribitNaming']
    available_maturities['RealNaming'] = pd.to_datetime(available_maturities['DeribitNaming'], format='%d%b%y')
    available_maturities = available_maturities.sort_values(by='RealNaming').reset_index(drop=True)
    print("Available maturities: \n", available_maturities)

    # TODO: uncomment
    # selected_maturity = int(input("Select number of interested maturity "))
    selected_maturity = 2
    selected_maturity = available_maturities.iloc[selected_maturity]['DeribitNaming']
    print('\nYou select:', selected_maturity)

    selected = list(map(lambda x: x["instrument_name"]
        ,list(filter(lambda x: (selected_maturity in x["instrument_name"]) and (x["option_type"] == "call"), answer))))

    make_subscriptions_list = [
        connection.send_subscribe_request(operation='subscribe',
                                          ws_channel=f"book.{instrument}.100ms")
        for instrument in selected
    ]

    # subscription_BTC = connection.send_subscribe_request(operation='subscribe', ws_channel='book.BTC-PERPETUAL.100ms')
    # subscription_ETH = connection.send_subscribe_request(operation='subscribe', ws_channel='book.ETH-PERPETUAL.100ms')

    connection.run(make_subscriptions_list)

