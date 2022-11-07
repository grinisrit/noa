"""
Description:
    Deribit WebSocket Asyncio Example.
    - Authenticated connection.
Usage:
    python3.9 dbt-ws-authenticated-example.py
Requirements:
    - websocket-client >= 1.2.1
"""

# built ins
import asyncio
import sys
import json
import logging
import time
from typing import Dict
from datetime import datetime, timedelta
import threading
import multiprocessing

# installed
import websockets


class main:
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


    def run(self, list_of_subscription):
        # Start Primary Coroutine
        print("TG")
        threading.Thread(target=self.ws_manager())
        print("TG")

    def ws_manager(self) -> None:
        while True:
            print("Manager works")
            self.websocket_client = websockets.connect(self.ws_connection_url, ping_interval=None, compression=None, close_timeout=60)

            while self.websocket_client.open:
                message: bytes = self.websocket_client.recv()
                message: Dict = json.loads(message)
                self.process_answer(message=message)

    def send_subscribe_request(self, operation: str, ws_channel: str) -> None:
        """
        Requests `public/subscribe` or `public/unsubscribe`
        to DBT's API for the specific WebSocket Channel.
        """
        time.sleep(3)
        msg: Dict = {
            "jsonrpc": "2.0",
            "method": f"public/{operation}",
            "id": 42,
            "params": {
                "channels": [ws_channel]
            }
        }

        self.websocket_client.send(json.dumps(msg))

    def process_answer(self, message: json):
        logging.info(message)
        # Process initial
        if 'result' in message:
            self.subscribed_instruments.add(message['result'][0])

        # Process initial snapshot
        if 'data' in message:
            if message['data']['type'] == "snapshot":
                pass
                


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
    ws_connection_url: str = 'wss://test.deribit.com/ws/api/v2'

    # DBT Client ID
    client_id: str = '<client-id>'
    # DBT Client Secret
    client_secret: str = '<client_secret>'
    connection = main(ws_connection_url=ws_connection_url, client_id=client_id, client_secret=client_secret)
    connection.run([])
    subscription_BTC = connection.send_subscribe_request(operation='subscribe', ws_channel='book.BTC-PERPETUAL.100ms')
    subscription_ETH = connection.send_subscribe_request(operation='subscribe', ws_channel='book.ETH-PERPETUAL.100ms')

    connection.run([subscription_BTC, subscription_ETH])

