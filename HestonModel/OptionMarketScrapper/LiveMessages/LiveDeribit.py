# built ins
import asyncio
import sys
import json
import logging
from typing import Dict, Union
from datetime import datetime, timedelta
from HestonModel.OptionMarketScrapper.AvailableRequests import *
from enum import Enum
from datetime import datetime

from threading import Thread
# installed
import websockets


class AvailableOperations(Enum):
    __slots__ = {"_operation"}
    name: str

    def __init__(self, name: str):
        self._name = name

    def __str__(self):
        return self.name

    @property
    def name(self):
        return self._name

    POST = 'POST'
    GET = 'GET'


class DeribitLive:
    def __init__(self, testnet: bool = True, authentication: bool = False, client_id: str = None, client_secret: str = None) -> None:
        if testnet:
            ws_connection_url = "wss://test.deribit.com/ws/api/v2/"
        else:
            ws_connection_url = "wss://www.deribit.com/ws/api/v2/"

        if authentication:
            if not client_id or not client_secret:
                raise ValueError('Need to put client data while auth mode')

        logging.basicConfig(
            format='%(asctime)s %(funcName)s %(message)s', level=logging.INFO)
        self.logger = logging.getLogger(f"deribit-{'public' if not authentication else 'private'}")

        self._json_rpc_version = "2.0"
        self._current_request_id = 0
        self.post_request_list: Dict[int, json] = dict()

        self.authentication_mode = authentication
        # Async Event Loop
        self.loop = asyncio.get_event_loop()

        # Instance Variables
        self.ws_connection_url: str = ws_connection_url

        self.client_id: str = client_id
        self.client_secret: str = client_secret
        self.refresh_token: str = None
        self.refresh_token_expiry_time: int = None

        self.websocket_client: websockets.WebSocketClientProtocol = None
        # Start Primary Coroutine
        self.loop.run_until_complete(
            self.websocket_manager()
            )

        self.loop.create_task(self.send_public_message(get_currencies_request()))

    @property
    def current_request_id(self):
        self._current_request_id += 1
        return self._current_request_id

    async def websocket_manager(self) -> None:
        async with websockets.connect(
                self.ws_connection_url, ping_interval=None, compression=None, close_timeout=60) as self.websocket_client:
            if self.authentication_mode:
                # Authenticate WebSocket Connection
                await self.ws_auth()

                # Establish Heartbeat
                await self.establish_heartbeat()

                # Start Authentication Refresh Task
                self.loop.create_task(
                    self.ws_refresh_auth()
                )

                # Subscribe to the specified WebSocket Channel
                # self.loop.create_task(self.send_public_message())

            if not self.authentication_mode:
                # Subscribe to the specified WebSocket Channel
                print('subcscribe')
                self.loop.create_task(self.send_public_message(get_currencies_request()))
            while self.websocket_client.open:
                message: bytes = await self.websocket_client.recv()
                print('New Message')
                message: Dict = json.loads(message)
                logging.info(message)

                if 'id' in list(message):
                    if message['id'] == 9929:
                        if self.refresh_token is None:
                            logging.info('Successfully authenticated WebSocket Connection')
                        else:
                            logging.info('Successfully refreshed the authentication of the WebSocket Connection')

                        self.refresh_token = message['result']['refresh_token']

                        # Refresh Authentication well before the required datetime
                        if message['testnet']:
                            expires_in: int = 300
                        else:
                            expires_in: int = message['result']['expires_in'] - 240

                        self.refresh_token_expiry_time = datetime.utcnow() + timedelta(seconds=expires_in)

                    elif message['id'] == 8212:
                        # Avoid logging Heartbeat messages
                        continue

                elif 'method' in list(message):
                    # Respond to Heartbeat Message
                    if message['method'] == 'heartbeat':
                        await self.heartbeat_response()

            else:
                logging.info('WebSocket connection has broken.')
                sys.exit(1)

    async def establish_heartbeat(self) -> None:
        """
        Requests DBT's `public/set_heartbeat` to
        establish a heartbeat connection.
        """
        msg: Dict = {
                    "jsonrpc": "2.0",
                    "id": 9098,
                    "method": "public/set_heartbeat",
                    "params": {
                              "interval": 10
                               }
                    }

        await self.websocket_client.send(json.dumps(msg))

    async def heartbeat_response(self) -> None:
        """
        Sends the required WebSocket response to
        the Deribit API Heartbeat message.
        """
        msg: Dict = {
                    "jsonrpc": "2.0",
                    "id": 8212,
                    "method": "public/test",
                    "params": {}
                    }

        await self.websocket_client.send(
            json.dumps(
                msg
                )
                )

    async def send_public_message(self, message: json) -> None:
        """
        Requests `public/subscribe` or `public/unsubscribe`
        to DBT's API for the specific WebSocket Channel.
        """
        await asyncio.sleep(5)

        if ('private' in message["method"]) and (not self.authentication_mode):
            raise ValueError("Can't send private methods without authentication")
        message["id"] = self.current_request_id
        self.logger.info(str(AvailableOperations.POST) + " | " + str(message))

        await self.websocket_client.send(json.dumps(message))


if __name__ == "__main__":
    deribit = DeribitLive(testnet=True, authentication=False)
    print('goes brrrr')
