import asyncio
import json
import logging
from typing import Callable, Any, Coroutine, Awaitable, Optional

import aiohttp
from time import time

logger = logging.getLogger(__name__)


class SessionWrapper:
    """
    Handlers:
     - on_connect_ws - Called after the connection is established.
            If auth_type is not equivalent to AuthType.NONE,
            on_connect_ws will be set to self.auth_login;
     - on_close_ws - Called after disconnection, default value is None;
     - on_authenticated - Called after authentication is confirmed, default value is None;
     - on_message - Called when a message is received, default value is self.handle_message;
     - on_before_handling -
    """

    def __init__(self, timeout: aiohttp.ClientTimeout = None, session: aiohttp.ClientSession = None):
        self.__internal_session: bool = True
        self.__is_session: bool = False
        self.__session: Optional[aiohttp.ClientSession] = None
        self._timeout: Optional[aiohttp.ClientTimeout] = None

        self._on_connect: Optional[Callable[[], Any]] = None
        self._async_on_connect: Optional[Callable[[], Awaitable[Any]]] = None

        self._on_message: Optional[Callable[[str], bool]] = None
        self._async_on_message: Optional[Callable[[str], Awaitable[bool]]] = None

        self.on_close_ws: Optional[Callable[[], Any]] = None
        self.on_before_handling: Optional[Callable[[str], Any]] = None

        self.ws_api: str = ""
        self.rest_api: str = ""
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.last_message: str = ""
        self.receipt_time: float = 0.

        if session:
            self._session = session
            self.__internal_session = False

        self._timeout = timeout if timeout else aiohttp.ClientTimeout(total=20)

        self.logger = logging.getLogger(__name__)


    @property
    def _session(self):
        if not self.__is_session:
            self.__check_session()
        return self.__session

    @_session.setter
    def _session(self, value):
        self.__is_session = True
        self.__internal_session = False
        self.__session = value

    def __check_session(self):
        if not self.__is_session:
            self.__is_session = True
            self.__internal_session = True
            self.__session = aiohttp.ClientSession(timeout=self._timeout)

    def __del__(self):
        self._close()

    def __enter__(self):
        self.__check_session()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._close()

    def _close(self):
        if self.__internal_session and self.__session:
            asyncio.ensure_future(self.__session.close())
            self.__is_session = False

    @property
    def on_message(self):
        return self._on_message

    @on_message.setter
    def on_message(self, handler):
        if handler is None:
            self._async_on_message = None
            self._on_message = None
        else:
            if asyncio.iscoroutinefunction(handler):
                self._async_on_message = handler
            else:
                self._on_message = handler

    @property
    def on_connect_ws(self):
        if self._async_on_connect:
            return self._async_on_connect
        else:
            return self._on_connect

    @on_connect_ws.setter
    def on_connect_ws(self, handler):
        if handler is None:
            self._async_on_connect = None
            self._on_connect = None
        else:
            if asyncio.iscoroutinefunction(handler):
                self._async_on_connect = handler
                self._on_connect = None
            else:
                self._async_on_connect = None
                self._on_connect = handler

    async def run_receiver(self):
        """
        Establish a connection and start the receiver loop.
        :return:
        """
        print('start connection')
        self.ws = await self._session.ws_connect(self.ws_api)
        if self._async_on_connect:
            await self._async_on_connect()
        elif self._on_connect:
            self._on_connect()

        # A receiver loop
        while self.ws and not self.ws.closed:
            print('wait for message')
            message = await self.ws.receive()
            print('no await for message')
            self.receipt_time = time()

            if message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                self.logger.warning(f"Connection close {repr(message)}")
                if self.on_close_ws:
                    await self.on_close_ws()

                continue
            if message.type == aiohttp.WSMsgType.CLOSING:
                self.logger.debug(f"Connection closing {repr(message)}")
                continue

            if message.type == aiohttp.WSMsgType.TEXT:
                print('get message')
                self.last_message = message.data
                print(self.last_message)
            #     if self.on_before_handling:
            #         self.on_before_handling(message.data)
            #
            #     processed = False
            #     if self._async_on_message:
            #         processed = await self._async_on_message(message.data)
            #     elif self._on_message:
            #         processed = self._on_message(message.data)
            #
            #     if not processed:
            #         self.handle_message(message.data)
            # else:
            #     self.logger.warning(f"Unknown type of message {repr(message)}")


    def handle_message(self, message: str):
        pass

    async def public_get(self, request_path, params=None):
        async with self._session.get(url=self.rest_api + request_path,
                                     params=params,
                                     headers=None) as response:
            if response.status == 200:
                data = await response.read()
                if response.content_type.endswith('json'):
                    data = json.loads(data)
                else:
                    logger.warning(f"Unexpected response.content_type:{response.content_type}")
                return data
            else:
                response.raise_for_status()

    async def stop(self):
        await self.ws.close()

        if self.__internal_session and self.__session:
            await self.__session.close()
            self.__session = None
            self.__is_session = False

    def is_connected(self) -> bool:
        if self.ws is None:
            return False

        return not self.ws.closed
