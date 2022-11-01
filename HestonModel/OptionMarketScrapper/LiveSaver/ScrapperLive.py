import time
import json
import asyncio
import hashlib
import hmac
import os
from websocket import WebSocketApp, enableTrace
from datetime import datetime, timedelta
import pandas as pd
import secrets
import sys
from argparse import ArgumentParser
from threading import Thread
import logging
from SessionWrapper import SessionWrapper


class DeribitClient(SessionWrapper):
    def __init__(self,
                 client_id: str = None,
                 client_secret: str = None,
                 scope: str = "session",
                 testnet: bool = True):

        super().__init__()

        self.on_connect_ws = None

        self.on_authenticated = None
        self.on_token = None
        self.on_response_error = None
        self.on_handle_response = None

        self.requests = {}
        self.ws_api = f"wss://{'test' if testnet else 'www'}.deribit.com/ws/api/v2/"
        # self.get_id = get_id
        self.logger = logging.getLogger(__name__)
        # self.auth_type = auth_type
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope

        self._get_id = 1

    def close(self):
        super()._close()

    @property
    def get_id(self):
        ref = self._get_id
        self._get_id += 1
        return ref

    @get_id.setter
    def get_id(self, value):
        self._get_id = value

    async def send_public(self, request: dict, callback=None, logging_it: bool = True) -> int:
        """
        Send a public request
        :param request: Request without jsonrpc and id fields
        :param callback: The function that will be called after receiving the query result. Default is None
        :param logging_it:
        :return: Request Id
        """
        request_id = self.get_id
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            **request
        }
        if logging_it:
            self.logger.info(f"sending:{request}")
        await self.ws.send_json(request)

        if callback:
            request["callback"] = callback

        self.requests[request_id] = request

        return request_id

    async def send(self, request: dict, callback=None) -> int:
        """
        A wrapper for send_private and send_public, defines the type of request by content.
        :param request: Request without jsonrpc and id fields
        :param callback: The function that will be called after receiving the query result. Default is None
        :return: Request Id
        """
        method = request["method"]
        if method.starts("public/"):
            request_id = await self.send_public(request, callback)
        else:
            raise ValueError('No private methods implemented')
            # request_id = await self.send_private(request, callback)

        return request_id

    async def set_heartbeat(self, interval: int = 15) -> int:
        """
        :param interval:
        :return:
        """
        request_id = await self.send_public({"method": "public/set_heartbeat", "params": {"interval": interval}})
        return request_id

    async def disable_heartbeat(self) -> int:
        """
        :return:
        """
        request_id = await self.send_public({"method": "public/disable_heartbeat", "params": {}})
        return request_id


    async def get_instruments(self, currency: str, kind: str = None, expired: bool = False, callback=None) -> int:
        """
        Send a request for a list available trading instruments
        :param currency: The currency symbol: BTC or ETH
        :param kind: Instrument kind: future or option, if not provided instruments of all kinds are considered
        :param expired:
        :param callback:
        :return: Request Id
        """
        request = {"method": "public/get_instruments",
                   "params": {
                       "currency": currency,
                       "expired": expired
                   }}
        if kind:
            request["params"]["kind"] = kind

        return await self.send_public(request=request, callback=callback)

    def handle_message(self, message: str) -> None:
        """
        :param message:
        :return:
        """
        data = json.loads(message)
        if "method" in data:
            self.handle_method_message(data)
        else:
            if "id" in data:
                if "error" in data:
                    self.handle_error(data)
                else:
                    request_id = data["id"]
                    if request_id:
                        self.handle_response(request_id, data)
            else:
                self.handle_method_message(data)

    def empty_handler(self, **kwargs) -> None:
        """
        A default handler
        :param kwargs:
        :return:
        """
        self.logger.debug(f"{repr(kwargs)}")

    def handle_method_message(self, data) -> None:
        """
        :param data:
        :return:
        """
        method = data["method"]
        handler = resolve_route(method, self.method_routes)

        if handler:
            if asyncio.iscoroutinefunction(handler):
                asyncio.ensure_future(handler(data))
            else:
                handler(data)

        elif not self.on_before_handling:
            self.logger.warning(f"Unhandled message:{repr(data)}.")

    def handle_response(self, request_id: int, response: dict) -> None:
        """
        :param request_id:
        :param response:
        :return:
        """

        request = self.requests.get(request_id)
        if request:
            if "callback" in request:
                callback = request["callback"]
                if asyncio.iscoroutinefunction(callback):
                    asyncio.ensure_future(callback(response))
                else:
                    callback(response)
            else:
                method = request["method"]

                print(response)

            del self.requests[request_id]