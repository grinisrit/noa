# New Deribit API. With it we can't get historical prices for a long horizon of time.
import pandas as pd
import websockets
import asyncio
import json
import AvailableRequests
from AvailableCurrencies import Currency
import nest_asyncio
from pprint import pprint

nest_asyncio.apply()


class DeribitConnection:
    __slots__ = {"api_key", "connectionStatus"}

    def __init__(self, api_key, log_available_currencies=False):
        self.api_key = api_key
        if log_available_currencies:
            print("Available Currencies")
            pprint(asyncio.get_event_loop().run_until_complete(
                DeribitConnection.call_api(json.dumps(AvailableRequests.currencyInfoMsg))))

    @classmethod
    async def call_api(cls, msg: AvailableRequests):
        async with websockets.connect('wss://test.deribit.com/ws/api/v2') as websocket:
            await websocket.send(msg)
            while websocket.open:
                response = await websocket.recv()
                json_par = json.loads(response)
                return json_par

    def get_currency_price(self, currency: str):
        pprint(asyncio.get_event_loop().run_until_complete(
            DeribitConnection.call_api(json.dumps(AvailableRequests.price_data_msg(currency)))))

    def get_last_trades(self, currency: Currency, number_of_last_trades: int):
        pprint(asyncio.get_event_loop().run_until_complete(
            DeribitConnection.call_api(json.dumps(AvailableRequests.get_last_trades(currency, number_of_last_trades)))))


if __name__ == "__main__":
    deribit = DeribitConnection(api_key="l4WX3QwSLx6araTB4_phZwpkVlPvs7rcY0ziIvfh7LM")
    # test_request()
    # deribit.get_currency_price(Currency.BITCOIN.currency + "-PERPETUAL")
    deribit.extended_get_last_trades(Currency.BITCOIN, 2)
