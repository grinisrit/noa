import asyncio
import websockets
import json
from pprint import pprint

import AvailableCurrencies
import AvailableInstrumentType


async def call_api(msg):
    async with websockets.connect('wss://test.deribit.com/ws/api/v2') as websocket:
        await websocket.send(msg)
        while websocket.open:
            response = await websocket.recv()
            return response


def send_request(msg) -> json:
    _answer = json.loads(asyncio.get_event_loop().run_until_complete(call_api(json.dumps(msg))))
    pprint(_answer)
    return _answer


if __name__ == "__main__":
    from AvailableRequests import get_instruments_by_currency_request
    answer = send_request(get_instruments_by_currency_request(AvailableCurrencies.Currency.BITCOIN,
                                                              AvailableInstrumentType.InstrumentType.OPTION))
