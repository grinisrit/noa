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


def send_request(msg, show_answer=False) -> json:
    _answer = json.loads(asyncio.get_event_loop().run_until_complete(call_api(json.dumps(msg))))
    if show_answer:
        pprint(_answer)

    return _answer


def send_batch_of_requests(msg_list, show_answer=False) -> json:
    return_list = list()

    async def circle(mutable_list):
        messages = [call_api(json.dumps(msg)) for msg in msg_list]
        # Schedule three calls *concurrently*:
        mutable_list += await asyncio.gather(*messages)

    asyncio.run(circle(return_list))
    if show_answer:
        pprint(return_list)

    return list(zip(msg_list, [json.loads(element) for element in return_list]))


def test_get_multiple_ticker_request():
    names = ["BTC-29SEP23-10000-C", "BTC-30DEC22-10000-C", "BTC-30JUN23-10000-C"
             "BTC-31MAR23-10000-C", "BTC-25NOV22-12000-C", "BTC-25NOV22-14000-C",
             "BTC-28OCT22-14000-C", "BTC-29SEP23-14000-C"]

    messages_list = list(map(get_ticker_by_instrument_request, names))
    pprint(send_batch_of_requests(messages_list, show_answer=True))


if __name__ == "__main__":
    from AvailableRequests import get_instruments_by_currency_request, get_ticker_by_instrument_request

    test_get_multiple_ticker_request()

    # answer = send_request(get_instruments_by_currency_request(AvailableCurrencies.Currency.BITCOIN,
    #                                                           AvailableInstrumentType.InstrumentType.OPTION))
