import asyncio
import time

import json
from pprint import pprint
from tqdm import tqdm
import websocket as web_sock
import websockets

from threading import Thread
from collections import deque
import AvailableCurrencies
import AvailableInstrumentType

from AvailableRequests import test_message

async def call_api(msg):
    # async with websockets.connect('wss://www.deribit.com/ws/api/v2') as websocket:
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
    """
    Be careful with _one_request_batch_size. Deribit can ban (not permanent) your ip if value is big. Need to be tested max value.

    :param msg_list:
    :param show_answer:
    :return:
    """
    return_list = list()
    _one_request_batch_size = 25

    async def circle(mutable_list, iteration_of_request, one_request_size=_one_request_batch_size, load_mod=False):
        if not load_mod:
            messages = [call_api(json.dumps(msg)) for msg in msg_list[iteration_of_request * one_request_size:
                                                                      (iteration_of_request+1) * one_request_size]]
        else:
            messages = [call_api(json.dumps(msg)) for msg in msg_list[iteration_of_request * one_request_size:]]
        # Schedule three calls *concurrently*:
        mutable_list += await asyncio.gather(*messages)

    if len(msg_list) >= _one_request_batch_size:
        for iteration_number in tqdm(range(0, len(msg_list) // _one_request_batch_size)):
            asyncio.run(circle(return_list, iteration_of_request=iteration_number))
            time.sleep(2)
        if len(return_list) != len(msg_list):
            print(f"Need to load mod part of map | Downloaded size {len(return_list)} | Message size {len(msg_list)}")
            asyncio.run(circle(return_list, iteration_of_request=len(msg_list) // _one_request_batch_size))

    else:
        asyncio.run(circle(return_list, iteration_of_request=0, one_request_size=len(msg_list)))
        time.sleep(2)

    if show_answer:
        pprint(return_list)

    return list(zip(msg_list, [json.loads(element) for element in return_list]))


async def api_listener(websocket):
    while True:
        try:
            message = await websocket.recv()
            print("< {}".format(message))

        except websockets.ConnectionClosed as cc:
            print('Connection closed')


def test_get_multiple_ticker_request():
    names = ["BTC-29SEP23-10000-C", "BTC-30DEC22-10000-C", "BTC-30JUN23-10000-C"
             "BTC-31MAR23-10000-C", "BTC-25NOV22-12000-C", "BTC-25NOV22-14000-C",
             "BTC-28OCT22-14000-C", "BTC-29SEP23-14000-C"]

    messages_list = list(map(get_ticker_by_instrument_request, names))
    pprint(send_batch_of_requests(messages_list, show_answer=True))



if __name__ == "__main__":
    from AvailableRequests import get_instruments_by_currency_request, get_ticker_by_instrument_request

    # test_get_multiple_ticker_request()
    # send_request(get_ticker_by_instrument_request("BTC-29SEP23-10000-C"), show_answer=True)
    # answer = send_request(get_instruments_by_currency_request(AvailableCurrencies.Currency.BITCOIN,
    #                                                           AvailableInstrumentType.InstrumentType.OPTION))

