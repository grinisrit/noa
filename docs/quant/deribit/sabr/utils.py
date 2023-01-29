import asyncio
import websockets
import json
import pandas as pd
import nest_asyncio
from loguru import logger
import datetime

nest_asyncio.apply()
import warnings
from typing import Union

warnings.filterwarnings("ignore")


def collect_all_instruments_ids(
    currency: str = "BTC", kind: str = "option"
) -> pd.DataFrame:
    msg = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "public/get_instruments",
            "params": {"currency": "BTC", "kind": "option", "expired": False},
        }
    )

    async def call_api(msg):
        async with websockets.connect("wss://test.deribit.com/ws/api/v2") as websocket:
            await websocket.send(msg)
            while websocket.open:
                response = await websocket.recv()
                return json.loads(response)

    instruments = asyncio.get_event_loop().run_until_complete(call_api(msg))
    instruments_df = []
    for instrument in instruments["result"]:
        instruments_df.append(
            [instrument["instrument_name"], instrument["instrument_id"]]
        )
    instruments_df = pd.DataFrame(
        instruments_df, columns=["instrument_name", "instrument_id"]
    )
    return instruments_df


def collect_single_instrument_data(
    instrument_id: int, number_of_ticks: int, depth: int = 1.0
) -> pd.DataFrame:
    msg = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "public/get_order_book_by_instrument_id",
            "id": instrument_id,
            "params": {"instrument_id": instrument_id, "depth": depth},
        }
    )

    async def call_api(msg):
        ticks = []
        counter = 0
        async with websockets.connect("wss://test.deribit.com/ws/api/v2") as websocket:
            while websocket.open:
                if counter >= number_of_ticks:
                    break
                await websocket.send(msg)
                response = await websocket.recv()
                ticks.append(json.loads(response))
                counter += 1
                continue
        logger.info(f"Collected {number_of_ticks} ticks for id = {instrument_id}")
        return ticks

    prices = asyncio.get_event_loop().run_until_complete(call_api(msg))

    df = []
    columns = [
        "instrument_name",
        "timestamp",
        "underlying_price",
        "mark_iv",
        "mark_price",
        "best_bid_price",
        "best_ask_price",
    ]

    for result in prices:
        res_tmp = result["result"]
        df.append(
            [
                res_tmp["instrument_name"],
                res_tmp["timestamp"],
                res_tmp["underlying_price"],
                res_tmp["mark_iv"],
                res_tmp["mark_price"],
                res_tmp["best_bid_price"],
                res_tmp["best_ask_price"],
            ]
        )
    df = pd.DataFrame(df, columns=columns)
    df = df.drop_duplicates()
    df["human_timestamp"] = df["timestamp"].apply(
        lambda x: datetime.datetime.fromtimestamp(x / 1000.0).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )
    )
    df["strike"] = int(df.iloc[0].instrument_name.split("-")[2])
    return df


def get_human_timestamp(timestamp: int):
    """Get human-readable timestamp from linux date"""
    return datetime.datetime.fromtimestamp(timestamp / 1000000.0).strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )


def get_difference_between_now_and_expirety_date(now: int, expiration_date: int):
    """Returns the time between now and expiration date in YEARS"""
    return (now - expiration_date) / (1_000_000 * 60 * 60 * 24 * 365)

