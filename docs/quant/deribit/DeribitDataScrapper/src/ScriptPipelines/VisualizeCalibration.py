from __future__ import annotations

import asyncio
import logging
import time
import threading

from DataBase import *
from OrderManager import OrderManager
from Utils import *
from Subsciption import *
from Strategy import *
from InstrumentManager import InstrumentManager

from SyncLib.AvailableRequests import get_ticker_by_instrument_request
from Scrapper.ScrapperWithPreSelectedMaturities import scrap_available_instruments_by_extended_config
from Scrapper.TradingInterface import validate_configuration_file, DeribitClient, scrap_available_instruments


async def start_scrapper(configuration_path=None):
    configuration = validate_configuration_file("../configuration.yaml")
    logging.basicConfig(
        level=configuration['orderBookScrapper']["logger_level"],
        format=f"%(asctime)s | [%(levelname)s] | [%(threadName)s] | %(name)s | FUNC: (%(filename)s).%(funcName)s(%(lineno)d) | %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"Loging.log"),
            logging.StreamHandler()])
    match configuration['orderBookScrapper']["currency"]:
        case "BTC":
            _currency = Currency.BITCOIN
        case "ETH":
            _currency = Currency.ETHER
        case _:
            loop.stop()
            raise ValueError("Unknown currency")

    derLoop = asyncio.new_event_loop()
    if not configuration["orderBookScrapper"]["use_configuration_to_select_maturities"]:
        instruments_list = await scrap_available_instruments(currency=_currency, cfg=configuration['orderBookScrapper'])
    else:
        instruments_list = await scrap_available_instruments_by_extended_config(currency=_currency, cfg=configuration['orderBookScrapper'])


    deribitWorker = DeribitClient(cfg=configuration, cfg_path="../configuration.yaml",
                                  instruments_listed=instruments_list, loopB=derLoop,
                                  client_currency=_currency)

    deribitWorker.add_order_manager()
    baseStrategy = EmptyStrategy()
    deribitWorker.add_strategy(baseStrategy)

    deribitWorker.start()
    th = threading.Thread(target=derLoop.run_forever)
    th.start()

    # TODO: implement auth for production
    if deribitWorker.testMode:
        while not deribitWorker.auth_complete:
            continue


if __name__ == '__main__':
    # Make sure that in configuration add_order_manager = True | add_instrument_manager = True
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(start_scrapper())
    loop.run_forever()
    time.sleep(1)

