
import asyncio
import logging
import os
import time
import threading

import yaml

from Utils import *
from Strategy import *

from Scrapper.ScrapperWithPreSelectedMaturities import scrap_available_instruments_by_extended_config
from Scrapper.TradingInterface import validate_configuration_file, DeribitClient, scrap_available_instruments


async def start_scrapper(configuration_path=None):
    try:
        print('getcwd:      ', os.getcwd())
        print('__file__:    ', __file__)
        print(f"script dir {os.path.dirname(__file__)}")
        os.chdir(os.path.dirname(__file__))
        configuration = validate_configuration_file("configuration.yaml")
        with open('developerConfiguration.yaml', "r") as ymlfile:
            devCFG = yaml.load(ymlfile, Loader=yaml.FullLoader)
        with open('StrategyConfig.yaml', "r") as ymlfile:
            CFG_Strategy = yaml.load(ymlfile, Loader=yaml.FullLoader)

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


        deribitWorker = DeribitClient(cfg=configuration, cfg_path=configuration,
                                      instruments_listed=instruments_list, loopB=derLoop,
                                      client_currency=_currency, dev_cfg=devCFG)

        deribitWorker.add_order_manager()
        baseStrategy = EmptyStrategy(CFG_Strategy)
        deribitWorker.add_strategy(baseStrategy)

        deribitWorker.start()
        th = threading.Thread(target=derLoop.run_forever)
        th.start()

        # TODO: implement auth for production
        if deribitWorker.testMode:
            while not deribitWorker.auth_complete:
                continue
    except Exception as E:
        print("error", E)
        logging.exception(E)

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(start_scrapper())
    loop.run_forever()
    time.sleep(1)