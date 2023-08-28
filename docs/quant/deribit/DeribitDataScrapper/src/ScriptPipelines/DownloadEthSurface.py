import logging

from TradingInterfaceBot.Scrapper.TradingInterface import validate_configuration_file, DeribitClient
from TradingInterfaceBot.Utils.AvailableCurrencies import Currency
import asyncio
import threading


async def scrap_all_instruments_from_currency(currency: Currency, cfg):
    """
    Функция для получения инструментов по всей поверхности.
    Предлагается ввод пользователем конкретного maturity
    :param currency: Валюта. BTC | ETH | SOL
    :param cfg: файл конфигурации бота
    :return: LIST[Instrument-name]
    """
    from TradingInterfaceBot.SyncLib.AvailableRequests import get_instruments_by_currency_request, get_ticker_by_instrument_request
    from TradingInterfaceBot.Utils.AvailableInstrumentType import InstrumentType
    from TradingInterfaceBot.SyncLib.Scrapper import send_request
    import pandas as pd
    import numpy as np
    make_subscriptions_list = await send_request(get_instruments_by_currency_request(currency=currency,
                                                                                     kind=InstrumentType.OPTION,
                                                                                     expired=False))

    all_available_underlying = await send_request(get_instruments_by_currency_request(currency=currency,
                                                                                     kind=InstrumentType.FUTURE,
                                                                                     expired=False))
    # Take only the result of answer. Now we have list of json contains information of option dotes.
    answer = make_subscriptions_list['result']
    answer_underlying = all_available_underlying['result']
    available_maturities = pd.DataFrame(np.unique(list(map(lambda x: x["instrument_name"].split('-')[1], answer))))

    available_maturities_underlying = np.unique(list(map(lambda x: x["instrument_name"].split('-')[1], answer_underlying)))

    available_maturities_underlying = pd.DataFrame(
        list(filter(lambda x: x != 'PERPETUAL', available_maturities_underlying))
        )
    available_maturities.columns = ['DeribitNaming']
    available_maturities_underlying.columns = ['DeribitNaming']

    available_maturities['RealNaming'] = pd.to_datetime(available_maturities['DeribitNaming'], format='%d%b%y')
    available_maturities_underlying['RealNaming'] = pd.to_datetime(available_maturities_underlying['DeribitNaming'], format='%d%b%y')

    available_maturities = available_maturities.sort_values(by='RealNaming').reset_index(drop=True)

    available_maturities_underlying = available_maturities_underlying.sort_values(by='RealNaming').reset_index(drop=True)
    print("Available maturities: \n", available_maturities)
    print("Available maturities for underlying", available_maturities_underlying)
    # TODO: uncomment
    full_list = []
    for selected_maturity in available_maturities.DeribitNaming:

        selected = list(map(lambda x: x["instrument_name"],
                            list(filter(lambda x: (selected_maturity in x["instrument_name"]) and (
                                    x["option_type"] == "call" or "put"), answer))))

        get_underlying = await send_request(get_ticker_by_instrument_request(selected[0]),
                                            show_answer=False)
        get_underlying = get_underlying['result']['underlying_index']
        if 'SYN' not in get_underlying:
            selected.append(get_underlying)
        else:
            if cfg["raise_error_at_synthetic"]:
                raise ValueError("Cannot subscribe to order book for synthetic underlying")
            else:
                logging.warning("Underlying is synthetic: {}".format(get_underlying))
        print(f"For maturity {selected_maturity} Selected Instruments: {selected}")

        full_list.extend(selected)
    full_list.extend([f"{currency.currency}-{obj}" for obj in available_maturities_underlying.DeribitNaming.values])
    print(f"Full list is {full_list}")
    print(f"Len of full list is {len(full_list)}")
    return full_list


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
    instruments_list = await scrap_all_instruments_from_currency(currency=_currency, cfg=configuration['orderBookScrapper'])

    deribitWorker = DeribitClient(cfg=configuration, cfg_path="../configuration.yaml",
                                  instruments_listed=instruments_list, loopB=derLoop,
                                  client_currency=_currency)


    deribitWorker.start()
    th = threading.Thread(target=derLoop.run_forever)
    th.start()

    # TODO: implement auth for production
    if deribitWorker.testMode:
        while not deribitWorker.auth_complete:
            continue


if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(start_scrapper())
    loop.run_forever()