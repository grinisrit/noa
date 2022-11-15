from docs.quant.OrderbookScrapper.Utils.AvailableCurrencies import Currency
from docs.quant.OrderbookScrapper.Utils.AvailableInstrumentType import InstrumentType
from Scrapper import send_request
from AvailableRequests import get_instruments_by_currency_request, get_ticker_by_instrument_request
from docs.quant.OrderbookScrapper.SyncLib.DeribitConnectionOld import DeribitConnectionOld
import pandas as pd
import numpy as np
from datetime import datetime


def download_data_for_SABR(currency: Currency, save_information=False):
    answer = send_request(get_instruments_by_currency_request(currency=currency,
                                                              kind=InstrumentType.OPTION,
                                                              expired=False))

    answer_id = answer['id']
    # Take only the result of answer. Now we have list of json contains information of option dotes.
    answer = answer['result']
    available_maturities = pd.DataFrame(np.unique(list(map(lambda x: x["instrument_name"].split('-')[1], answer))))
    available_maturities.columns = ['DeribitNaming']
    available_maturities['RealNaming'] = pd.to_datetime(available_maturities['DeribitNaming'], format='%d%b%y')
    available_maturities = available_maturities.sort_values(by='RealNaming').reset_index(drop=True)
    print("Available maturities: \n", available_maturities)

    # TODO: uncomment
    # selected_maturity = int(input("Select number of interested maturity "))
    selected_maturity = 2
    selected_maturity = available_maturities.iloc[selected_maturity]['DeribitNaming']
    print('\nYou select:', selected_maturity)

    selected = list(map(lambda x: x["instrument_name"]
        ,list(filter(lambda x: (selected_maturity in x["instrument_name"]) and (x["option_type"] == "call"), answer))))

    print(selected)

    get_underlying = send_request(get_ticker_by_instrument_request(selected[0]),
                                  show_answer=False)['result']['underlying_index']

    CURRENT_TIME = int(datetime.now().timestamp() * 1000)

    print(get_underlying)
    # Download underlying
    deribit_old = DeribitConnectionOld("")
    df = deribit_old.get_instrument_last_prices(get_underlying, 10_00, number_of_requests=5,
                                            date_of_start_loading_data=CURRENT_TIME)
    df.index = pd.to_datetime(df.timestamp * 10 ** 6)
    underlyingBars = DeribitConnectionOld.create_bars(df)

    # Download Option prices
    strikes_bars = list()
    for select in selected:
        try:
            df = deribit_old.get_instrument_last_prices(select, 10_00, number_of_requests=1,
                                                        date_of_start_loading_data=CURRENT_TIME)
        except IndexError:
            print("No trades Data for Instrument:", select)
            continue
        df.index = pd.to_datetime(df.timestamp * 10 ** 6)
        strikes_bars.append(DeribitConnectionOld.create_bars(df))

    # TODO: Change / to \\ on Windows System
    if SAVE_INFO:
        PATH = f"{SAVE_STORAGE_NAME}/{selected_maturity}"
        if os.path.isdir(PATH):
            shutil.rmtree(PATH, ignore_errors=True)
        os.mkdir(PATH)
        underlyingBars.to_csv(f"{PATH}/underlyingBars.csv")

        for i, _ in enumerate(strikes_bars):
            _.to_csv(f"{PATH}/strikeBars_{selected[i].split('-')[2]}.csv")






if __name__ == "__main__":
    SAVE_INFO = True
    DELETE_OLD = False
    SAVE_STORAGE_NAME = "saveStorage"
    if SAVE_INFO:
        import os
        import shutil
        if DELETE_OLD:
            if os.path.isdir(SAVE_STORAGE_NAME):
                shutil.rmtree(SAVE_STORAGE_NAME, ignore_errors=True)
            os.mkdir(SAVE_STORAGE_NAME)
        if not DELETE_OLD:
            if os.path.isdir(SAVE_STORAGE_NAME):
                pass
            else:
                raise ValueError("No working Dir")

    download_data_for_SABR(Currency.BITCOIN, save_information=SAVE_INFO)