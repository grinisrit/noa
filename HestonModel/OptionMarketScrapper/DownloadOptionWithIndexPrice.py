from AvailableCurrencies import Currency
from AvailableInstrumentType import InstrumentType
from Scrapper import send_request, send_batch_of_requests
from AvailableRequests import get_instruments_by_currency_request, get_ticker_by_instrument_request
from pprint import pprint
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
    selected_maturity = 0
    selected_maturity = available_maturities.iloc[selected_maturity]['DeribitNaming']
    print('\nYou select:', selected_maturity)

    select_all_strikes

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