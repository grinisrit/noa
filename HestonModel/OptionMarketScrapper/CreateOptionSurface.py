from AvailableCurrencies import Currency
from AvailableInstrumentType import InstrumentType
from Scrapper import send_request, send_batch_of_requests
from AvailableRequests import get_instruments_by_currency_request, get_ticker_by_instrument_request
from pprint import pprint
import pandas as pd
import numpy as np
from tqdm import tqdm

from typing import Callable


def get_price_of_option_function(elementaryName: str) -> float:
    """
    Function with logic what price we get like Option price
    :param elementaryName: Instrument name. Looks like BTC-18OCT22-17000-C
    :return: float. Price
    """
    elem_last_info = send_request(get_ticker_by_instrument_request(elementaryName))
    return elem_last_info["result"]["last_price"]

def get_price_of_option_batch_function(answer_list:list[list[dict, float]]) -> dict:
    def what_field_we_need_to_get_from_ticker(ticker_obj):
        return ticker_obj["result"]["mark_price"]
    return dict([[elementary[0]["params"]["instrument_name"], what_field_we_need_to_get_from_ticker(elementary[1])]
                 for elementary in answer_list])


def create_option_surface(currency: Currency):
    answer = send_request(get_instruments_by_currency_request(currency=currency,
                                                              kind=InstrumentType.OPTION,
                                                              expired=False))

    answer_id = answer['id']
    # Take only the result of answer. Now we have list of json contains information of option dotes.
    answer = answer['result']
    # Assertion for correct inside settlement_currency
    assert np.unique(list(map(lambda x: x["instrument_name"].split('-')[0], answer)))[0] == currency.currency
    # Select only put options
    puts_answer = list(filter(lambda x: x["option_type"] == "call", answer))
    # Zip instrument_names with objects to speed up matrix generation
    _zipped_map = dict([[_elementaryOBJ["instrument_name"], _elementaryOBJ] for _elementaryOBJ in puts_answer])
    # Select unique expiration dates from all put options. Next it will be x axis of table
    unique_puts_maturities = np.unique(list(map(lambda x: x["instrument_name"].split('-')[1], puts_answer)))
    # Get all available strikes for all maturities. Next it will be y axis of table
    unique_puts_strikes = np.unique(list(map(lambda x: float(x["instrument_name"].split('-')[2]), puts_answer)))

    # Create empty surface matrix.
    _numpyMatrixMask = np.empty((len(unique_puts_strikes), len(unique_puts_maturities), ))
    _numpyMatrixMask.fill(np.NaN)
    surface_matrix = pd.DataFrame(_numpyMatrixMask)
    del _numpyMatrixMask
    surface_matrix.columns = unique_puts_maturities
    surface_matrix.index = unique_puts_strikes


    # TODO: not int strikes?
    construct_instrument_name_for_call: Callable[[str, float], str] = lambda x, y: f"{currency.currency}-{x}-{int(y)}-C"
    # Create messages for multiple request
    names = list(_zipped_map.keys())
    print(names)
    messages_list = list(map(get_ticker_by_instrument_request, names[:2]))
    _answer = send_batch_of_requests(messages_list, show_answer=False)
    pprint(_answer)
    print(get_price_of_option_batch_function(_answer))
    # Fill All Available Instruments with Option Prices. Logic at get_price_of_option_function.
    # TODO: mapping for pandas?



if __name__ == "__main__":
    create_option_surface(Currency.BITCOIN)