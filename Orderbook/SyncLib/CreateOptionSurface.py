from Orderbook.Utils.AvailableCurrencies import Currency
from Orderbook.Utils.AvailableInstrumentType import InstrumentType
from Scrapper import send_request, send_batch_of_requests
from AvailableRequests import get_instruments_by_currency_request, get_ticker_by_instrument_request
from pprint import pprint
import pandas as pd
import numpy as np
from datetime import datetime


from typing import Callable
global SAVE_STORAGE_NAME


def get_price_of_option_function(elementaryName: str) -> float:
    """
    Function with logic what price we get like Option price
    :param elementaryName: Instrument name. Looks like BTC-18OCT22-17000-C
    :return: float. Price
    """
    elem_last_info = send_request(get_ticker_by_instrument_request(elementaryName))
    return elem_last_info["result"]["last_price"]


def get_price_of_option_batch_function(answer_list: list[list[dict, float | list[float]]]) -> dict:
    def what_field_we_need_to_get_from_ticker(ticker_obj) -> float | list:
        # HARDCODED underlying_price at position 1!!!
        return [ticker_obj["result"]["mark_iv"],
                ticker_obj["result"]["underlying_price"],
                ticker_obj["result"]["underlying_index"],
                ticker_obj["result"]["last_price"],
                ticker_obj["result"]["mark_price"]]

    return dict([[elementary[0]["params"]["instrument_name"], what_field_we_need_to_get_from_ticker(elementary[1])]
                 for elementary in answer_list])


def fill_surface_matrix(surface_matrix: pd.DataFrame, _answer: dict, number_of_element: int,
                        construct_instrument_name_for_call) -> pd.DataFrame:

    IF_NEED_OOM = False
    pprint(_answer)
    pass_matrix = surface_matrix.copy()
    for _strike in pass_matrix.index:
        for _maturity in pass_matrix.columns:
            element_name = construct_instrument_name_for_call(_maturity, _strike)
            if element_name in _answer.keys():
                # Out of the money
                if IF_NEED_OOM:
                    if _strike > _answer.get(element_name)[1]:
                        pass_matrix.loc[_strike, _maturity] = _answer.get(element_name)[number_of_element]
                else:
                    pass_matrix.loc[_strike, _maturity] = _answer.get(element_name)[number_of_element]

    return pass_matrix


def create_option_surface(currency: Currency, save_information=False):
    answer = send_request(get_instruments_by_currency_request(currency=currency,
                                                              kind=InstrumentType.OPTION,
                                                              expired=False))

    answer_id = answer['id']
    # Take only the result of answer. Now we have list of json contains information of option dotes.
    answer = answer['result']
    # Assertion for correct inside settlement_currency
    assert np.unique(list(map(lambda x: x["instrument_name"].split('-')[0], answer)))[0] == currency.currency
    # Select only put options
    calls_answer = list(filter(lambda x: x["option_type"] == "call", answer))
    # Zip instrument_names with objects to speed up matrix generation
    _zipped_map = dict([[_elementaryOBJ["instrument_name"], _elementaryOBJ] for _elementaryOBJ in calls_answer])
    # Select unique expiration dates from all put options. Next it will be x axis of table
    unique_puts_maturities = np.unique(list(map(lambda x: x["instrument_name"].split('-')[1], calls_answer)))
    # Get all available strikes for all maturities. Next it will be y axis of table
    unique_puts_strikes = np.unique(list(map(lambda x: float(x["instrument_name"].split('-')[2]), calls_answer)))

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

    messages_list = list(map(get_ticker_by_instrument_request, names))
    _answer = send_batch_of_requests(messages_list, show_answer=False)
    _answer = get_price_of_option_batch_function(_answer)
    # TODO: mapping for pandas?

    # Select element at extend_info
    # Number of element: = What we want to get
    filled_matrix = fill_surface_matrix(surface_matrix=surface_matrix,
                                        _answer=_answer,
                                        number_of_element=3,
                                        construct_instrument_name_for_call=construct_instrument_name_for_call)

    print(filled_matrix)
    print(filled_matrix.columns)
    if save_information:
        date_now = datetime.now().date()
        filled_matrix.to_csv(f"{SAVE_STORAGE_NAME}/optionMap_{date_now}.csv")


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

    create_option_surface(Currency.BITCOIN, save_information=SAVE_INFO)
