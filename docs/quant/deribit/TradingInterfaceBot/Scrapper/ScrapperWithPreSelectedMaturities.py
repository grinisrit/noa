import logging
import warnings

import yaml

from docs.quant.deribit.TradingInterfaceBot.Utils import Currency


def scrap_available_instruments_by_extended_config(currency: Currency, cfg):
    from docs.quant.deribit.TradingInterfaceBot.SyncLib.AvailableRequests import get_instruments_by_currency_request
    from docs.quant.deribit.TradingInterfaceBot.Utils import InstrumentType, get_ticker_by_instrument_request
    from docs.quant.deribit.TradingInterfaceBot.SyncLib.Scrapper import send_request
    import pandas as pd
    import numpy as np
    make_subscriptions_list = send_request(get_instruments_by_currency_request(currency=currency,
                                                                               kind=InstrumentType.OPTION,
                                                                               expired=False))

    # Take only the result of answer. Now we have list of json contains information of option dotes.
    answer = make_subscriptions_list['result']
    available_maturities = pd.DataFrame(np.unique(list(map(lambda x: x["instrument_name"].split('-')[1], answer))))
    available_maturities.columns = ['DeribitNaming']
    available_maturities['RealNaming'] = pd.to_datetime(available_maturities['DeribitNaming'], format='%d%b%y')
    available_maturities = available_maturities.sort_values(by='RealNaming').reset_index(drop=True)
    print("Available maturities: \n", available_maturities)
    if not cfg["use_configuration_to_select_maturities"]:
        # selected_maturity = int(input("Select number of interested maturity "))
        selected_maturity = -1
        if selected_maturity == -1:
            warnings.warn("Selected list of instruments is empty")
            return []
        # selected_maturity = 3
        selected_maturity = available_maturities.iloc[selected_maturity]['DeribitNaming']
        print('\nYou select:', selected_maturity)

        selected = list(map(lambda x: x["instrument_name"],
                            list(filter(lambda x: (selected_maturity in x["instrument_name"]) and (
                                    x["option_type"] == "call" or "put"), answer))))

        get_underlying = send_request(get_ticker_by_instrument_request(selected[0]),
                                      show_answer=False)['result']['underlying_index']
        if 'SYN' not in get_underlying:
            selected.append(get_underlying)
        else:
            if cfg["raise_error_at_synthetic"]:
                raise ValueError("Cannot subscribe to order book for synthetic underlying")
            else:
                warnings.warn("Underlying is synthetic: {}".format(get_underlying))
        print("Selected Instruments")
        print(selected)
        return selected
    elif cfg["use_configuration_to_select_maturities"]:
        try:
            with open("/".join(__file__.split('/')[:-1]) + "/ConfigurationStorage/" + \
                      cfg["maturities_configuration_path"], "r") as ymlfile:
                maturities_selected = yaml.load(ymlfile, Loader=yaml.FullLoader)['maturities_list']
                print(maturities_selected)
        except FileNotFoundError:
            logging.warning("No configuration file with maturities")
            raise FileNotFoundError("ERROR!")

        total_instrument_list: list = []
        print(f'\nSelect maturities by configuration file: ({cfg["maturities_configuration_path"]})',
              maturities_selected)
        for maturity in maturities_selected:
            selected = list(map(lambda x: x["instrument_name"],
                                list(filter(lambda x: (maturity in x["instrument_name"]) and (
                                        x["option_type"] == "call" or "put"), answer))))

            get_underlying = send_request(get_ticker_by_instrument_request(selected[0]),
                                          show_answer=False)['result']['underlying_index']
            if 'SYN' not in get_underlying:
                selected.append(get_underlying)
            else:
                if cfg["raise_error_at_synthetic"]:
                    raise ValueError("Cannot subscribe to order book for synthetic underlying")
                else:
                    warnings.warn("Underlying is synthetic: {}".format(get_underlying))
            total_instrument_list.extend(selected)
        print("Selected Instruments")
        print(total_instrument_list)
        return total_instrument_list
