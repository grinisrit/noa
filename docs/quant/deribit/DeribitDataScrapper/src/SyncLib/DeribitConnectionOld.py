# Old Deribit API. With it's endpoints we can get historical data from last 5 years.

import requests
from Utils.AvailableCurrencies import Currency
from Utils.AvailableInstruments import Instrument
from pprint import pprint
import pandas as pd
from tqdm import tqdm
import time


class DeribitConnectionOld:
    __slots__ = {"api_key", "connectionStatus"}

    def __init__(self, api_key):
        self.api_key = api_key

    @staticmethod
    def call_api_without_key(additional, query: dict):
        """
        Method to call deribit api.
        :param additional: endpoint for request
        :param query: query of parameters
        :return: response json.
        """
        response = requests.get(f"https://history.deribit.com/api/v2/public/{additional}", params=query)
        return response.json()

    def get_last_trades(self, currency: Currency, number_of_last_trades: int):
        """
        Get trades by instrument. For each request, we take the last N trades,
        :param currency: Currency Enum.
        :param number_of_last_trades: How many last N trades we want to get.
        :return:
        """
        if number_of_last_trades > 10_000:
            raise ValueError("Too much number_of_last_trades")
        query = {'currency': currency.currency, 'count': f'{number_of_last_trades}', 'include_old': 'true'}
        additional = "get_last_trades_by_currency"
        pprint(self.call_api_without_key(additional=additional, query=query))

    def get_instrument_position(self, instrument_name: str):
        query = {''}
    def get_instrument_last_prices(self, instrument: Instrument | str, number_of_last_trades: int, number_of_requests=10_00,
                                   date_of_start_loading_data=None):
        """
        Get trades by instrument. For each request, we take the last N trades, we make such requests M pieces.
        The API method takes as parameters end_timestamp, after each request, the last collected time_stamp
        set as end_timestamp for the next request. Duplicates are cut using drop_duplicates.
        :param instrument: Get Info about Instrument ENUM
        :param number_of_last_trades: Dotes in one request N
        :param number_of_requests: Number of requests M
        :param date_of_start_loading_data: Time in MicroSec for what date we start collecting data
        :return: pd.DataFrame with information of all trades.
        """
        if number_of_last_trades > 10_000:
            raise ValueError("Too much number_of_last_trades")

        if type(instrument) != str:
            instrument_request_name = instrument.instrument
        else:
            instrument_request_name = instrument

        if date_of_start_loading_data is None:
            query = {'instrument_name': instrument_request_name, 'count': f'{number_of_last_trades}',
                     'include_old': 'true', 'end_timestamp': f'{int(round(time.time() * 1000))}'}
        else:
            query = {'instrument_name': instrument_request_name, 'count': f'{number_of_last_trades}',
                     'include_old': 'true', 'end_timestamp': f'{int(date_of_start_loading_data)}'}

        additional = "get_last_trades_by_instrument_and_time"
        response = self.call_api_without_key(additional=additional, query=query)
        df = pd.DataFrame(response["result"]["trades"])

        for _pointer in tqdm(range(0, number_of_requests)):
            query = {'instrument_name': instrument_request_name, 'count': f'{number_of_last_trades}',
                     'include_old': 'true', 'end_timestamp': f"{df.iloc[-1].timestamp}"}
            additional = "get_last_trades_by_instrument_and_time"
            response = self.call_api_without_key(additional=additional, query=query)
            df = pd.concat([df, pd.DataFrame(response["result"]["trades"])]).drop_duplicates(
                subset=['trade_seq']).reset_index(drop=True)

        return df

    @classmethod
    def create_bars(cls, tick_dataframe: pd.DataFrame, by_what='minutes'):
        tick_dataframe.index = pd.to_datetime(tick_dataframe.timestamp * 10 ** 6)
        if by_what == 'minutes':
            tick_dataframe['minutesTime'] = pd.to_datetime(tick_dataframe.timestamp * 10 ** 6,
                                                           format='%Y%m%d %H:%M:%S').dt.floor('T')
        else:
            raise ValueError('Unknown type of slicing method')

        def _slice_on_bars(obj):
            return pd.DataFrame([obj.iloc[-1], max(obj), min(obj), obj.iloc[0]]).T

        bars = tick_dataframe.groupby(by='minutesTime').price.apply(_slice_on_bars).droplevel(1)
        bars.columns = ['Open', 'High', 'Low', 'Close']

        return bars


if __name__ == "__main__":

    deribit = DeribitConnectionOld(api_key='asdadssadas')
    # Sample to get information about BTC-PERPETUAL.
    deribit.get_instrument_last_prices(Instrument.BTC_PERPETUAL, 100, number_of_requests=5)