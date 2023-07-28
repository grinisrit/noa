import time

from docs.quant.deribit.TradingInterfaceBotJulyVersion.Subsciption.AbstractSubscription import AbstractSubscription, flatten, RequestTypo
from docs.quant.deribit.TradingInterfaceBotJulyVersion.Utils import *

from numpy import ndarray
from pandas import DataFrame
from typing import List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from docs.quant.deribit.TradingInterfaceBotJulyVersion.Scrapper.TradingInterface import DeribitClient

    scrapper_typing = DeribitClient
else:
    scrapper_typing = object


class UserPortfolioSubscription(AbstractSubscription):
    tables_names = ["User_Portfolio_{}"]

    def __init__(self, scrapper: scrapper_typing):
        self.tables_names = [f"User_Portfolio_test"]
        self.tables_names_creation = list(map(REQUEST_TO_CREATE_USER_PORTFOLIO_TABLE, self.tables_names))

        super(UserPortfolioSubscription, self).__init__(scrapper=scrapper, request_typo=RequestTypo.PRIVATE)
        self.number_of_columns = 12
        self.instrument_name_instrument_id_map = self.scrapper.instrument_name_instrument_id_map

    def _place_here_tables_names_and_creation_requests(self):
        self.tables_names = [f"User_Portfolio_test"]
        self.tables_names_creation = list(map(REQUEST_TO_CREATE_USER_PORTFOLIO_TABLE, self.tables_names))

    def create_columns_list(self) -> List[str]:
        columns = ["CHANGE_ID", "CREATION_TIMESTAMP", "TOTAL_PL", "MARGIN_BALANCE", "MAINTENANCE_MARGIN",
                   "INITIAL_MARGIN", "ESTIMATED_LIQUIDATION_RATIO", "EQUITY", "DELTA_TOTAL",
                   "BALANCE", "AVAILABLE_WITHDRAWAL_FUNDS", "AVAILABLE_FUNDS"]
        columns = flatten(columns)
        return columns

    async def _process_response(self, response: dict):
        # SUBSCRIPTION processing
        if response['method'] == "subscription":
            # ORDER BOOK processing. For constant book depth
            if 'params' in response:
                if 'channel' in response['params']:
                    if 'portfolio' in response['params']['channel']:
                        # TODO: HERE IS PROBLEM
                        # if self.scrapper.connected_strategy is not None:
                        #     await self.scrapper.connected_strategy.on_trade_update(callback=response)

                        if self.database:
                            await self.database.add_data(
                                update_line=self.extract_data_from_response(input_response=response)
                            )
                            return 1

    def extract_data_from_response(self, input_response: dict) -> ndarray:
        _data_obj = input_response["params"]["data"]

        _change_id = 666
        _timestamp = int(time.time_ns() / 1_000_000)
        _total_pnl = _data_obj["total_pl"]
        _margin_balance = _data_obj["margin_balance"]
        _maintenance_margin = _data_obj["maintenance_margin"]
        _initial_margin = _data_obj["initial_margin"]
        _estimated_liquidation_ratio = _data_obj["estimated_liquidation_ratio"]
        _equity = _data_obj["equity"]
        _delta_total = _data_obj["delta_total"]
        _balance = _data_obj["balance"]
        _available_withdrawal_funds = _data_obj["available_withdrawal_funds"]
        _available_funds = _data_obj["available_funds"]

        _ret_arr = [_change_id, _timestamp, _total_pnl, _margin_balance, _maintenance_margin,
                    _initial_margin, _estimated_liquidation_ratio, _equity, _delta_total,
                    _balance, _available_withdrawal_funds, _available_funds]
        return np.array(_ret_arr)

    def _create_subscription_request(self):
        self._user_portfolio_changes_subscription_request()

    def _record_to_daemon_database_pipeline(self, record_dataframe: DataFrame, tag_of_data: str) -> DataFrame:
        if 'CHANGE_ID' in record_dataframe.columns:
            return record_dataframe.iloc[:, 1:]
        return record_dataframe

    def _user_portfolio_changes_subscription_request(self):
        subscription_message = \
            MSG_LIST.get_user_portfolio_request(
                currency=self.scrapper.client_currency,
            )
        self.scrapper.send_new_request(request=subscription_message)

