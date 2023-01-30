from docs.quant.deribit.TradingInterfaceBot.Subsciption.AbstractSubscription import AbstractSubscription, flatten

from numpy import ndarray
from functools import partial
from pandas import DataFrame

from docs.quant.deribit.TradingInterfaceBot.DataBase.mysqlRecording.cleanUpRequestsLimited import \
    REQUEST_TO_CREATE_LIMITED_ORDER_BOOK_CONTENT


class NullSub(AbstractSubscription):

    def _place_here_tables_names_and_creation_requests(self):
        self.tables_names = ["TEST_TABLE_NULL"]
        self.tables_names_creation = list(map(partial(REQUEST_TO_CREATE_LIMITED_ORDER_BOOK_CONTENT,
                                                      depth_size=2), self.tables_names))

    def create_columns_list(self) -> list[str]:
        columns = ["CHANGE_ID", "NAME_INSTRUMENT", "TIMESTAMP_VALUE"]
        columns.extend(map(lambda x: [f"BID_{x}_PRICE", f"BID_{x}_AMOUNT"], range(2)))
        columns.extend(map(lambda x: [f"ASK_{x}_PRICE", f"ASK_{x}_AMOUNT"], range(2)))

        columns = flatten(columns)
        columns[0] = "TEST_UNIT_TABLE"
        columns[1] = "UNKNOWN_TABLE"
        return columns

    def create_subscription_request(self) -> str:
        pass

    def _process_response(self, response: dict):
        pass

    def extract_data_from_response(self, input_response: dict) -> ndarray:
        pass

    def _record_to_daemon_database_pipeline(self, record_dataframe: DataFrame, tag_of_data: str) -> DataFrame:
        pass
