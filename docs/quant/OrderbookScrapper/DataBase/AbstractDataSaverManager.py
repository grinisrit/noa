from abc import ABC, abstractmethod


class AbstractDataManager(ABC):
    @abstractmethod
    def add_order_book_content_limited_depth(self, bids, asks, change_id, timestamp, instrument_name):
        pass

    @abstractmethod
    def add_instrument_change_order_book_unlimited_depth(self, request_change_id: int, request_previous_change_id: int,
                                                         change_timestamp: int,
                                                         bids_list: list[list[str, float, float]],
                                                         asks_list: list[list[str, float, float]]
                                                         ):
        pass

    @abstractmethod
    def add_instrument_init_snapshot(self, instrument_name: str,
                                     start_instrument_scrap_time: int,
                                     request_change_id: int,
                                     bids_list,
                                     asks_list: list[list[str, float, float]]
                                     ):
        pass
