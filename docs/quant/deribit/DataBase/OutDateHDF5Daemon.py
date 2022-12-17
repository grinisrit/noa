import time
from typing import Union, Optional, Dict

import numpy as np

from docs.quant.deribit.DataBase.AbstractDataSaverManager import AbstractDataManager
import logging
import yaml
import pandas as pd
import os



class HDF5Daemon(AbstractDataManager):
    TEMPLATE_FOR_LIMIT_DEPTH_TABLES_NAME = "TABLE_DEPTH_{}"

    LIMITS_OF_COLUMNS = {
        "CHANGE_ID": 15,
        "NAME_INSTRUMENT": 20,
        "TIMESTAMP_VALUE": 16,
        "BID_PRICE": 10,
        "BID_AMOUNT": 10,
        "ASK_PRICE": 10,
        "ASK_AMOUNT": 10,

    }

    def __init__(self, constant_depth_mode: Union[bool, int], clean_tables: bool = False):
        # Config file
        with open("../configuration.yaml", "r") as ymlfile:
            self.cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)['hdf5']

        super().__init__("../configuration.yaml")
        logging.basicConfig(
            level='INFO',
            format='%(asctime)s | %(levelname)s %(module)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        if type(constant_depth_mode) == int:
            self.depth_size = constant_depth_mode
        elif constant_depth_mode is False:
            self.depth_size = 0
        else:
            raise ValueError("Error in depth")

        self.connection = None
        self.db_cursor = None

        self.columns_naming = None
        self.columns_data_sizing = None
        # Check if storage folder for db exists
        if not os.path.exists(f"../dataStorage"):
            os.mkdir(f"../dataStorage")

        else:
            if clean_tables:
                if os.path.exists(f"../dataStorage/{self.cfg['hdf5_database_file']}"):
                    os.remove(f"../dataStorage/{self.cfg['hdf5_database_file']}")
                    os.remove(f"../dataStorage/pd_{self.cfg['hdf5_database_file']}")
                    time.sleep(0.5)
        try:
            self.path_to_hdf5_file = f"../dataStorage/{self.cfg['hdf5_database_file']}"
            self.path_to_hdf5_file_pd = f"../dataStorage/pd_{self.cfg['hdf5_database_file']}"
            # self.connection = db_system.File(f"../dataStorage/{cfg['hdf5_database_file']}", "w")

            self.connection = pd.HDFStore(self.path_to_hdf5_file_pd, mode='w')
            self.db_cursor = None
            logging.info("Success connection to HDF5 database")
        except Exception as e:
            logging.error("Connection to database raise error: \n {error}".format(error=e))
            raise ConnectionError("Cannot connect to HDF5 database")

        # Unlimited mode
        if self.depth_size == 0:
            # Validate that tables exists. Create if not.
            raise ValueError("Raw order book mode for HDF5 not available current now")
        # Limited mode
        else:
            self.check_if_tables_exists_limited_depth()

        # self.connection.close()

    def check_if_tables_exists_limited_depth(self):
        if not self.cfg["use_bathes_to_record"]:
            self.no_batch_check_if_tables_exists_limited_depth()
        else:
            self.batch_check_if_table_exists_limited_depth()

    def no_batch_check_if_tables_exists_limited_depth(self):
        _all_exist = True
        _table_name = self.TEMPLATE_FOR_LIMIT_DEPTH_TABLES_NAME.format(self.depth_size)
        if _table_name not in self.connection:
            logging.warning(f"{_table_name} NOT exist; Table will be creating...")
            _all_exist = False
            # _set = self.connection.create_dataset(_table_name, shape=(1, self.depth_size * 2 + 3))

        if _all_exist:
            logging.info("All need tables already exists. That's good!")

        self.columns_naming = ['CHANGE_ID', 'NAME_INSTRUMENT', 'TIMESTAMP_VALUE']
        self.columns_data_sizing = {"CHANGE_ID": self.LIMITS_OF_COLUMNS["CHANGE_ID"],
                                    "NAME_INSTRUMENT": self.LIMITS_OF_COLUMNS["NAME_INSTRUMENT"],
                                    "TIMESTAMP_VALUE": self.LIMITS_OF_COLUMNS["TIMESTAMP_VALUE"]}

        for _pointer in range(self.depth_size):
            self.columns_naming.extend([f"BID_{_pointer}_PRICE", f"BID_{_pointer}_AMOUNT"])
            self.columns_data_sizing[f"BID_{_pointer}_PRICE"] = self.LIMITS_OF_COLUMNS["BID_PRICE"]
            self.columns_data_sizing[f"BID_{_pointer}_AMOUNT"] = self.LIMITS_OF_COLUMNS["BID_AMOUNT"]

        for _pointer in range(self.depth_size):
            self.columns_naming.extend([f"ASK_{_pointer}_PRICE", f"ASK_{_pointer}_AMOUNT"])
            self.columns_data_sizing[f"ASK_{_pointer}_PRICE"] = self.LIMITS_OF_COLUMNS["ASK_PRICE"]
            self.columns_data_sizing[f"ASK_{_pointer}_AMOUNT"] = self.LIMITS_OF_COLUMNS["ASK_AMOUNT"]

    def add_order_book_content_limited_depth(self, bids, asks, change_id, timestamp, instrument_name):
        if not self.cfg["use_bathes_to_record"]:
            self.no_batch_add_order_book_content_limited_depth(bids, asks, change_id, timestamp, instrument_name)
        else:
            self.batch_add_order_book_content_limited_depth(bids, asks, change_id, timestamp, instrument_name)

    def no_batch_add_order_book_content_limited_depth(self, bids, asks, change_id, timestamp, instrument_name):
        _table_name = self.TEMPLATE_FOR_LIMIT_DEPTH_TABLES_NAME.format(self.depth_size)

        bids = sorted(bids, key=lambda x: x[0], reverse=True)
        asks = sorted(asks, key=lambda x: x[0], reverse=False)

        bids_insert_array = [[-1.0, -1.0] for _i in range(self.depth_size)]
        asks_insert_array = [[-1.0, -1.0] for _i in range(self.depth_size)]
        _pointer = self.depth_size - 1
        for i, bid in enumerate(bids):
            bids_insert_array[_pointer] = bid
            _pointer -= 1

        _pointer = self.depth_size - 1
        for i, ask in enumerate(asks):
            asks_insert_array[i] = ask
            _pointer -= 1

        data = [change_id, instrument_name, timestamp]
        for _pointer in range(self.depth_size):
            data.extend([bids_insert_array[_pointer][0], bids_insert_array[_pointer][1]])

        for _pointer in range(self.depth_size):
            data.extend([asks_insert_array[_pointer][0], asks_insert_array[_pointer][1]])

        line_write = pd.Series(data=data, index=self.columns_naming, dtype=str).to_frame().T
        self.connection.append(_table_name, line_write, data_columns=self.columns_naming, index=False,
                               min_itemsize=self.columns_data_sizing)

    def add_instrument_change_order_book_unlimited_depth(self, request_change_id: int, request_previous_change_id: int,
                                                         change_timestamp: int,
                                                         bids_list: list[list[str, float, float]],
                                                         asks_list: list[list[str, float, float]]):
        raise ValueError("Raw order book mode for HDF5 not available current now")

    def add_instrument_init_snapshot(self, instrument_name: str, start_instrument_scrap_time: int,
                                     request_change_id: int, bids_list, asks_list: list[list[str, float, float]]):
        raise ValueError("Raw order book mode for HDF5 not available current now")


    def batch_add_order_book_content_limited_depth(self, bids, asks, change_id, timestamp, instrument_name):

        # Refresh tables when filled
        self.batch_mutable_pointer += 1
        print(f'Table pointer = {self.batch_mutable_pointer}')
        if self.batch_mutable_pointer > self.cfg["batch_size"]:
            self.batch_mutable_pointer = 0
            self.batch_currently_selected_table += 1
            logging.info(f'TMP table has been filled. Setting pointer to zero. Start transfer data to db ({self.batch_currently_selected_table})')
            if self.batch_currently_selected_table == len(self.batch_mode_tables_storage):
                self.batch_currently_selected_table = 0

            # Send request to database
        else:
            # Add new line to tmp table
            pass



if __name__ == "__main__":
    # Testing
    hdf5Daemon = HDF5Daemon(2, True)
    # hdf5Daemon.add_order_book_content_limited_depth(bids=[[0.1, 0.11], [0.2, 0.22]],
    #                                                 asks=[[0.3, 0.33], [0.4, 0.44]],
    #                                                 change_id=12312,
    #                                                 timestamp="231231213",
    #                                                 instrument_name="BTC-p")
    #
    # hdf5Daemon.add_order_book_content_limited_depth(bids=[[0.11231, 0.1123123], [0.2, 0.22]],
    #                                                 asks=[[0.32312, 0.321313], [0.4, 0.44]],
    #                                                 change_id=12312321,
    #                                                 timestamp="231231213",
    #                                                 instrument_name="BTC-p")
    #
    # hdf5Daemon.add_order_book_content_limited_depth(bids=[[0.11123231, 0.1123123], [0.2, 0.22]],
    #                                                 asks=[[0.3212312312, 0.321313], [0.4, 0.44]],
    #                                                 change_id=1232131239423412312321,
    #                                                 timestamp="231231213",
    #                                                 instrument_name="BTC-p")
    file = hdf5Daemon.connection
    file.close()
