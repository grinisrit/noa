from AbstractDataSaverManager import AbstractDataManager
import logging
import yaml
import h5py as db_system
import os
import shutil


class HDF5Daemon(AbstractDataManager):
    def __init__(self, constant_depth_mode: bool | int, clean_tables: bool = False):
        # Config file
        with open("../configuration.yaml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)['mysql']

        super().__init__()
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
        # Check if storage folder for db exists
        if not os.path.exists(f"../dataStorage"):
            os.mkdir(f"../dataStorage")

        else:
            if clean_tables:
                os.remove(f"../dataStorage/{cfg['hdf5_database_file']}")

        try:
            # self.connection = db_system.File()
            # self.db_cursor = self.connection.cursor()
            logging.info("Success connection to MySQL database")
        except Exception as e:
            logging.error("Connection to database raise error: \n {error}".format(error=e))
            raise ConnectionError("Cannot connect to HDF5 database")

    def add_order_book_content_limited_depth(self, bids, asks, change_id, timestamp, instrument_name):
        pass

    def add_instrument_change_order_book_unlimited_depth(self, request_change_id: int, request_previous_change_id: int,
                                                         change_timestamp: int,
                                                         bids_list: list[list[str, float, float]],
                                                         asks_list: list[list[str, float, float]]):
        pass

    def add_instrument_init_snapshot(self, instrument_name: str, start_instrument_scrap_time: int,
                                     request_change_id: int, bids_list, asks_list: list[list[str, float, float]]):
        pass

if __name__ == "__main__":
    hdf5Daemon = HDF5Daemon()