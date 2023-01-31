import asyncio
from abc import ABC, abstractmethod
from numpy import ndarray
from typing import Dict, Optional
from pandas import DataFrame
import os
import logging
import json
import numpy as np

from docs.quant.deribit.TradingInterfaceBot.Subsciption.AbstractSubscription import AbstractSubscription

# Block with developing module | START
import yaml
import sys

with open(sys.path[1] + "/docs/quant/deribit/TradingInterfaceBot/developerConfiguration.yaml", "r") as _file:
    developConfiguration = yaml.load(_file, Loader=yaml.FullLoader)
del _file
# Block with developing module | END


class AutoIncrementDict(dict):
    pointer = -1

    def __init__(self, path_to_file):
        super().__init__()
        self.path_to_file = "/".join(__file__.split('/')[:-1]) + "/" + path_to_file
        if not os.path.exists(self.path_to_file):
            logging.info("No cached instruments map exist")
            self.add_instrument(key="EMPTY-INSTRUMENT")

        else:
            logging.info("Cache instruments map exist")
            self.download_cache_from_file(path_to_file=self.path_to_file)

    def download_cache_from_file(self, path_to_file: str):
        # Load existed instrument map
        with open(path_to_file, "r") as _file:
            instrument_name_instrument_id_map = json.load(_file)

        for objects in instrument_name_instrument_id_map.items():
            self.add_instrument(key=objects[0], value=objects[1])

        self.pointer = max(instrument_name_instrument_id_map.values())

        logging.info(f"Dict map has last pointer equals to {self.pointer}")

    def add_instrument(self, key, value=None):
        if not value:
            self.pointer += 1
            super().__setitem__(key, self.pointer)
        else:
            super().__setitem__(key, value)

    def _save_after_adding(self):
        with open(f'{self.path_to_file}', 'w') as fp:
            json.dump(self, fp)
        logging.info("Saved new instrument to map")

    def __getitem__(self, item):
        if item not in self:
            self.add_instrument(item)
            self._save_after_adding()
        return super().__getitem__(item)


class AbstractDataManager(ABC):
    # instrument_name_instrument_id_map: AutoIncrementDict[str, int] = None
    circular_batch_tables: Dict[int, DataFrame]

    batch_mutable_pointer: Optional[int] = None
    batch_number_of_tables: Optional[int] = None
    batch_size_of_table: Optional[int] = None
    batch_currently_selected_table: Optional[int] = None
    async_loop: asyncio.unix_events.SelectorEventLoop

    subscription_type: Optional[AbstractSubscription] = None

    def __init__(self, config_path, subscription_type: Optional[AbstractSubscription],
                 loop: asyncio.unix_events.SelectorEventLoop):
        # Config file
        with open(config_path, "r") as ymlfile:
            self.cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.subscription_type = subscription_type

        # Download instrument hashMap
        # self.instrument_name_instrument_id_map = AutoIncrementDict(path_to_file=
        #                                                            self.cfg["record_system"][
        #                                                                "instrumentNameToIdMapFile"])

        # Check if all structure and content of record system is correct
        self.async_loop = loop

        asyncio.run_coroutine_threadsafe(self._connect_to_database(), self.async_loop)
        asyncio.run_coroutine_threadsafe(self._validate_existing_of_database_structure(), self.async_loop)
        # Create tmp_storages
        self._create_tmp_batch_tables()

    async def _validate_existing_of_database_structure(self):
        """
        Validate correct structure of record system. Check if all need tables are exist.
        :return:
        """
        # Check if storages exists
        if self.cfg["orderBookScrapper"]["enable_database_record"]:
            await self._create_not_exist_database()
        else:
            logging.warning("Selected no record system")

        if self.cfg["record_system"]["clean_database_at_startup"]:
            await self._clean_exist_database()

        return None

    @abstractmethod
    async def _connect_to_database(self):
        pass

    @abstractmethod
    async def _clean_exist_database(self):
        pass

    @abstractmethod
    async def _create_not_exist_database(self):
        pass

    async def add_data(self, update_line: ndarray):
        # TODO: Make ability to record 2D array

        # Hardcoded solution. in case 1D array
        if len(update_line.shape) == 1:
            assert update_line.shape[0] == len(self.subscription_type.create_columns_list())
            self.circular_batch_tables[self.batch_currently_selected_table].iloc[self.batch_mutable_pointer] = update_line
            self.batch_mutable_pointer += 1
            if self.batch_mutable_pointer >= self.batch_size_of_table:
                if developConfiguration["DATA_MANAGER"]["SHOW_WHEN_DATA_TRANSFERS"]:
                    print("Transfer data:\n", self.circular_batch_tables[self.batch_currently_selected_table], "\n")
                    print(f"Pointer In Table: ({self.batch_mutable_pointer}) | Pointer Out Table: ({self.batch_currently_selected_table})")
                    print("=====" * 20)

                self.batch_mutable_pointer = 0
                await self._place_data_to_database(record_dataframe=
                                             self.circular_batch_tables[self.batch_currently_selected_table])
                self.batch_currently_selected_table += 1
                if self.batch_currently_selected_table >= self.batch_number_of_tables:
                    self.batch_currently_selected_table = 0

        # Hardcoded solution. in case 2D array
        if len(update_line.shape) == 2:
            assert update_line.shape[1] == len(self.subscription_type.create_columns_list())
            for update_object in update_line:
                self.circular_batch_tables[self.batch_currently_selected_table].iloc[
                    self.batch_mutable_pointer] = update_object
                self.batch_mutable_pointer += 1
                if self.batch_mutable_pointer >= self.batch_size_of_table:
                    if developConfiguration["DATA_MANAGER"]["SHOW_WHEN_DATA_TRANSFERS"]:
                        print("Transfer data:\n", self.circular_batch_tables[self.batch_currently_selected_table], "\n")
                        print(
                            f"Pointer In Table: ({self.batch_mutable_pointer}) | Pointer Out Table: ({self.batch_currently_selected_table})")
                        print("=====" * 20)

                    self.batch_mutable_pointer = 0
                    await self._place_data_to_database(record_dataframe=
                                                       self.circular_batch_tables[self.batch_currently_selected_table])
                    self.batch_currently_selected_table += 1
                    if self.batch_currently_selected_table >= self.batch_number_of_tables:
                        self.batch_currently_selected_table = 0

    def _create_tmp_batch_tables(self):
        """
        Creates tmp batches for batch record system.
        :return:
        """
        columns = self.subscription_type.create_columns_list()
        self.batch_mutable_pointer = 0
        self.batch_currently_selected_table = 0

        # Create columns for tmp tables
        if self.cfg["record_system"]["use_batches_to_record"]:
            self.batch_number_of_tables = self.cfg["record_system"]["number_of_tmp_tables"]
            self.batch_size_of_table = self.cfg["record_system"]["size_of_tmp_batch_table"]

            # TODO: remove depth size. Change to number of columns
            _local = np.zeros(shape=(self.cfg["record_system"]["size_of_tmp_batch_table"],
                                     self.subscription_type.number_of_columns))
            _local[:] = np.NaN
            # Create tmp tables
            self.circular_batch_tables = {_: DataFrame(_local, columns=columns)
                                          for _ in range(self.cfg["record_system"]["number_of_tmp_tables"])}

            assert len(self.circular_batch_tables) == self.cfg["record_system"]["number_of_tmp_tables"]

            del _local, columns
            logging.info(f"""
            TMP tables for batching has been created. Number of tables = ({len(self.circular_batch_tables)}),
            Size of one table is ({self.circular_batch_tables[0].shape})  
            """)
        # No batch system enabled
        else:
            self.batch_number_of_tables = 1
            self.batch_size_of_table = 1
            # TODO: remove depth size. Change to number of columns
            # _local = np.zeros(shape=(1, self.depth_size * 4 + 3))
            _local = np.zeros(shape=(1, self.subscription_type.number_of_columns))
            _local[:] = np.NaN
            # Create tmp tables
            self.circular_batch_tables = {0: DataFrame(_local, columns=columns)}

            del _local, columns
            logging.info(f"""
            NO BATCH MODE: TMP tables for batching has been created. Number of tables = ({len(self.circular_batch_tables)}),
            Size of one table is ({self.circular_batch_tables[0].shape})  
            """)

    @abstractmethod
    async def _place_data_to_database(self, record_dataframe: DataFrame) -> int:
        """
        Implement this for every unique method for placing data
        :param record_dataframe:
        :return:
        """
        pass

