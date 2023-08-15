import asyncio
from pandas import DataFrame, HDFStore

from docs.quant.deribit.TradingInterfaceBotJulyVersion.DataBase.AbstractDataSaverManager import AbstractDataManager
from docs.quant.deribit.TradingInterfaceBotJulyVersion.Subsciption.AbstractSubscription import AbstractSubscription

from typing import Optional
import logging
import os


class HDF5Daemon(AbstractDataManager):
    """
    Daemon for HDF5 record type.
    TODO: insert docstring
    """
    connection: HDFStore = None
    database_cursor = None

    def __init__(self, configuration_path, subscription_type: Optional[AbstractSubscription],
                 loop: asyncio.unix_events.SelectorEventLoop):
        logging.basicConfig(
            level='INFO',
            format='%(asctime)s | %(levelname)s %(module)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        super().__init__(config_path=configuration_path, subscription_type=subscription_type,
                         loop=loop)

    async def _connect_to_database(self):
        print("Connect HDF5")
        try:
            self.path_to_hdf5_file = \
                f"../{self.cfg['hdf5']['hdf5_database_directory']}/{self.subscription_type.__class__.__name__}_{self.subscription_type.tables_names[0]}.h5"
            if not os.path.exists(f"../{self.cfg['hdf5']['hdf5_database_directory']}/"):
                os.mkdir(f"../{self.cfg['hdf5']['hdf5_database_directory']}/")
                logging.warning("Create folder for storage")
            if not os.path.exists(self.path_to_hdf5_file):
                logging.warning("Create HDF5 File")
                self.connection = HDFStore(self.path_to_hdf5_file, mode='w')
                self.connection.close()

            self.connection = HDFStore(self.path_to_hdf5_file, mode='r+')
            self.connection.close()
            self.db_cursor = None
            logging.info("Success connection to HDF5 database")
            return
        except Exception as e:
            logging.error("Connection to database raise error: \n {error}".format(error=e))
            raise ConnectionError("Cannot connect to HDF5 database")

    async def _clean_exist_database(self):
        if os.path.exists(self.path_to_hdf5_file):
            logging.warning("CleanUP HDF5 file")
            os.remove(self.path_to_hdf5_file)

            self.connection = HDFStore(self.path_to_hdf5_file, mode='w')
            self.connection.close()
        return
        # if os.path.exists(f"../dataStorage/{self.cfg['hdf5']['hdf5_database_file']}"):
        #     os.remove(f"../dataStorage/{self.cfg['hdf5']['hdf5_database_file']}")
        #     time.sleep(0.5)

    async def _create_not_exist_database(self):
        pass

    async def __hdf5_appending_one_table(self, record_dataframe: DataFrame):
        record_dataframe.to_hdf(self.connection, key=self.subscription_type.tables_names[0],
                                format='t', append=True, index=False,
                                data_columns=True)
        # self.connection.append(self.subscription_type.tables_names[0], record_dataframe,
        #                        data_columns=self.subscription_type.create_columns_list(), format='t')
        return 1

    async def _place_data_to_database(self, record_dataframe: DataFrame) -> int:
        await self.__hdf5_appending_one_table(record_dataframe=record_dataframe)
        return 1
