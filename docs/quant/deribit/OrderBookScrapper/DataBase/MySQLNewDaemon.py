import asyncio
from datetime import datetime
from typing import Optional

from docs.quant.deribit.OrderBookScrapper.DataBase.AbstractDataSaverManager import AbstractDataManager
import logging
import mysql.connector as connector
from docs.quant.deribit.OrderBookScrapper.DataBase.mysqlRecording.cleanUpRequestsUnlimited import *
from docs.quant.deribit.OrderBookScrapper.DataBase.mysqlRecording.postDataTemplateLimited import *
from docs.quant.deribit.OrderBookScrapper.Subsciption.AbstractSubscription import AbstractSubscription

# Block with developing module | START
import yaml
import sys

with open(sys.path[1] + "/docs/quant/deribit/OrderBookScrapper/developerConfiguration.yaml", "r") as _file:
    developConfiguration = yaml.load(_file, Loader=yaml.FullLoader)
del _file
# Block with developing module | END


class MySqlDaemon(AbstractDataManager):
    """
    Daemon for MySQL record type.
    """
    connection: connector.connection.MySQLConnection
    database_cursor: connector.connection.MySQLCursor

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
        """
        Connection to MySQL database
        :return:
        """
        flag = 0
        while flag < self.cfg["mysql"]["reconnect_max_attempts"]:
            if flag >= self.cfg["mysql"]["reconnect_max_attempts"]:
                raise ConnectionError("Cannot connect to MySQL. Reached maximum attempts")

            try:
                self.connection = connector.connect(host=self.cfg["mysql"]["host"],
                                                    user=self.cfg["mysql"]["user"],
                                                    database=self.cfg["mysql"]["database"])
                self.database_cursor = self.connection.cursor()
                logging.info("Success connection to MySQL database")
                return 1
            except connector.Error as e:
                flag += 1
                logging.error("Connection to database raise error: \n {error}".format(error=e))
                await asyncio.sleep(self.cfg["mysql"]["reconnect_wait_time"])

    async def _mysql_post_execution_handler(self, query, need_to_commit: bool = False) -> int:
        """
        Interface to execute POST request to MySQL database
        :param query:
        :return:
        """
        if developConfiguration["MY_SQL_DAEMON"]["SHOW_QUERY_FOR_POST"]:
            print(f"POST MYSQL REQUEST: QUERY | {query} | TIME {datetime.now()}")
        flag = 0
        while flag < self.cfg["mysql"]["reconnect_max_attempts"]:
            if flag >= self.cfg["mysql"]["reconnect_max_attempts"]:
                raise ConnectionError("Cannot execute MySQL query. Reached maximum attempts")
            try:
                self.database_cursor.execute(query)
                if need_to_commit:
                    self.connection.commit()
                return 1
            except connector.Error as e:
                flag += 1
                logging.error("MySQL execution error: \n {error}".format(error=e))
                await asyncio.sleep(self.cfg["mysql"]["reconnect_wait_time"])

    # TODO: typing
    async def _mysql_get_execution_handler(self, query) -> object:
        """
        Interface to execute GET request to MySQL database
        :param query:
        :return:
        """
        if developConfiguration["MY_SQL_DAEMON"]["SHOW_QUERY_FOR_GET"]:
            print(f"GET MYSQL REQUEST: QUERY | {query} | TIME {datetime.now()}")
        flag = 0
        while flag < self.cfg["mysql"]["reconnect_max_attempts"]:
            if flag >= self.cfg["mysql"]["reconnect_max_attempts"]:
                raise ConnectionError("Cannot execute MySQL query. Reached maximum attempts")
            try:
                self.database_cursor.execute(query)
                return self.database_cursor.fetchone()
            except connector.Error as e:
                flag += 1
                logging.error("MySQL execution error: \n {error}".format(error=e))
                await asyncio.sleep(self.cfg["mysql"]["reconnect_wait_time"])

    async def _clean_exist_database(self):
        """
        Clean MySQL database body method
        :return:
        """
        flag = 0
        while flag < self.cfg["mysql"]["reconnect_max_attempts"]:
            if flag >= self.cfg["mysql"]["reconnect_max_attempts"]:
                raise ConnectionError("Cannot connect to MySQL. Reached maximum attempts")
            try:
                await self.__clean_up_pipeline()
                return 0

            except connector.Error as error:
                flag += 1
                logging.warning("Database clean up error! :{}".format(error))
                await asyncio.sleep(self.cfg["mysql"]["reconnect_wait_time"])

    async def __clean_up_pipeline(self):
        """
        Query to cleanUP mySQL.
        :return:
        """
        if self.depth_size == 0:
            for table_name in self.subscription_type.tables_names:
                _truncate_query = """TRUNCATE table {}""".format(table_name)
                await self._mysql_post_execution_handler(_truncate_query)
                del _truncate_query
        # Limited mode
        else:
            for table_name in self.subscription_type.tables_names:
                _truncate_query = """TRUNCATE table {}""".format(table_name)
                await self._mysql_post_execution_handler(_truncate_query)
                del _truncate_query

    async def _create_not_exist_database(self):
        """
        Check if all need tables are exiting. If not creates them.
        :return:
        """
        # TODO: CLEAN UP THIS in format of subscriptionType;
        if self.depth_size == 0:
            raise NotImplementedError
            _all_exist = True
            _query = """SHOW TABLES LIKE '{}'"""
            # script_snapshot_id
            for table_name in self.subscription_type.tables_names:
                result = await self._mysql_get_execution_handler(_query.format(table_name))
                if not result:
                    logging.warning(f"{table_name} table NOT exist; Start creating...")
                    await self._mysql_post_execution_handler(REQUEST_TO_CREATE_SCRIPT_SNAPSHOT_ID)
                    _all_exist = False

            if _all_exist:
                logging.info("All need tables already exists. That's good!")

        # Limited mode
        elif (type(self.depth_size) == int) and (self.depth_size > 0):
            _all_exist = True
            _query = """SHOW TABLES LIKE '{}'"""
            for table_name, table_creation in zip(self.subscription_type.tables_names,
                                                  self.subscription_type.tables_names_creation):
                result = await self._mysql_get_execution_handler(_query.format(table_name))
                if not result:
                    logging.warning(f"Table {table_name} NOT exist; Start creating...")
                    await self._mysql_post_execution_handler(
                        table_creation)
                    _all_exist = False

            if _all_exist:
                logging.info("All need tables already exists. That's good!")

        else:
            raise NotImplementedError

    # TODO: make it better
    async def __database_one_table_record(self, record_dataframe: DataFrame):
        data = self.subscription_type.record_to_database(record_dataframe=record_dataframe, tag_of_data="LIMITED")
        query = INSERT_MULTIPLE_DATA_HEADER_TEMPLATE.format(self.subscription_type.tables_names[0])
        # -1 for delete last coma
        query += INSERT_MULTIPLE_DATA_VALUES_SYMBOL_TEMPLATE(dataframe=data)
        # await asyncio.gather(self._mysql_post_execution_handler(query=query, need_to_commit=True))
        await self._mysql_post_execution_handler(query=query, need_to_commit=True)

    def __database_several_tables_record(self, record_dataframe: DataFrame):
        # TODO: implement
        # self.subscription_type.record_to_database(record_dataframe=record_dataframe, tag_of_data="UNLIMITED")
        raise NotImplementedError

    async def _place_data_to_database(self, record_dataframe: DataFrame):
        if self.depth_size == 0:
            await self.__database_several_tables_record(record_dataframe=record_dataframe)
        elif (type(self.depth_size) == int) and (self.depth_size > 0):
            await self.__database_one_table_record(record_dataframe=record_dataframe)
