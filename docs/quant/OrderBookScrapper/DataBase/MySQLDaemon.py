import mysql.connector as connector
from docs.quant.OrderBookScrapper.DataBase.AbstractDataSaverManager import AbstractDataManager
import logging
import yaml

REQUEST_TO_CREATE_SCRIPT_SNAPSHOT_ID = """
create table script_snapshot_id
(
    PRIMARY_KEY bigint auto_increment
        primary key,
    INSTRUMENT  blob   not null,
    START_TIME  bigint not null,
    CHANGE_ID   bigint not null,
    constraint CHANGE_ID
        unique (PRIMARY_KEY),
    constraint CHANGE_ID_2
        unique (CHANGE_ID)
)
    comment 'Таблица связывающая change_id и время когда он пришел';


"""

REQUEST_TO_CREATE_PAIRS_NEW_OLD = """
create table pairs_new_old
(
    PRIMARY_KEY    int auto_increment
        primary key,
    CHANGE_ID      bigint not null,
    PREV_CHANGE_ID bigint not null,
    UPDATE_TIME    bigint not null,
    constraint CHANGE_ID
        unique (CHANGE_ID),
    constraint PREV_CHANGE_ID
        unique (PREV_CHANGE_ID),
    constraint PRIMARY_KEY
        unique (PRIMARY_KEY)
)
    comment 'Хранилище для инициализации запуска скрипта.
Создает уникальный ID для пар (Инструмент, Время запуска скрипта)';


"""

REQUEST_TO_CREATE_ORDER_BOOK_CONTENT = """
create table order_book_content
(
    CONNECT_TO_LAST bigint                           not null
        primary key,
    OPERATION       enum ('NEW', 'CHANGE', 'DELETE') not null,
    PRICE           float                            not null,
    AMOUNT          float                            not null,
    WAY             enum ('BID', 'ASK')              not null,
    constraint ID_PK
        unique (CONNECT_TO_LAST)
)
    comment 'Таблица содержащая ордербук на момент запуска скрипта';


"""

INSERT_TO_SNAPSHOT_QUERY = """INSERT INTO script_snapshot_id (INSTRUMENT, START_TIME, CHANGE_ID) VALUES (%s, %s, %s)"""

INSERT_TO_NEW_OLD_PAIR_QUERY = """INSERT INTO pairs_new_old (CHANGE_ID, PREV_CHANGE_ID, UPDATE_TIME) VALUES (%s, %s, %s)"""

INSERT_CONTENT = """INSERT INTO order_book_content (CONNECT_TO_LAST, OPERATION, PRICE, AMOUNT, WAY) VALUES (%s, %s, %s, %s, %s)"""

FIND_PRIMARY_KEY_BY_CURRENT_CHANGE_ID = """
SELECT PRIMARY_KEY FROM pairs_new_old
WHERE CHANGE_ID = %s
"""

def REQUEST_TO_CREATE_LIMITED_ORDER_BOOK_CONTENT(table_name:str, depth_size: int):
    HEADER = "create table {}".format(table_name)
    REQUIRED_FIELDS = """(
    CHANGE_ID bigint not null,
    NAME_INSTRUMENT blob                           not null,
    TIMESTAMP_VALUE bigint                           not null,
    """
    ADDITIONAL_FIELDS_BIDS = """
    BID_{}_PRICE float not null,
    BID_{}_AMOUNT float not null, 
    """

    ADDITIONAL_FIELDS_ASKS = """
    ASK_{}_PRICE float not null,
    ASK_{}_AMOUNT float not null, 
    """

    LOWER_HEADER = """
    PRIMARY KEY(CHANGE_ID)
    )
    comment 'Таблица содержащая ордербук на момент запуска скрипта'
    """

    REQUEST = HEADER + REQUIRED_FIELDS
    for pointer in range(depth_size):
        REQUEST += ADDITIONAL_FIELDS_BIDS.format(pointer, pointer)

    for pointer in range(depth_size):
        REQUEST += ADDITIONAL_FIELDS_ASKS.format(pointer, pointer)

    REQUEST += LOWER_HEADER

    return REQUEST


HEADER_INSERTION_LIMITED_DEPTH = """INSERT INTO {table_name} (CHANGE_ID, NAME_INSTRUMENT, TIMESTAMP_VALUE, """


class MySqlDaemon(AbstractDataManager):
    TEMPLATE_FOR_LIMIT_DEPTH_TABLES_NAME = "TABLE_DEPTH_{}"

    def __init__(self,  constant_depth_mode: bool | int, clean_tables: bool = False):
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
        try:
            with open("../configuration.yaml", "r") as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)['mysql']
            self.connection = connector.connect(host=cfg["host"], user=cfg["user"], database=cfg["database"])
            self.db_cursor = self.connection.cursor()
            logging.info("Success connection to MySQL database")
        except connector.Error as e:
            logging.error("Connection to database raise error: \n {error}".format(error=e))
            raise ConnectionError("Cannot connect to MySQL")

        # Unlimited mode
        if self.depth_size == 0:
            # Validate that tables exists. Create if not.
            self.check_if_tables_exists_unlimited_depth()
        # Limited mode
        else:
            self.check_if_tables_exists_limited_depth()

        if clean_tables:
            # Unlimited mode
            if self.depth_size == 0:
                with self.connection.cursor() as cursor:
                    _truncate_query = """TRUNCATE table order_book_content"""
                    cursor.execute(_truncate_query)
                    _truncate_query = """TRUNCATE table pairs_new_old"""
                    cursor.execute(_truncate_query)
                    _truncate_query = """TRUNCATE table script_snapshot_id"""
                    cursor.execute(_truncate_query)
                    # cursor.execute(_truncate_query, 'pairs_new_old')
                    # cursor.execute(_truncate_query, 'script_snapshot_id')

                    del _truncate_query
            # Limited mode
            else:
                with self.connection.cursor() as cursor:
                    _truncate_query = f"""TRUNCATE table {self.TEMPLATE_FOR_LIMIT_DEPTH_TABLES_NAME.format(
                        self.depth_size)}"""
                    cursor.execute(_truncate_query)

                    del _truncate_query

    def check_if_tables_exists_limited_depth(self):
        _all_exist = True
        _table_name = self.TEMPLATE_FOR_LIMIT_DEPTH_TABLES_NAME.format(self.depth_size)
        _query = """SHOW TABLES LIKE '{}'"""
        # script_snapshot_id
        self.db_cursor.execute(_query.format(_table_name))
        result = self.db_cursor.fetchone()
        if not result:
            logging.warning("script_snapshot_id table NOT exist; Start creating...")
            self.db_cursor.execute(REQUEST_TO_CREATE_LIMITED_ORDER_BOOK_CONTENT(_table_name, self.depth_size))
            _all_exist = False

        del _table_name
        if _all_exist:
            logging.info("All need tables already exists. That's good!")

    def check_if_tables_exists_unlimited_depth(self):
        _all_exist = True
        _query = """SHOW TABLES LIKE '{}'"""
        # script_snapshot_id
        self.db_cursor.execute(_query.format("script_snapshot_id"))
        result = self.db_cursor.fetchone()
        if not result:
            logging.warning("script_snapshot_id table NOT exist; Start creating...")
            self.db_cursor.execute(REQUEST_TO_CREATE_SCRIPT_SNAPSHOT_ID)
            _all_exist = False

        # pairs_new_old
        self.db_cursor.execute(_query.format("pairs_new_old"))
        result = self.db_cursor.fetchone()
        if not result:
            logging.warning("pairs_new_old table NOT exist; Start creating...")
            self.db_cursor.execute(REQUEST_TO_CREATE_PAIRS_NEW_OLD)
            _all_exist = False

        # order_book_content
        self.db_cursor.execute(_query.format("order_book_content"))
        result = self.db_cursor.fetchone()
        if not result:
            logging.warning("order_book_content table NOT exist; Start creating...")
            self.db_cursor.execute(REQUEST_TO_CREATE_ORDER_BOOK_CONTENT)
            _all_exist = False

        if _all_exist:
            logging.info("All need tables already exists. That's good!")

    def insert_to_script_snapshot_unlimited_depth(self, instrument, start_time, change_id):
        insert = (instrument, start_time, change_id)
        with self.connection.cursor() as cursor:
            cursor.execute(INSERT_TO_SNAPSHOT_QUERY, insert)
            self.connection.commit()

    def create_new_old_pair_unlimited_depth(self, change_id, old_change_id, update_time):
        insert = (change_id, old_change_id, update_time)
        with self.connection.cursor() as cursor:
            cursor.execute(INSERT_TO_NEW_OLD_PAIR_QUERY, insert)
            self.connection.commit()

    def add_order_book_content_unlimited_depth(self, bids, asks, change_id):

        # TODO: Can remove this request by creating HashMap inside Daemon. (Only one request when initialized)
        with self.connection.cursor() as cursor:
            cursor.execute(FIND_PRIMARY_KEY_BY_CURRENT_CHANGE_ID, [change_id])
            last_connection = cursor.fetchone()[0]

        bids_values = [(last_connection, bid[0].upper(), bid[1], bid[2], "BID") for bid in bids]
        asks_values = [(last_connection, ask[0].upper(), ask[1], ask[2], "ASK") for ask in asks]

        with self.connection.cursor() as cursor:
            cursor.executemany(INSERT_CONTENT, bids_values)
            cursor.executemany(INSERT_CONTENT, asks_values)
            self.connection.commit()

    def add_instrument_init_snapshot(self, instrument_name: str,
                                     start_instrument_scrap_time: int,
                                     request_change_id: int,
                                     bids_list,
                                     asks_list: list[list[str, float, float]]
                                     ):

        self.insert_to_script_snapshot_unlimited_depth(instrument=instrument_name,
                                                       start_time=start_instrument_scrap_time,
                                                       change_id=request_change_id)
        self.create_new_old_pair_unlimited_depth(change_id=request_change_id, old_change_id=-1,
                                                 update_time=start_instrument_scrap_time)
        self.add_order_book_content_unlimited_depth(bids=bids_list, asks=asks_list, change_id=request_change_id)

    def add_instrument_change_order_book_unlimited_depth(self, request_change_id: int, request_previous_change_id: int,
                                                         change_timestamp: int,
                                                         bids_list: list[list[str, float, float]],
                                                         asks_list: list[list[str, float, float]]
                                                         ):
        self.create_new_old_pair_unlimited_depth(change_id=request_change_id, old_change_id=request_previous_change_id,
                                                 update_time=change_timestamp)

        self.add_order_book_content_unlimited_depth(bids=bids_list, asks=asks_list, change_id=request_change_id)

    def add_order_book_content_limited_depth(self, bids, asks, change_id, timestamp, instrument_name):
        _table_name = self.TEMPLATE_FOR_LIMIT_DEPTH_TABLES_NAME.format(self.depth_size)
        insert_header = HEADER_INSERTION_LIMITED_DEPTH.format(table_name=_table_name)

        bids = sorted(bids, key=lambda x: x[0], reverse=True)
        asks = sorted(asks, key=lambda x: x[0], reverse=False)

        bids_insert_array = [[-1.0, -1.0] for _i in range(self.depth_size)]
        asks_insert_array = [[-1.0, -1.0] for _i in range(self.depth_size)]
        _pointer = self.depth_size-1
        for i, bid in enumerate(bids):
            bids_insert_array[_pointer] = bid
            _pointer -= 1

        _pointer = self.depth_size-1
        for i, ask in enumerate(asks):
            asks_insert_array[i] = ask
            _pointer -= 1

        insert_meta = """"""
        for _pointer in range(self.depth_size):
            insert_meta += f"BID_{_pointer}_PRICE, BID_{_pointer}_AMOUNT, "
        for _pointer in range(self.depth_size):
            if _pointer == 0:
                insert_meta += f" ASK_{_pointer}_PRICE, ASK_{_pointer}_AMOUNT"
            else:
                insert_meta += f", ASK_{_pointer}_PRICE, ASK_{_pointer}_AMOUNT"

        insert_meta += ") "
        insert_body = """VALUES ({change_id}, "{instrument_name}", {timestamp_value}""".format(change_id=change_id,
                                                                                               instrument_name=instrument_name,
                                                                                               timestamp_value=timestamp)
        for _pointer in range(self.depth_size):
            insert_body += f", {bids_insert_array[_pointer][0]}, {bids_insert_array[_pointer][1]}"
            pass
        for _pointer in range(self.depth_size):
            insert_body += f", {asks_insert_array[_pointer][0]}, {asks_insert_array[_pointer][1]}"
            pass

        insert_request = insert_header + insert_meta + insert_body + ")"
        with self.connection.cursor() as cursor:
            cursor.execute(insert_request)
            self.connection.commit()

    def insert_snapshot_to_limit_depth_table(self, request_change_id: int,
                                             change_timestamp: int,
                                             bids_list: list[list[str, float, float]],
                                             asks_list: list[list[str, float, float]],
                                             instrument_name):
        self.add_order_book_content_limited_depth(bids=bids_list, asks=asks_list, change_id=request_change_id,
                                                  timestamp=change_timestamp, instrument_name=instrument_name)

if __name__ == "__main__":
    # print(HEADER_INSERTION_LIMITED_DEPTH.format(table_name="boba", change_id="@3",
    #                                             instrument_name="sad", timestamp_value=12312))
    daemon = MySqlDaemon(constant_depth_mode=5, clean_tables=True)

