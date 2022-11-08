import mysql.connector as connector
import logging

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

class MySqlDaemon:
    def __init__(self, clean_tables=False):
        logging.basicConfig(
            level='INFO',
            format='%(asctime)s | %(levelname)s %(module)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        self.connection = None
        self.db_cursor = None
        try:
            self.connection = connector.connect(host="localhost", user="root", database="DeribitOrderBook")
            self.db_cursor = self.connection.cursor()
            logging.info("Success connection to MySQL database")
        except connector.Error as e:
            logging.error("Connection to database raise error: \n {error}".format(error=e))
            raise ConnectionError("Cannot connect to MySQL")
        # Validate that tables exists. Create if not.
        self.check_if_tables_exists()

        if clean_tables:
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
    def check_if_tables_exists(self):
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

    def insert_to_script_snapshot(self, instrument, start_time, change_id):
        insert = (instrument, start_time, change_id)
        with self.connection.cursor() as cursor:
            cursor.execute(INSERT_TO_SNAPSHOT_QUERY, insert)
            self.connection.commit()

    def create_new_old_pair(self, change_id, old_change_id, update_time):
        insert = (change_id, old_change_id, update_time)
        with self.connection.cursor() as cursor:
            cursor.execute(INSERT_TO_NEW_OLD_PAIR_QUERY, insert)
            self.connection.commit()

    def add_order_book_content(self, bids, asks, change_id):

        # TODO: Can remove this request by creating HashMap inside Daemon. (Only one request when initialized)
        with self.connection.cursor() as cursor:
            cursor.execute(FIND_PRIMARY_KEY_BY_CURRENT_CHANGE_ID, [change_id])
            last_connection = cursor.fetchone()[0]

        bids_values = [(last_connection, bid[0].upper(), bid[1], bid[2], "BID") for bid in bids]
        asks_values = [(last_connection, ask[0].upper(), ask[1], ask[2], "ASK") for ask in asks]

        print(bids_values)
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

        self.insert_to_script_snapshot(instrument=instrument_name,
                                       start_time=start_instrument_scrap_time,
                                       change_id=request_change_id)
        self.create_new_old_pair(change_id=request_change_id, old_change_id=-1, update_time=start_instrument_scrap_time)
        self.add_order_book_content(bids=bids_list, asks=asks_list, change_id=request_change_id)

    def add_instrument_change_order_book(self, request_change_id: int, request_previous_change_id: int,
                                         change_timestamp: int,
                                         bids_list: list[list[str, float, float]],
                                         asks_list: list[list[str, float, float]]
                                         ):
        self.create_new_old_pair(change_id=request_change_id, old_change_id=request_previous_change_id,
                                 update_time=change_timestamp)

        self.add_order_book_content(bids=bids_list, asks=asks_list, change_id=request_change_id)


if __name__ == "__main__":
    daemon = MySqlDaemon(clean_tables=True)

