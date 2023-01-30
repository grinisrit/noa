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