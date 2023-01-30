Данный модуль позволяет подписаться на изменения в order book для большого количества инструментов.

Для запуска необходимо запустить скрипт Scrappers.DeribitClient.py. 
В конкретной реализации будет выполнена подписка на всевозможные опционы на выбранную пользователем дату maturity.

Поля файла конфигурации:

depth: Глубина собираемого ордербука. False - сырой ордербук. Int значение - заданная глубина.

test_net: True => test.deribit.com, False => deribit.com

currency: BTC or ETH

enable_traceback: False. Используется для дебага.

enable_database_record: True/False. Запись собираемого ордер бука в базу данных.

clean_database: Очищать ли базу данных перед началом сбора данных.

hearth_beat_time: int value. Частота с которой сервер deribit проверяет соединение.

group_in_limited_order_book: см. документацию Deribit

raise_error_at_synthetic: Синтетические фьючерсы не торгуются. В случае значения True выкинет ошибку при попытке добавить такой фьючерс в список подписок.

logger_level: INFO, WARN, ERROR. Уровень логгера.

database_daemon: mysql or hdf5. Способ записи



add_extra_instruments: Используется для добавления подписки на дополнительные инструменты.



