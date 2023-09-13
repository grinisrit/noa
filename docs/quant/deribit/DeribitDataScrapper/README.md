# TODO: make readme

## Installation

Python 3.10 or later. To install run:

```bash
pip install scrapperDeribit
```

## Configuration explained
Each run of DeribitClient will take configuration file as input.

### mysql:

host - mySQL host \
user: mySQL user with write access \
password: mySQL password \
database: mySQL db name \
use_bathes_to_record: unused, will be deprecated soon \
reconnect_max_attempts: number of reconnect try on DB errors \
reconnect_wait_time: time in sec before new reconnect try

### hdf5:
hdf5_database_directory: string with hdf5 storage path

### record_system:
use_batches_to_record: True or False. Unable batch system \
number_of_tmp_tables: number of circular batch tables \
size_of_tmp_batch_table: number of lines in one table \
instrumentNameToIdMapFile: unused, will be deprecated soon \

clean_database_at_startup: True or False CleanUp on start

### user_data:
#### test_net:
client_id: deribit client id

client_secret: deribit client secret
#### production:
client_id: None

client_secret: None

### externalModules:
add_order_manager: True or False \
add_instrument_manager: True or False


### orderBookScrapper:
scrapper_body: right now [OrderBook, Trades, OwnOrderChange, Portfolio] available. You can use only some of them \
depth: order book depth/ Highly recommend to use Union[1, 10, 100] \
test_net: True or False \
currency: BTC or ETH or SOL \
enable_traceback: False # False - default \
enable_database_record: True or False. Enable database record? \
clean_database: False unused, will be deprecated soon \
hearth_beat_time: default 60. connection ping-pong time \ 
group_in_limited_order_book: none (better don't touch) \
raise_error_at_synthetic: True or False. Should scrapper raise error when underlying for maturity is synthetic? \
logger_level: WARN | INFO | ERROR. Logger level. INFO can broke buffer when full surface collecting \
select_all_order_book: True \
only_api_orders_processing: True \
database_daemon: mysql | hdf5

add_extra_instruments: example ['BTC-5MAY23-28000-C', 'BTC-5MAY23-28000-P', 'BTC-PERPETUAL']. List of instruments that should be collected \
use_configuration_to_select_maturities: False used in several scripts with pre-selected configuration about maturities. Will be deprecated soon \
maturities_configuration_path: "Maturities_configuration.yaml" Will be deprecated soon


## Documentation
TODO: add documentation coverage badge

## Testing
TODO: add testing coverage badge

TODO: link to repo with ZeroMQ test server 

## Example (scrapping data)
TODO: add

## Example (make and connect own subscription)
TODO: add

## Example (make and connect own strategy)
TODO: add

## PR Pipeline
TODO: add