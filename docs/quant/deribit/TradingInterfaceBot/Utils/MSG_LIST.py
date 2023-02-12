import warnings

import docs.quant.deribit.TradingInterfaceBot.Utils.AvailableCurrencies as AvailableCurrencies


def hello_message() -> dict:
    _msg = \
        {
            "jsonrpc": "2.0",
            "id": 2841,
            "method": "public/hello",
            "params": {
                "client_name": "Deribit OrderBook Scrapper",
                "client_version": "0.0.1"
            }
        }
    return _msg


def test_message() -> dict:
    _msg = \
        {
            "jsonrpc": "2.0",
            "id": 8212,
            "method": "public/test",
            "params": {

            }
        }
    return _msg


def set_heartbeat(interval=60) -> dict:
    _msg = \
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "public/set_heartbeat",
            "params": {
                "interval": interval
            }
        }
    return _msg


def make_subscription_all_book(instrument_name: str, type_of_data="book", interval="100ms") -> dict:
    channel = f"{type_of_data}.{instrument_name}.{interval}"
    _msg = \
        {
            "jsonrpc": "2.0",
            "method": f"public/subscribe",
            "id": 42,
            "params": {
                "channels": [channel]
            }
        }

    return _msg


def make_subscription_constant_book_depth(instrument_name: str, type_of_data="book",
                                          interval="100ms", depth=None, group=None) -> dict:
    if not depth:
        warnings.warn("You use constant depth request. Depth need to be passed")
        raise ValueError("No depth")
    else:
        if not group:
            channel = f"{type_of_data}.{instrument_name}.none.{depth}.{interval}"
        else:
            warnings.warn("Highly recommended not to use group. It can be unstable right now")
            channel = f"{type_of_data}.{instrument_name}.{group}.{depth}.{interval}"

    _msg = \
        {
            "jsonrpc": "2.0",
            "method": f"public/subscribe",
            "id": 42,
            "params": {
                "channels": [channel]
            }
        }

    return _msg


def unsubscribe_all() -> dict:
    _msg = \
        {
            "jsonrpc": "2.0",
            "id": 153,
            "method": "public/unsubscribe_all",
            "params": {

            }
        }
    return _msg


def make_trades_subscription_request_by_currency(currency: AvailableCurrencies.Currency,
                                                 kind="option",
                                                 interval="100ms"):
    print(currency.currency)
    _msg = \
        {"jsonrpc": "2.0",
         "method": "public/subscribe",
         "id": 42,
         "params": {
             "channels": [f"trades.{kind}.{currency.currency}.{interval}"]}
         }
    return _msg


def make_trades_subscription_request_by_instrument(instrument_name: str,
                                                   interval="100ms"):
    _msg = \
        {"jsonrpc": "2.0",
         "method": "public/subscribe",
         "id": 42,
         "params": {
             "channels": [f"trades.{instrument_name}.{interval}"]}
         }
    return _msg


def make_user_orders_subscription_request_by_instrument(instrument_name: str):
    _msg = \
        {"jsonrpc": "2.0",
         "method": "private/subscribe",
         "id": 42,
         "params": {
             "channels": [f"user.orders.{instrument_name}.raw"]}
         }
    return _msg

def auth_message(client_id: str, client_secret: str):
    _msg = \
        {
            "jsonrpc": "2.0",
            "id": 9929,
            "method": "public/auth",
            "params": {
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret
            }
        }
    return _msg


def order_request(order_side: str, instrument_name: str, amount: int,
                  order_type: str, order_tag: str = 'defaultTag', order_price=None):

    _msg = \
        {
            "jsonrpc": "2.0",
            "id": 5275,
            "method": f"private/{order_side}",
            "params": {
                "instrument_name": instrument_name,
                "amount": amount,
                "type": order_type,
                "label": order_tag
            }
        }
    if order_type == "limit":
        _msg["params"]["price"] = order_price

    return _msg


def cancel_order_request(order_id: int) -> dict:
    _msg = \
        {
            "jsonrpc": "2.0",
            "id": 4214,
            "method": "private/cancel",
            "params": {
                "order_id": f"{order_id}"
            }
        }
    return _msg


def get_ticker_by_instrument_request(instrument_request: str) -> dict:
    _msg = {
            "jsonrpc": "2.0",
            "id": 8106,
            "method": "public/ticker",
            "params": {
                "instrument_name": f"{instrument_request}"
            }
        }
    return _msg


def get_user_portfolio_request(currency: AvailableCurrencies.Currency) -> dict:
    _msg = \
        {"jsonrpc": "2.0",
         "method": "private/subscribe",
         "id": 42,
         "params": {
             "channels": [f"user.portfolio.{currency.currency.lower()}"]}
         }

    return _msg
