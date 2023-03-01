import warnings


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