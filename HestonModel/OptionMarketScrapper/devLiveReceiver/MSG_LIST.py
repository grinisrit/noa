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
