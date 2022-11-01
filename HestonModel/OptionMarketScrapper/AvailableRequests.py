import AvailableCurrencies
from AvailableInstrumentType import InstrumentType
import AvailableInstruments
import json


def construct_book_summary_by_instrument(instrument: AvailableInstruments.Instrument) -> json:
    _msg = {
        "jsonrpc": "2.0",
        "id": 3659,
        "method": "public/get_book_summary_by_instrument",
        "params": {
            "instrument_name": f"{instrument}"
        }
    }
    return _msg


def get_currencies_request() -> json:
    _msg = {
            "jsonrpc": "2.0",
            "id": 7538,
            "method": "public/get_currencies",
            "params": {

            }
        }
    return _msg


def get_instruments_by_currency_request(currency: AvailableCurrencies.Currency,
                                        kind: InstrumentType = InstrumentType.OPTION,
                                        expired=False) -> json:
    _msg = {
            "jsonrpc": "2.0",
            "id": 7617,
            "method": "public/get_instruments",
            "params": {
                "currency": f"{currency.currency}",
                "kind": f"{kind.instrument_type}",
                "expired": expired
            }
        }
    return _msg


def get_ticker_by_instrument_request(instrument_request: str) -> json:
    _msg = {
            "jsonrpc": "2.0",
            "id": 8106,
            "method": "public/ticker",
            "params": {
                "instrument_name": f"{instrument_request}"
            }
        }
    return _msg


def test_message() -> json:
    _msg = {
      "jsonrpc": "2.0",
      "id": 8212,
      "method": "public/test",
      "params": {

      }
    }
    return _msg
