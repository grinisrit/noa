from AvailableCurrencies import Currency
from typing import Callable

currencyInfoMsg = {
    "jsonrpc": "2.0",
    "id": 7538,
    "method": "public/get_currencies",
    "params": {

    }
}

price_data_msg: Callable[[str], dict] = lambda instrument_name: {
    "jsonrpc": "2.0",
    "id": 8106,
    "method": "public/ticker",
    "params": {
        "instrument_name": f"{instrument_name}"
    }
}

get_last_trades: Callable[[Currency, int], dict] = lambda instrument_name, number_of_last_trades: {
  "jsonrpc": "2.0",
  "id": 9290,
  "method": "public/get_last_trades_by_currency",
  "params": {
    "currency": f"{instrument_name.currency}",
    "count": f"{number_of_last_trades}",
  }
}


