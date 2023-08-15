from enum import Enum


class InstrumentType(Enum):
    __slots__ = {"_instrument_type"}
    _instrument_type: str

    def __init__(self, value, instrument_type: str):
        self.number = value
        self._instrument_type = instrument_type

    @property
    def instrument_type(self):
        return self._instrument_type

    FUTURE = 1, "future"
    OPTION = 2, "option"
    FUTURE_COMBO = 3, "future_combo"
    OPTION_COMBO = 4, "option_combo"

    CALL_OPTION = 5, "call_option"
    PUT_OPTION = 6, "put_option"

    ASSET = 7, "asset"

if __name__ == '__main__':
    future = InstrumentType.CALL_OPTION
    print(future.number)