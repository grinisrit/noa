from enum import Enum


class InstrumentType(Enum):
    __slots__ = {"_instrument_type"}
    _instrument_type: str

    def __init__(self, instrument_type: str):
        self._instrument_type = instrument_type

    @property
    def instrument_type(self):
        return self._instrument_type

    FUTURE = "future"
    OPTION = "option"
    FUTURE_COMBO = "future_combo"
    OPTION_COMBO = "option_combo"


