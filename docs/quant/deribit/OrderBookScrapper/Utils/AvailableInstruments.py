# Enum for all available Instruments. If you need, you can add new currencies here.
from enum import Enum


class Instrument(Enum):
    __slots__ = {"_instrument"}
    _instrument: str

    def __init__(self, instrument: str):
        self._instrument = instrument

    @property
    def instrument(self):
        return self._instrument

    BTC_PERPETUAL = "BTC-PERPETUAL"


if __name__ == '__main__':
    print(Instrument.BTC_PERPETUAL.instrument)
