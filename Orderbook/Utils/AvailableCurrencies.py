# Enum for all available Currencies. If you need, you can add new currencies here.
from enum import Enum


class Currency(Enum):
    __slots__ = {"_coin_type", "_currency", "_currency_long"}
    _coin_type: str
    _currency: str
    _currency_long: str

    def __init__(self, coin_type: str, currency: str, currency_long: str):
        self._coin_type = coin_type
        self._currency = currency
        self._currency_long = currency_long

    @property
    def currency(self):
        return self._currency

    @property
    def coin_type(self):
        return self._coin_type

    @property
    def currency_long(self):
        return self._currency_long

    ETHpoW = ('ETHER', 'ETHW', 'Ethereum PoW')
    SOL = ('SOL', 'SOL', 'Solana')
    USDC = ('USDC', 'USDC', 'USD Coin')
    ETHER = ('ETHER', 'ETH', 'Ethereum')
    BITCOIN = ('BITCOIN', 'BTC', 'Bitcoin')


if __name__ == '__main__':
    print(Currency.ETHER.coin_type)
