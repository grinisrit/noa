import nest_asyncio

nest_asyncio.apply()
import asyncio
from typing import TYPE_CHECKING
import threading
from time import time

strategyType = object
if TYPE_CHECKING:
    from docs.quant.deribit.TradingInterfaceBotJulyVersion.Strategy.AbstractStrategy import AbstractStrategy
    strategyType = AbstractStrategy


class TickerNode:
    """
    Class that will ping strategy block with ping_time frequency
    """
    ping_time: float
    last_ping_time: float

    ticker_wait_time: float
    tickerThread: threading.Thread = None
    connected_strategy: strategyType = None

    ticker_loop: asyncio.AbstractEventLoop

    def __init__(self, ping_time: float, wait_parameter: float = 0.0001):
        """

        :param ping_time: tickerNode frequency (in sec)
        :param wait_parameter: (in sec)
        """
        if wait_parameter > ping_time:
            wait_parameter = ping_time / 2
        self.ping_time = ping_time * 1_000
        self.last_ping_time = time() * 1_000

        self.ticker_wait_time = wait_parameter

    def connect_strategy(self, plug_strategy: strategyType):
        self.connected_strategy = plug_strategy

    def run_ticker_node(self):
        if self.connected_strategy is None:
            raise ConnectionError("No strategy plugged to tickerNode")
        else:
            self.tickerThread = threading.Thread(target=self.run_ticker_node_task)
            self.tickerThread.start()

    def run_ticker_node_task(self):
        self.ticker_loop = asyncio.new_event_loop()
        self.ticker_loop.run_until_complete(self._ticker_worker())

    async def _ticker_worker(self):
        while True:
            if time() * 1_000 - self.last_ping_time >= self.ping_time:
                self.last_ping_time = time() * 1_000
                asyncio.run_coroutine_threadsafe(
                    coro=self.connected_strategy.on_tick_update(callback={"time": self.last_ping_time}),
                    loop=self.ticker_loop)
            else:
                await asyncio.sleep(self.ticker_wait_time)


if __name__ == '__main__':
    from docs.quant.deribit.TradingInterfaceBotJulyVersion.Strategy.BasicStrategy import BaseStrategy
    ticker = TickerNode(ping_time=1)

    base_strategy = BaseStrategy()
    ticker.connect_strategy(base_strategy)

    ticker.run_ticker_node()

