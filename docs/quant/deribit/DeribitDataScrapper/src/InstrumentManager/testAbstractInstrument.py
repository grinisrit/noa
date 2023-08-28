import unittest
from AbstractInstrument import AbstractInstrument


class MyTestCase(unittest.TestCase):
    def test_trades_appending(self):
        test_instrument = AbstractInstrument(trades_buffer_size=3, order_book_changes_buffer_size=1,
                                             user_trades_buffer_size=1)
        trades = [
            [1000, 1, -1],
            [2000, 23, -1],
            [123450, 10, -1],
            [6666, -10, -1]
        ]
        for _ in trades:
            test_instrument.place_last_trade(trade_price=_[0], trade_amount=_[1], trade_time=_[2])

        self.assertEqual(str(test_instrument.last_trades[0]), str({'trade_time': -1, 'trade_price': 2000, 'trade_amount': 23}))
        self.assertEqual(str(test_instrument.last_trades[1]), str({'trade_time': -1, 'trade_price': 123450, 'trade_amount': 10}))
        self.assertEqual(str(test_instrument.last_trades[2]), str({'trade_time': -1, 'trade_price': 6666, 'trade_amount': -10}))
        self.assertEqual(str(test_instrument.last_trades[-1]), str({'trade_time': -1, 'trade_price': 6666, 'trade_amount': -10}))
        self.assertEqual(str(test_instrument.last_trades[-2]), str({'trade_time': -1, 'trade_price': 123450, 'trade_amount': 10}))
        self.assertEqual(str(test_instrument.last_trades[-4]), str({'trade_time': -1, 'trade_price': 6666, 'trade_amount': -10}))


if __name__ == '__main__':
    unittest.main()
