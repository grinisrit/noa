# TODO: add comments
# TODO: inherit from Named Tuple
class Underlying:

    def __init__(self,
                 price: float,
                 volatility: float,
                 interest=0.0):
        self.price = price
        self.interest = interest
        self.volatility = volatility


class Option:

    def __init__(self,
                 call: bool,
                 strike: float,
                 maturity=1.0,
                 style=str):

        self.style = style
        self.call = call
        self.strike = strike
        self.maturity = maturity
