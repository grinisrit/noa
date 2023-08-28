from dataclasses import dataclass
from enum import Enum


class OrderSide(Enum):
    BUY = 1,
    SELL = -1


class OrderType(Enum):
    deribit_name: str

    def __init__(self, deribit_naming: str):
        self.deribit_name = deribit_naming

    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    TAKE_LIMIT = "take_limit"
    MARKET = "market"
    STOP_MARKET = "stop_market"
    TAKE_MARKET = "take_market"
    MARKET_LIMIT = "market_limit"
    TRAILING_STOP = "trailing_stop"


class OrderState(Enum):
    deribit_name: str

    def __init__(self, deribit_naming: str):
        self.deribit_name = deribit_naming

    OPEN = "open"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    UNTRIGGERED = "untriggered"


def convert_deribit_order_type_to_structure(der_ans: str) -> OrderType:
    match der_ans:
        case "limit":
            return OrderType.LIMIT
        case "market":
            return OrderType.MARKET
        case _:
            raise NotImplementedError


def convert_deribit_order_status_to_structure(der_ans: str) -> OrderState:
    match der_ans:
        case "open":
            return OrderState.OPEN
        case "filled":
            return OrderState.FILLED
        case "rejected":
            return OrderState.REJECTED
        case "cancelled":
            return OrderState.CANCELLED
        case _:
            raise NotImplementedError


@dataclass()
class OrderStructure:
    order_tag: str
    order_id: int | None
    open_time: int
    price: float | None
    executed_price: float
    total_commission: float
    direction: str
    order_amount: float
    filled_amount: float
    last_update_time: int
    order_exist_time: int
    instrument: str
    order_type: OrderType
    order_state: OrderState
