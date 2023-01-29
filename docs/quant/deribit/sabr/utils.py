import datetime


def get_human_timestamp(timestamp: int):
    """Get human-readable timestamp from linux date"""
    return datetime.datetime.fromtimestamp(timestamp / 1000000.0).strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )


def get_difference_between_now_and_expirety_date(now: int, expiration_date: int):
    """Returns the time between now and expiration date in YEARS"""
    return (now - expiration_date) / (1_000_000 * 60 * 60 * 24 * 365)

