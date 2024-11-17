from __future__ import annotations
import logging
import yaml
import datetime as dt
from dataclasses import dataclass
from enum import IntEnum


@dataclass
class DatabaseConfiguration:
  root: str


@dataclass
class DeribitConfiguration:
  endpoint: str
  client_id: str
  client_secret: str
  heartbeat_interval: int
  timeout: float


@dataclass
class TradingConfiguration:
  currencies: list[str]
  start_maturity: dt.datetime 
  end_maturity: dt.datetime  



@dataclass
class Configuration:
  version: str
  database: DatabaseConfiguration
  deribit: DeribitConfiguration
  trading: TradingConfiguration

  __VERSION__ = '0.0.1'

  @staticmethod
  def load(path: str = 'conf.yaml') -> Configuration:
    logging.debug(f'Loading configuration from: {path}')
    conf = None
    try:
      with open(path) as conf_file:
          conf = yaml.safe_load(conf_file)
    except:
      logging.error(f'Failed to load configuration from: {path}', exc_info=True)
      raise RuntimeError
      
    version = conf['version']
    database = conf['database']
    deribit = conf['deribit']
    trading = conf['trading']
    current_time = dt.datetime.utcnow()
    current_date = dt.datetime(current_time.year, current_time.month, current_time.day)

    assert version == Configuration.__VERSION__

    return Configuration(
      version = conf['version'],
      database = DatabaseConfiguration(root = database['root']),
      deribit = DeribitConfiguration(
        endpoint = deribit['endpoint'],
        client_id = deribit['client_id'],
        client_secret = deribit['client_secret'],
        heartbeat_interval = deribit['heartbeat_interval'],
        timeout = deribit['timeout']
      ),
      trading = TradingConfiguration(
        currencies = trading['currencies'],
        start_maturity = current_date + dt.timedelta(days=trading['start_maturity']),
        end_maturity = current_date + dt.timedelta(days=trading['end_maturity'])
      )
    )
  

class MessageId(IntEnum):
  HEARTBEAT = 1
  TEST = 2
  AUTH = 3
