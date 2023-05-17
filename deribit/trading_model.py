from typing import Optional
import ray
import logging

from .configuration import Configuration, TradingConfiguration, MessageId
from .connector import Subscriber, DeribitConnector


@ray.remote 
class TradingModel(Subscriber):

  def __init__(self, configuration: Configuration) -> None:
    self._conf: TradingConfiguration = configuration.trading
    self._deribit_connector: Optional[DeribitConnector] = None

  def set_connector(self, deribit_connector: DeribitConnector) -> None:
    self._deribit_connector = deribit_connector

  def process_message(self, message: str) -> None:
    for request in self._infra_model.process_message(message):
      self._deribit_connector.send_request.remote(request)
    
    id = self._infra_model.get_id()
    if id == MessageId.HEARTBEAT:
      self._on_hearbeat()

  def _on_hearbeat(self):
    logging.info('Updating maturities')
