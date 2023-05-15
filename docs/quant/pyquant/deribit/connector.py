from typing import Optional
from threading import Thread
from websocket import WebSocketApp, ABNF
import logging
import json
import time
import ray

from configuration import Configuration, DeribitConfiguration, MessageId


class Subscriber:
  def process_message(self, message: str) -> None:
    logging.info(f'Subscriber received {message}')


@ray.remote
class DeribitConnector:

  def __init__(self, configuration: Configuration) -> None:
    self._conf: DeribitConfiguration = configuration.deribit
    self._subscribers: list[Subscriber] = []
    self._subscriptions: dict[str, str] = {}
    self._socket_thread: Thread = Thread(target=self._thread_task)
    self._socket: WebSocketApp = WebSocketApp(self._conf.endpoint,
                                  on_message=self._on_message, on_open=self._on_open, on_error=self._on_error)
    self._last_request_time: Optional[float] = None
    self._run_socket: bool = True

  def add_subscriber(self, subscriber: Subscriber) -> None:
    self._subscribers.append(subscriber)

  def add_subscription(self, name: str, request: str) -> None:
    self._subscriptions[name] = request

  def remove_subscription(self, name: str) -> None:
    del self._subscriptions[name]
      
  def start(self) -> None:
    self._run_socket = True
    self._socket_thread.start()
  
  def stop(self) -> None:
    self._run_socket = False
    self._socket.close()
    self._socket_thread.join()

  def send_request(self, message: str) -> None:
    if self._last_request_time is not None:
      timeout_before_next_request = self._conf.timeout - (time.time() - self._last_request_time)
      if timeout_before_next_request > 0:
        logging.info(f'Deribit Connector: waiting for {timeout_before_next_request:.2f} seconds before next request...')
        time.sleep(timeout_before_next_request)

    self._last_request_time = time.time()
    self._socket.send(message, ABNF.OPCODE_TEXT)

  def _thread_task(self) -> None:
    while self._run_socket:
      try:
        self._socket.run_forever()
      except:
        logging.error(f'Failure in Deribit Connector', exc_info=True)
        continue

  def _on_open(self, _) -> None:
    self.send_request(self._auth_message())
    self.send_request(self._heartbeat_message())
    for _, subscription in self._subscriptions.items():
      self.send_request(subscription)
  
  def _on_error(self, _, error) -> None:
    logging.error(error)

  def _on_message(self, _, message) -> None:
    for subscriber in self._subscribers:
      subscriber.process_message.remote(message)

  def _heartbeat_message(self) -> str:
    return json.dumps(
        {
        "jsonrpc": "2.0",
        "id": MessageId.HEARTBEAT,
        "method": "public/set_heartbeat",
        "params": {
          "interval": self._conf.heartbeat_interval
        }
      }
    )
  
  def _auth_message(self) -> str:
    return json.dumps(
      {
        "jsonrpc": "2.0",
        "id": MessageId.AUTH,
        "method": "public/auth",
        "params": {
          "grant_type": "client_credentials",
          "client_id": self._conf.client_id,
          "client_secret": self._conf.client_secret
        }
      }
    )
  
            