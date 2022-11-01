
import asyncio
import logging
import os
import re
from uuid import uuid4
from typing import Pattern
from ScrapperLive import DeribitClient

class MyApp:
    def __init__(self):
        logging.basicConfig(
            format='%(asctime)s %(name)s %(funcName)s %(levelname)s %(message)s', level=logging.INFO)
        self.logger = logging.getLogger("deribit-private")
        self.direct_requests = {}

        self.deribit = DeribitClient('', '', '', True)
        self.deribit.on_handle_response = self.on_handle_response
        # self.deribit.on_response_error = self.on_response_error

    def on_handle_response(self, data):
        request_id = data["id"]
        if request_id in self.direct_requests:
            self.logger.info(
                f"Caught response {repr(data)} to direct request {self.direct_requests[request_id]}")
        else:
            self.logger.error(
                f"Can't find request with id:{request_id} for response:{repr(data)}")

    async def run(self):
        await self.deribit.run_receiver()
        await self.deribit.send(_msg)

    async def stop(self):
        await self.deribit.stop()


_msg ={"method": "public/get_instruments",
            "params": {
                "currency": f"BTC",
                "kind": f"option",
                "expired": True
            }
        }

if __name__ == '__main__':
    app = MyApp()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(app.run())
    except KeyboardInterrupt:
        loop.run_until_complete(app.stop())
    finally:
        app.logger.info('Program finished')