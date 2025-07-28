import asyncio
import json
import aiohttp
import time
from picows import WebSocketClient

DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3"
PREDICT_URL = "http://localhost:10000/predict"  # Adjust if your main.py is hosted elsewhere
SYMBOL = "stpRNG"

# Load users from /tmp/user.json
def load_users():
    with open("/tmp/user.json", "r") as f:
        return json.load(f)

# Deriv WebSocket handler
class DerivTrader:
    def __init__(self, user_token):
        self.token = user_token
        self.ws = WebSocketClient(DERIV_WS_URL)
        self.authorized = False
        self.current_price = None

    async def authorize(self):
        await self.ws.connect()
        await self.ws.send({
            "authorize": self.token
        })
        auth = await self.ws.recv()
        if "error" in auth:
            print("Authorization failed:", auth)
        else:
            self.authorized = True

    async def subscribe_ticks(self):
        await self.ws.send({
            "ticks": SYMBOL,
            "subscribe": 1
        })

    async def tick_loop(self):
        while True:
            msg = await self.ws.recv()
            if msg and "tick" in msg:
                self.current_price = float(msg["tick"]["quote"])

    async def buy(self):
        proposal = {
            "buy": "1",  # dummy id, server will reject without a valid proposal
            "price": 1,
        }
        await self.ws.send(proposal)
        print(f"[BUY] Attempted buy for {self.token}")

    async def sell(self):
        proposal = {
            "buy": "1",
            "price": 1,
        }
        await self.ws.send(proposal)
        print(f"[SELL] Attempted sell for {self.token}")

    async def run(self):
        await self.authorize()
        if not self.authorized:
            return
        await self.subscribe_ticks()
        await self.tick_loop()  # infinite

# Core logic: price check + trading
async def trader_loop(traders):
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(PREDICT_URL) as resp:
                    data = await resp.json()

                predicted_high = float(data["predicted_high"])
                predicted_low = float(data["predicted_low"])
                last_high = float(data["last_candle_high"])
                last_low = float(data["last_candle_low"])
                midpoint = (predicted_high + predicted_low) / 2

                print(f"[PREDICT] High={predicted_high:.5f} Low={predicted_low:.5f} Mid={midpoint:.5f}")

                for t in traders:
                    price = t.current_price
                    if price is None:
                        continue

                    if price <= predicted_low and last_high < midpoint:
                        await t.buy()
                    elif price >= predicted_high and last_low > midpoint:
                        await t.sell()

            except Exception as e:
                print("Prediction or trade error:", e)

            await asyncio.sleep(60)  # every minute

# Main startup
async def main():
    users = load_users()
    traders = []

    for u in users:
        token = u.get("token")
        if not token:
            continue
        t = DerivTrader(token)
        traders.append(t)

    # Connect and run each trader
    tasks = []
    for t in traders:
        tasks.append(asyncio.create_task(t.run()))

    # Start trading loop
    tasks.append(asyncio.create_task(trader_loop(traders)))

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
