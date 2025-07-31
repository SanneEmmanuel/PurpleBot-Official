import asyncio, json, aiohttp
from picows import WebSocketClient

DERIV_WS = "wss://ws.derivws.com/websockets/v3"
PREDICT_URL = "http://localhost:10000/predict"
SYMBOL = "stpRNG"

def load_users():
    with open("/tmp/user.json") as f: return json.load(f)

class Trader:
    def __init__(self, token):
        self.token = token
        self.ws = WebSocketClient(DERIV_WS)
        self.price = None

    async def start(self):
        await self.ws.connect()
        await self.ws.send({"authorize": self.token})
        auth = await self.ws.recv()
        if "error" in auth: return
        await self.ws.send({"ticks": SYMBOL, "subscribe": 1})
        asyncio.create_task(self.listen())

    async def listen(self):
        while True:
            msg = await self.ws.recv()
            if "tick" in msg:
                self.price = float(msg["tick"]["quote"])

    async def buy(self): await self.ws.send({"buy": "1", "price": 1})
    async def sell(self): await self.ws.send({"buy": "1", "price": 1})

async def trade_loop(traders):
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                r = await session.get(PREDICT_URL)
                p = await r.json()
                hi, lo = float(p["predicted_high"]), float(p["predicted_low"])
                lhi, llo = float(p["last_candle_high"]), float(p["last_candle_low"])
                mid = (hi + lo) / 2
                print(f"[PREDICT] High={hi:.5f} Low={lo:.5f} Mid={mid:.5f}")

                for t in traders:
                    if not t.price: continue
                    if t.price <= lo and lhi < mid: await t.buy()
                    if t.price >= hi and llo > mid: await t.sell()
            except Exception as e:
                print("Trade error:", e)
            await asyncio.sleep(60)

async def main():
    traders = [Trader(u["token"]) for u in load_users() if u.get("token")]
    await asyncio.gather(*(t.start() for t in traders), trade_loop(traders))

if __name__ == "__main__":
    asyncio.run(main())
