from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio, websockets, json, numpy as np
from collections import deque
from typing import Optional

from libra6 import Libra6  # ✅ Directly import the class

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Tick buffer (latest 400 ticks)
tick_buffer = deque(maxlen=400)

# ✅ Load model once globally
model = Libra6()
try:
    model.download_model_from_cloudinary()
    print("✅ Model loaded from Cloudinary.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")

# ✅ Persistent WebSocket listener
async def websocket_listener():
    uri = "wss://ws.derivws.com/websockets/v3?app_id=1089"
    while True:
        try:
            async with websockets.connect(uri) as ws:
                await ws.send(json.dumps({
                    "ticks": "stpRNG",
                    "subscribe": 1
                }))
                print("🔌 Subscribed to Deriv ticks...")
                while True:
                    message = json.loads(await ws.recv())
                    tick = message.get("tick", {}).get("quote")
                    tick and tick_buffer.append(float(tick))
        except Exception as e:
            print(f"❌ WebSocket error: {e}. Reconnecting in 3s...")
            await asyncio.sleep(3)

# ✅ Fetch from WebSocket if buffer is cold
async def fetch_and_store_ticks(count):
    print("⚠️ Not enough ticks in buffer — fetching from Deriv...")
    uri = "wss://ws.derivws.com/websockets/v3?app_id=1089"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "ticks_history": "stpRNG",
            "count": count,
            "end": "latest",
            "style": "ticks"
        }))
        message = json.loads(await ws.recv())
        fetched = message.get("history", {}).get("prices", [])
        if not fetched:
            raise RuntimeError("❌ Could not fetch fallback ticks from Deriv.")
        tick_buffer.extend(fetched)
        print(f"🧠 Buffer filled with {len(fetched)} fallback ticks.")
        return fetched

# ✅ Branchless getTicks with fallback
async def getTicks(count=301):
    return (
        list(tick_buffer)[-count:]
        if len(tick_buffer) >= count
        else await fetch_and_store_ticks(count)
    )

@app.get("/prices")
async def get_prices(count: Optional[int] = 300):
    try:
        ticks = await getTicks(count)
        return {
            "data": ticks,
            "count": len(ticks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        


# ✅ Updated Prediction endpoint with ticks parameter
@app.post("/predict")
async def predict(ticks: Optional[int] = 5):
    if model is None:
        return {"error": "Model not loaded."}

    if ticks <= 0:
        raise HTTPException(status_code=400, detail="Number of ticks must be positive")

    history = await getTicks()
    model.update(history)
    predicted = model.predictWithConfidence(num_ticks=ticks) 
    return predicted


@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
def root():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "tick_buffer_len": len(tick_buffer)
    }

# ✅ Startup: Launch WebSocket listener
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(websocket_listener())
