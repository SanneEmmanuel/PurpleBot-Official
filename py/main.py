from fastapi import FastAPI
from Libra import load_model, predict_ticks, retrain_and_upload
import asyncio, websockets, nest_asyncio, json
from fastapi.middleware.cors import CORSMiddleware
from collections import deque
import numpy as np

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
async def getTicks(count=300):
    return (
        list(tick_buffer)[-count:]
        if len(tick_buffer) >= count
        else await fetch_and_store_ticks(count)
    )

# ✅ Background retrain
async def post_prediction_learn(predicted):
    try:
        print("✅ Prediction Success ::", predicted)

        model = load_model()  # ⏱️ Load model early to parallelize waiting
        await asyncio.sleep(5)

        ticks = await getTicks(305)
        actual, history = ticks[:5], ticks[5:]

        pred, act = np.array(predicted, dtype=np.float32), np.array(actual, dtype=np.float32)
        diffs = np.abs(act - pred)
        avg = diffs.mean()

        print("📉 Error:", np.round(diffs, 2).tolist(), "| Avg:", round(float(avg), 3))

        epochs = int(np.clip((avg / 0.2) * 10, 1, 10)) * int(avg >= 0.1)
        print("🔁 Retrain:", f"{epochs} epoch(s)" if epochs else "No retraining")

        if epochs and len(history) >= 300:
            await asyncio.to_thread(retrain_and_upload, model, [history[:300]], [actual], epochs)

    except Exception as e:
        print("❌ post_prediction_learn error:", e)


# ✅ Prediction endpoint
@app.post("/predict")
async def predict():
    if model is None:
        return { "error": "Model not loaded." }
    history = await getTicks()
    predicted = predict_ticks(model, history)
    asyncio.create_task(post_prediction_learn(predicted['prices']))
    return { "predicted": predicted }

# ✅ Health check
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "tick_buffer_len": len(tick_buffer)
    }

# ✅ Startup: Load model + start WebSocket
@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = load_model()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None
    asyncio.create_task(websocket_listener())
