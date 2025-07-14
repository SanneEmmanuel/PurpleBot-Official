

# @title Training on Collab by Sanne Karibo\n 🔁 Upload Model, Install Dependencies & Train
from google.colab import files
import os, asyncio, json,time
import importlib.util
import logging
logging.basicConfig(level=logging.INFO, format="🔧 %(message)s")

print("📤 Upload your `Libra.py` file...")
uploaded = files.upload()

# Save the uploaded Python file
for filename in uploaded:
    if filename.endswith(".py"):
        model_filename = filename
        break
else:
    raise RuntimeError("❌ No valid .py file uploaded.")

print(f"✅ Uploaded: {model_filename}")

# ✅ Install all required dependencies quietly
!pip install -q torch numpy requests cloudinary

# === 🧠 Load model file dynamically ===
spec = importlib.util.spec_from_file_location("libra_module", model_filename)
libra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(libra)

# === 🔁 Async setup ===
import nest_asyncio
nest_asyncio.apply()
import websockets

# === 🌐 Define async WebSocket getTicks() ===
async def getTicks(count = 300):
    ticks = []
    remaining = count
    end_time = "latest"
    uri = "wss://ws.derivws.com/websockets/v3?app_id=1089"

    async with websockets.connect(uri) as ws:
        while remaining > 0:
            await ws.send(json.dumps({
                "ticks_history": "stpRNG",
                "count": min(remaining, 5000),
                "end": end_time,
                "style": "ticks"
            }))
            res = json.loads(await ws.recv())

            if "error" in res:
                raise RuntimeError("❌ API Error:", res["error"]["message"])
            prices = res.get("history", {}).get("prices", [])
            if not prices:
                break

            ticks = prices + ticks
            remaining -= len(prices)
            end_time = res["history"]["times"][-1]
            await asyncio.sleep(0.1)

    if not ticks:
        raise ValueError("❌ getTicks() returned 0 ticks.")

    return ticks[-count:]


# === 🔃 Fetch ticks ===
tick_data = asyncio.run(getTicks(89600))
print(f"✊received {len(tick_data)}ticks to Ram✔️")

# === 🧹 Prepare (X, Y) data for training ===
X, Y = [], []
window = 300
output = 5
for i in range(len(tick_data) - window - output):
    x_seq = tick_data[i:i+window]
    y_seq = tick_data[i+window:i+window+output]
    X.append(x_seq)
    Y.append(y_seq)

print(f"🧪 Prepared {len(X)} samples.")

# === 🔧 Train model ===
model = libra.load_model()
trained = libra.retrain_and_upload(model, X, Y, epochs=100, peft_rank=0)

print("✅ Training complete::",trained)
