# @title Training on Collab by Sanne Karibo\n 🔁 Upload Model, Install Dependencies & Train
from google.colab import files
import os, asyncio, json, time
import importlib.util
import logging
logging.basicConfig(level=logging.INFO, format="🔧 %(message)s")

print("📤 Upload your `Libra6.py` file...")
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
!pip install -q torch numpy requests cloudinary websockets nest_asyncio

# === 🧠 Load Libra6 dynamically ===
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
print(f"✊ Received {len(tick_data)} ticks to RAM ✔️")

# === 🧹 Prepare sequences of at least 301 prices ===
window_size = 301
sequences = []
for i in range(len(tick_data) - window_size):
    window = tick_data[i:i+window_size]
    sequences.append(window)

print(f"🧪 Prepared {len(sequences)} sequences of length {window_size}")

# === 🔧 Train model ===
model = libra.Libra6()
model.download_model_from_cloudinary()
model.continuous_train(sequences, epochs=100)
model.upload_model_to_cloudinary()

print("✅ Training complete ✅")
