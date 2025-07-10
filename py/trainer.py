# @title 🔁 Upload Model, Install Dependencies & Train
from google.colab import files
import os, asyncio, json
import importlib.util

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

# === 📦 Install dependencies ===
!pip install websockets nest_asyncio torch numpy requests --quiet

# === 🧠 Load model file dynamically ===
spec = importlib.util.spec_from_file_location("libra_module", model_filename)
libra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(libra)

# === 🔁 Async setup ===
import nest_asyncio
nest_asyncio.apply()
import websockets

# === 🌐 Define async WebSocket getTicks() ===
async def getTicks(count=300):
    uri = "wss://ws.derivws.com/websockets/v3?app_id=1089"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "ticks_history": "stpRNG",
            "count": count,
            "end": "latest",
            "style": "ticks"
        })) 
        response = json.loads(await ws.recv())
        prices = response["history"]["prices"]
        print("📨 Received", len(prices), "ticks.")
        return prices

# === 🔃 Fetch ticks ===
tick_data = asyncio.run(getTicks(10000))

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
model = libra.LibraModel()
trained = libra.retrain_and_upload(model, X, Y, epochs=50)
print("✅ Training complete.")
