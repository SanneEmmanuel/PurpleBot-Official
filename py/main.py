from fastapi import FastAPI
from Libra import load_model, predict_ticks, retrain_and_upload
import asyncio, websockets, nest_asyncio, json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ Allow all origins (for dev; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend domain like ["https://yourfrontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
try:
    model = load_model()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# 📥 Get 300 latest historical ticks
async def getTicks(count=300):
    uri = "wss://ws.derivws.com/websockets/v3?app_id=1089"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "ticks_history": "stpRNG",
            "count": count,
            "end": "latest",
            "style": "ticks"
        })) 

        prices = json.loads(await ws.recv())["history"]["prices"]
        print("📨 Received:",prices )
        return prices


# 🔧 Background task for listening and retraining
async def post_prediction_learn(predicted):
    await asyncio.sleep(5)  # Wait 5 seconds
    ticks = await getTicks(305)  # Get 305 ticks
    actual = ticks[:5]           # First 5 are the actual next ticks
    history = ticks[5:]          # Remaining 300 are the history

    print("📈 Predicted:", predicted)
    print("📊 Actual   :", actual)
    print("🔍 Difference:", [round(a - p, 5) for a, p in zip(actual, predicted)])
    retrain_and_upload(model, [history], [actual])

@app.post("/predict")
async def predict():
    # Step 1: Fetch 300 historical ticks
    history = await getTicks()
    # Step 2: Predict next 5 ticks
    predicted = predict_ticks(model, history)
    print("Prediction Ran Successfully\n::",predicted)
    # Step 3: Start background task for collecting actual + retraining
    asyncio.create_task(post_prediction_learn(predicted['prices']))
    # Step 4: Return predicted values immediately
    return { "predicted": predicted }
