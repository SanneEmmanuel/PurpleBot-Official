from fastapi import FastAPI
from Libra import load_model, predict_ticks, retrain_and_upload
import asyncio, websockets, nest_asyncio, json

app = FastAPI()
model = load_model()

SYMBOL = "stpRng"

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
async def post_prediction_learn(history, predicted):
    actual = await collect_next_ticks()
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
    asyncio.create_task(post_prediction_learn(history, predicted))

    # Step 4: Return predicted values immediately
    return { "predicted": predicted }
