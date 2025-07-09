from fastapi import FastAPI
from Libra import load_model, predict_ticks, retrain_and_upload
import asyncio
from deriv_api import DerivAPI

app = FastAPI()
model = load_model()

SYMBOL = "stpRng"

# 📥 Get 300 latest historical ticks
async def fetch_history(symbol=SYMBOL, count=300):
    api = DerivAPI(app_id=1089)
    response = await api.send({
        "ticks_history": symbol,
        "count": count,
        "end": "latest",
        "style": "ticks"
    })
    await api.close()
    return response["history"]["prices"]

# 📡 Collect next 5 live ticks
async def collect_next_ticks(symbol=SYMBOL, count=5):
    api = DerivAPI(app_id=1089)
    result = []
    event = asyncio.Event()

    stream = await api.subscribe({"ticks": symbol})

    def on_tick(data):
        try:
            price = data["tick"]["quote"]
            result.append(price)
            if len(result) >= count:
                event.set()
        except:
            pass

    stream.subscribe(on_tick)
    await event.wait()
    await stream.unsubscribe()
    await api.close()
    return result

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
    history = await fetch_history()

    # Step 2: Predict next 5 ticks
    predicted = predict_ticks(model, history)

    # Step 3: Start background task for collecting actual + retraining
    asyncio.create_task(post_prediction_learn(history, predicted))

    # Step 4: Return predicted values immediately
    return { "predicted": predicted }
