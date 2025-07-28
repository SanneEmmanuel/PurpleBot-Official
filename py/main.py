import os
import json
import asyncio
import sys
from datetime import datetime, timedelta
import websockets
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
from mind import Mind

# Logger
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# Constants
DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3"
APP_ID = os.getenv("DERIV_APP_ID", "1089")
SYMBOL = "stpRNG"
GRANULARITY = 60  # seconds
SEQUENCE_LENGTH = 20

# App & model
app = FastAPI()
mind = Mind(sequence_length=SEQUENCE_LENGTH, download_on_init=True)

# Response schema
class PredictionResponse(BaseModel):
    predicted_high: float
    predicted_low: float
    last_candle_high: float
    last_candle_low: float

# Fetch past candles, excluding most recent 1 minute
async def get_candles():
    # Compute UTC timestamp 1 minute ago
    end_time = int((datetime.utcnow() - timedelta(minutes=1)).timestamp())
    payload = {
        "ticks_history": SYMBOL,
        "count": SEQUENCE_LENGTH,
        "granularity": GRANULARITY,
        "end": end_time,
        "style": "candles"
    }

    for attempt in range(3):
        try:
            async with websockets.connect(f"{DERIV_WS_URL}?app_id={APP_ID}") as ws:
                await ws.send(json.dumps(payload))
                response = await ws.recv()
                data = json.loads(response)
                if "candles" in data:
                    return [[c["high"], c["low"]] for c in data["candles"]]
        except Exception as e:
            logger.warning("Attempt {}: Candle fetch failed - {}", attempt + 1, str(e))
            await asyncio.sleep(1)

    raise HTTPException(status_code=504, detail="Failed to fetch candle data")

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": mind.model_loaded_successfully
    }

# Predict endpoint (GET)
@app.get("/predict", response_model=PredictionResponse)
async def predict():
    if not mind.model_loaded_successfully:
        raise HTTPException(status_code=503, detail="Model not loaded")

    candles = await get_candles()
    if len(candles) < SEQUENCE_LENGTH:
        raise HTTPException(status_code=422, detail="Not enough candle data")

    try:
        prediction = mind.predict(candles)
        last = candles[-1]
        return {
            "predicted_high": prediction["Predicted High"],
            "predicted_low": prediction["Predicted Low"],
            "last_candle_high": last[0],
            "last_candle_low": last[1]
        }
    except Exception as e:
        logger.error("Prediction error: {}", str(e))
        raise HTTPException(status_code=500, detail="Prediction failed")
