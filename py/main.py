import os
import sys
import json
import asyncio
import websockets
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
from mind import Mind  # Your custom AI model wrapper

# Logger setup
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3"
APP_ID = os.getenv("DERIV_APP_ID", "1089")  # Use your App ID or default
SYMBOL = "stpRNG"
GRANULARITY = 60
SEQUENCE_LENGTH = 20

# Model
mind = Mind(sequence_length=SEQUENCE_LENGTH, download_on_init=True)

# Response model
class PredictionResponse(BaseModel):
    predicted_high: float
    predicted_low: float
    last_candle_high: float
    last_candle_low: float

# Candle fetcher
async def get_candles():
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

                if "candles" in data and isinstance(data["candles"], list):
                    candles = [[c["high"], c["low"]] for c in data["candles"]]
                    if len(candles) >= SEQUENCE_LENGTH:
                        return candles
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}: Candle fetch failed - {str(e)}")
            await asyncio.sleep(1)

    raise HTTPException(status_code=504, detail="Failed to fetch candle data")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": mind.model_loaded_successfully}

# Prediction endpoint
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
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")
