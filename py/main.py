import os
import sys
import json
import asyncio
import subprocess
import websockets
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
from mind import Mind

# Constants
USER_FILE = "/tmp/user.json"
DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3"
APP_ID = os.getenv("DERIV_APP_ID", "1089")
SYMBOL = "stpRNG"
GRANULARITY = 60
SEQUENCE_LENGTH = 20

# Logger setup
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# App setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
mind = Mind(sequence_length=SEQUENCE_LENGTH, download_on_init=True)

# Ensure user file exists
if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump([], f)

# --- Models ---
class PredictionResponse(BaseModel):
    predicted_high: float
    predicted_low: float
    last_candle_high: float
    last_candle_low: float

class TokenRequest(BaseModel):
    token: str

# --- Utility Functions ---
def load_users():
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

def is_registered(token: str):
    users = load_users()
    return any(u.get("token") == token for u in users)

# --- WebSocket Candle Fetch ---
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

# --- Endpoints ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": mind.model_loaded_successfully}

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

@app.post("/register")
async def register_user(req: TokenRequest):
    if is_registered(req.token):
        raise HTTPException(status_code=400, detail="User already registered")
    users = load_users()
    users.append({"token": req.token})
    save_users(users)
    return {"message": "User registered successfully"}

@app.post("/check")
async def check_user(req: TokenRequest):
    if is_registered(req.token):
        return {"registered": True}
    return {"registered": False}

@app.post("/delete")
async def delete_user(req: TokenRequest):
    users = load_users()
    users = [u for u in users if u.get("token") != req.token]
    save_users(users)
    return {"message": "User deleted if existed"}

# --- Launch trader.py on startup ---
@app.on_event("startup")
async def launch_trader():
    try:
        subprocess.Popen(["python3", "trader.py"])
        logger.info("✅ trader.py started successfully")
    except Exception as e:
        logger.error(f"❌ Failed to launch trader.py: {str(e)}")
