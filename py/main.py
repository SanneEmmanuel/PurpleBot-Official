import os
import json
import asyncio
import logging
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
import websockets
from pydantic import BaseModel
from typing import Dict
from mind import Mind

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MindAPI")

# Initialize FastAPI app
app = FastAPI(
    title="PurplePlatform Trading Prediction API",
    description="AI-powered high/low prediction for trading instruments",
    version="2.0.2",
    docs_url="/docs",
    redoc_url=None
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# User DB path (ephemeral, resets on restart)
USER_DB_PATH = "/tmp/user_db.json"

# Load user DB
try:
    with open(USER_DB_PATH, "r") as f:
        USER_DB = json.load(f)
    logger.info(f"Loaded user DB with {len(USER_DB)} users")
except (FileNotFoundError, json.JSONDecodeError):
    USER_DB = {}
    logger.info("Initialized empty user DB")

# Deriv API
DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3"
APP_ID = os.getenv("DERIV_APP_ID", "1089")
SYMBOL = "stpRNG"
GRANULARITY = 60

# Initialize model (no training or fallback)
mind = Mind(
    sequence_length=20,
    download_on_init=True,
    upload_on_fail=False
)
logger.info("Mind model initialized. Status: %s", "Loaded" if mind.model_loaded_successfully else "Not Loaded")

# Pydantic models
class UserCreate(BaseModel):
    username: str
    token: str

class UserResponse(BaseModel):
    username: str
    token: str

class PredictionRequest(BaseModel):
    symbol: str = SYMBOL
    granularity: int = GRANULARITY
    token: str

class PredictionResponse(BaseModel):
    predicted_high: float
    predicted_low: float
    last_candle_high: float
    last_candle_low: float

# Save user DB
def save_user_db():
    with open(USER_DB_PATH, "w") as f:
        json.dump(USER_DB, f)

# Token validator
def validate_token(token: str):
    if token not in USER_DB.values():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

# Fetch candles from Deriv
async def get_candles(symbol: str, count: int, granularity: int) -> list:
    logger.info("Fetching %d candles for %s (%ds)", count, symbol, granularity)
    payload = {
        "ticks_history": symbol,
        "count": count,
        "granularity": granularity,
        "end": "latest",
        "style": "candles"
    }
    for attempt in range(3):
        try:
            async with websockets.connect(f"{DERIV_WS_URL}?app_id={APP_ID}") as ws:
                await ws.send(json.dumps(payload))
                response = json.loads(await ws.recv())
                candles = response.get("candles")
                if candles:
                    return [[c["high"], c["low"]] for c in candles]
        except Exception as e:
            logger.error("Attempt %d failed: %s", attempt + 1, str(e))
            await asyncio.sleep(2 ** attempt)
    raise HTTPException(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        detail="Unable to fetch candle data"
    )

# Endpoints
@app.post("/signup", status_code=status.HTTP_201_CREATED, response_model=UserResponse)
async def signup(user: UserCreate):
    if user.username in USER_DB:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists"
        )
    USER_DB[user.username] = user.token
    save_user_db()
    logger.info("User created: %s", user.username)
    return {"username": user.username, "token": user.token}

@app.get("/findUser", response_model=UserResponse)
async def find_user(username: str = Query(...)):
    if username not in USER_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return {"username": username, "token": USER_DB[username]}

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": mind.model_loaded_successfully,
        "users_registered": len(USER_DB),
        "storage": "ephemeral (/tmp)"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    validate_token(request.token)
    try:
        candles = await get_candles(
            symbol=request.symbol,
            count=mind.sequence_length,
            granularity=request.granularity
        )
    except Exception as e:
        logger.exception("Failed to get candles")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Candle fetch error: {str(e)}"
        )

    if len(candles) != mind.sequence_length:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Expected {mind.sequence_length} candles, got {len(candles)}"
        )

    try:
        last_candle = candles[-1]
        prediction = mind.predict(candles)
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

    return {
        "predicted_high": prediction["Predicted High"],
        "predicted_low": prediction["Predicted Low"],
        "last_candle_high": last_candle[0],
        "last_candle_low": last_candle[1]
    }

@app.on_event("shutdown")
def persist_shutdown():
    save_user_db()
    logger.info("Saved DB on shutdown")

# Dev server entry
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
