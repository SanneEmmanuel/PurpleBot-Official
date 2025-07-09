#@title Full Training and Deployment Script

# 1. Install Dependencies
# This block handles the installation of required packages quietly.
try:
    import nest_asyncio
    import python_deriv_api
    import cloudinary
    import torch
except ImportError:
    print("Installing dependencies...")
    !pip install python_deriv_api cloudinary nest_asyncio torch --quiet
    print("Dependencies installed.")

# 2. Import Libraries and Configure
import os
import requests
import zipfile
import logging
import asyncio
import time
from io import BytesIO

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from deriv_api import DerivAPI
import cloudinary
import cloudinary.uploader
import nest_asyncio

# --- Configuration ---
# Cloudinary credentials for model storage
cloudinary.config(
    cloud_name="dj4bwntzb",
    api_key="354656419316393",
    api_secret="M-Trl9ltKDHyo1dIP2AaLOG-WPM"
)

# Paths and device setup
MODEL_PATH = "/tmp/model.pt"
ZIP_PATH = "/tmp/model.pt.zip"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"🔌 Using device: {DEVICE}")

# 3. Define Model and Utilities
class LibraModel(nn.Module):
    """
    A PyTorch model for time-series prediction using an LSTM and a MultiheadAttention layer.

    Args:
        input_size (int): The number of expected features in the input `x`.
        hidden_size (int): The number of features in the hidden state `h`.
        lstm_layers (int): Number of recurrent layers.
        dropout (float): If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer.
        attn_heads (int): Number of parallel attention heads.
    """
    def __init__(self, input_size=1, hidden_size=64, lstm_layers=2, dropout=0.2, attn_heads=2):
        super().__init__()
        # Ensure hidden_size is divisible by attn_heads for valid attention mechanism
        if hidden_size % attn_heads != 0:
            raise ValueError("hidden_size must be divisible by attn_heads")

        # Long Short-Term Memory (LSTM) layer
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            lstm_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Multi-head Attention layer
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attn_heads,
            batch_first=True
        )
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer to produce the final output
        self.fc = nn.Linear(hidden_size, 5)

    def forward(self, x):
        """
        Defines the forward pass of the LibraModel.

        Args:
            x (torch.Tensor): The input features.

        Returns:
            torch.Tensor: The output of the model.
        """
        # Pass input through the LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Apply the attention mechanism
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        
        # Apply dropout to the last time step's output before the final layer
        # The output of interest is the last one in the sequence, hence attn_out[:, -1, :]
        out = self.dropout(attn_out[:, -1, :])
        
        # Pass through the final fully connected layer
        return self.fc(out)


def download_model_from_cloudinary(retries=3, delay=2):
    """Downloads and extracts the model from Cloudinary with retries."""
    for attempt in range(1, retries + 1):
        try:
            logging.info(f"📦 Attempt {attempt}: Downloading model from Cloudinary...")
            url, _ = cloudinary.utils.cloudinary_url("model.pt.zip", resource_type="raw")
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            with zipfile.ZipFile(BytesIO(r.content)) as z:
                z.extractall("/tmp")
            logging.info("✅ Model extracted to /tmp.")
            return True
        except requests.exceptions.RequestException as e:
            logging.warning(f"❌ Download failed (attempt {attempt}): {e}")
            time.sleep(delay * attempt)  # Exponential backoff
    logging.error("🚫 All download attempts failed.")
    return False

def upload_model_to_cloudinary():
    """Zips and uploads the trained model to Cloudinary."""
    if not os.path.exists(MODEL_PATH):
        logging.error("❌ Cannot upload: model.pt not found.")
        return
    try:
        with zipfile.ZipFile(ZIP_PATH, 'w') as zipf:
            zipf.write(MODEL_PATH, arcname="model.pt")
        logging.info(f"📦 Model zipped to {ZIP_PATH}")

        response = cloudinary.uploader.upload(
            ZIP_PATH,
            public_id="model.pt",
            resource_type="raw",
            overwrite=True
        )
        logging.info(f"🚀 Upload successful. Public ID: {response.get('public_id')}")
    except Exception as e:
        logging.error(f"❌ Upload failed: {e}")

def load_model():
    """Initializes the model and loads weights from Cloudinary if available."""
    download_success = download_model_from_cloudinary()
    
    logging.info("🧠 Initializing LibraModel...")
    model = LibraModel().to(DEVICE)
    logging.info(f"📦 Model moved to device: {DEVICE}")
    
    if download_success and os.path.exists(MODEL_PATH):
        try:
            logging.info(f"📥 Loading model weights from {MODEL_PATH}...")
            start_time = time.time()
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            elapsed = time.time() - start_time
            logging.info(f"✅ Model loaded in {elapsed:.2f} seconds.")
        except Exception as e:
            logging.warning(f"⚠️ Could not load model weights: {e}. A new model will be used.")
    else:
        logging.warning("⚠️ No pre-trained model found. A new model will be trained and uploaded.")
        
    model.eval()
    return model

# 4. Training and Upload Logic
def retrain_and_upload(model, x_data, y_data, epochs=50):
    """Trains the model on new data and uploads it."""
    if not x_data or not y_data:
        logging.error("❌ No training data available. Skipping training.")
        return

    model.train()
    x_tensor = torch.tensor(x_data, dtype=torch.float32).view(-1, 300, 1)
    y_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 5)
    
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    logging.info("💪 Starting model training...")
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        logging.info(f"📚 Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    logging.info("💾 Model saved locally.")
    upload_model_to_cloudinary()

# 5. Data Fetching and Main Function
async def fetch_and_prepare_data():
    """Connects to Deriv API to fetch and prepare training data."""
    logging.info("🌐 Connecting to Deriv API...")
    api = DerivAPI(app_id=1089)
    
    total_ticks_needed = 9000
    batch_size = 3000
    ticks = []
    end_time = 'latest'

    while len(ticks) < total_ticks_needed:
        try:
            logging.info(f"📥 Requesting {batch_size} ticks ending at {end_time}...")
            res = await api.ticks_history({
                "ticks_history": "stpRNG",
                "end": end_time,
                "count": batch_size,
                "style": "ticks"
            })
            
            history = res.get("history", {})
            prices = history.get("prices", [])
            times = history.get("times", [])
            
            if not prices:
                logging.warning("⚠️ No prices returned in this batch; stopping fetch.")
                break

            batch = sorted(zip(times, prices)) # Ensure chronological order
            ticks.extend([p for _, p in batch])
            
            # Set the end time for the next request to be just before the oldest tick fetched
            if batch:
                end_time = int(batch[0][0]) - 1
            logging.info(f"📊 Fetched {len(prices)} prices. Total ticks: {len(ticks)}")

        except Exception as e:
            logging.warning(f"⚠️ An error occurred during fetch: {e}")
            await asyncio.sleep(2) # Wait before retrying
    
    await api.disconnect()
    logging.info(f"🔌 Disconnected from Deriv API. Total ticks fetched: {len(ticks)}.")

    if len(ticks) < 305:
        logging.error(f"❌ Insufficient data: {len(ticks)} ticks fetched, but need at least 305.")
        return [], []

    x_data, y_data = [], []
    for i in range(len(ticks) - 305):
        x_data.append(ticks[i : i + 300])
        y_data.append(ticks[i + 300 : i + 305])

    logging.info(f"📊 Prepared {len(x_data)} training samples.")
    return x_data, y_data

async def main():
    """Main function to orchestrate the data fetching, model loading, and training."""
    x_data, y_data = await fetch_and_prepare_data()
    if not x_data or not y_data:
        logging.error("🚫 Training skipped due to data issues.")
        return

    model = load_model()
    if model:
        retrain_and_upload(model, x_data, y_data)
    else:
        logging.error("🚫 Model initialization failed.")

# 6. Run Training
if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())
