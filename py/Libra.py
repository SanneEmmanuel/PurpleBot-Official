# Libra Pro_V5.4 by Sanne Karibo (Cython-Optimized with CuPy Support)
import os
import requests
import zipfile
import torch
import logging
import math
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict
import cython

# Conditional GPU imports
if torch.cuda.is_available():
    try:
        import cupy as cp
        USE_CUPY = True
    except ImportError:
        logging.warning("CuPy not installed, falling back to NumPy")
        USE_CUPY = False
else:
    USE_CUPY = False

# Import cloudinary only if credentials are present
CLOUDINARY_AVAILABLE = all(os.getenv(f"CLOUDINARY_{var}") for var in ["CLOUD_NAME", "API_KEY", "API_SECRET"])
if CLOUDINARY_AVAILABLE:
    import cloudinary
    import cloudinary.uploader
    import cloudinary.api

# ========== 🔐 Cloudinary Config ==========
CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "dj4bwntzb")
API_KEY = os.getenv("CLOUDINARY_API_KEY", "354656419316393")
API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "M-Trl9ltKDHyo1dIP2AaLOG-WPM")
MODEL_URL = f"https://res.cloudinary.com/{CLOUD_NAME}/raw/upload/v1/model.pt.zip" if CLOUDINARY_AVAILABLE else ""
MODEL_PATH = "/tmp/model.pt"
ZIP_PATH = "/tmp/model.pt.zip"
CHECKPOINT_DIR = "/tmp/checkpoints"

# ========== ⚙️ Device Setup ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
logging.info(f"🔌 Using device: {DEVICE}")

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ========== 🚀 Optimized Vectorized Functions ==========
def _get_array_module():
    return cp if USE_CUPY and torch.cuda.is_available() else np

def convert_to_log_returns(prices: List[float]) -> np.ndarray:
    """Branchless vectorized conversion to log returns"""
    xp = _get_array_module()
    prices_arr = xp.asarray(prices, dtype=xp.float32)
    shifted = xp.roll(prices_arr, 1)
    shifted = xp.where(xp.arange(len(prices_arr)) == 0, 0, shifted)
    
    # Branchless division and log
    ratio = xp.divide(prices_arr, shifted, 
                      out=xp.ones_like(prices_arr), 
                      where=shifted != 0)
    return xp.log(ratio, 
                  out=xp.zeros_like(prices_arr), 
                  where=ratio > 0)

def convert_to_future_returns(input_prices: List[float], output_prices: List[float]) -> np.ndarray:
    """Branchless vectorized future log returns"""
    xp = _get_array_module()
    input_arr = xp.asarray(input_prices, dtype=xp.float32)
    output_arr = xp.asarray(output_prices, dtype=xp.float32)
    
    # Branchless chaining and calculation
    full_price_series = xp.concatenate((xp.array([input_arr[-1]]), output_arr))
    shifted = xp.roll(full_price_series, 1)
    
    ratio = xp.divide(full_price_series, shifted,
                      out=xp.ones_like(full_price_series),
                      where=shifted != 0)
    log_returns = xp.log(ratio, 
                         out=xp.zeros_like(full_price_series),
                         where=ratio > 0)
    
    return log_returns[1:]

def decode_log_returns(last_price: float, log_returns: np.ndarray) -> List[float]:
    """Vectorized conversion of log returns to price predictions"""
    xp = _get_array_module()
    cum_log_returns = xp.cumsum(log_returns)
    return (last_price * xp.exp(cum_log_returns)).astype(xp.float32).tolist()

# ========== 🧠 Enhanced Model Definition ==========
class TemporalDecayAttention(nn.Module):
    """Branchless attention with temporal decay"""

    def __init__(self, embed_dim: int, num_heads: int, decay_factor: float = 0.95):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.decay_factor = decay_factor
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Precomputed indices for branchless mask
        indices = torch.arange(seq_len, device=x.device)
        time_diffs = torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0))
        decay_mask = (self.decay_factor ** time_diffs).unsqueeze(0).unsqueeze(0)

        # Project inputs
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply decay mask
        attn_scores = attn_scores * decay_mask

        # Branchless softmax and output
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(attn_output)

class PEFTAdapter(nn.Module):
    """Parameter Efficient Fine-Tuning Adapter using LoRA"""
    def __init__(self, layer: nn.Linear, rank: int = 8):
        super().__init__()
        self.layer = layer
        self.rank = rank

        # Freeze original parameters
        for param in self.layer.parameters():
            param.requires_grad = False

        # Add low-rank adapters
        self.lora_A = nn.Parameter(torch.randn(layer.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, layer.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.layer(x)
        lora_output = (x @ self.lora_A @ self.lora_B)
        return base_output + lora_output

class LibraModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, 
                 lstm_layers: int = 2, dropout: float = 0.2,
                 attn_heads: int = 4):
        super().__init__()
        logging.info("🧠 Initializing Enhanced LibraModel...")

        # LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=True
        )

        # Temporal Decay Attention
        self.attn = TemporalDecayAttention(
            embed_dim=hidden_size,
            num_heads=attn_heads,
            decay_factor=0.95
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Dual-head output
        self.fc_mean = nn.Linear(hidden_size, 5)
        self.fc_var = nn.Linear(hidden_size, 5)

    def enable_peft(self, rank: int = 8):
        if rank <= 0:
            return self
        self.fc_mean = PEFTAdapter(self.fc_mean, rank)
        self.fc_var = PEFTAdapter(self.fc_var, rank)
        return self

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # LSTM processing
        lstm_out, _ = self.lstm(x)

        # Attention with temporal decay
        attn_out = self.attn(lstm_out)

        # Last timestep features
        context = self.dropout(attn_out[:, -1, :])

        # Dual outputs
        mean = self.fc_mean(context)
        log_var = self.fc_var(context)

        return mean, log_var

# ========== ⬇️⬆️ Upload/Download Helpers ==========
def upload_model_with_retry(max_attempts: int = 3) -> bool:
    if not CLOUDINARY_AVAILABLE:
        logging.warning("Skipping upload - Cloudinary credentials missing")
        return False

    try:
        # Configure Cloudinary
        cloudinary.config(
            cloud_name=CLOUD_NAME,
            api_key=API_KEY,
            api_secret=API_SECRET,
            secure=True
        )
        
        # Zip and upload
        with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(MODEL_PATH, arcname="model.pt")

        for attempt in range(max_attempts):
            try:
                result = cloudinary.uploader.upload(
                    ZIP_PATH,
                    resource_type='raw',
                    public_id="model.pt.zip",
                    overwrite=True
                )
                logging.info(f"✅ Upload successful: {result.get('secure_url')}")
                return True
            except Exception as e:
                wait_time = 2 ** attempt
                logging.warning(f"⚠️ Upload failed (attempt {attempt+1}): {str(e)}")
                time.sleep(wait_time)
                
        return False
    except Exception as e:
        logging.error(f"❌ Upload failed: {str(e)}")
        return False

def download_model_from_cloudinary() -> None:
    if not CLOUDINARY_AVAILABLE:
        raise RuntimeError("Cloudinary credentials not available")
    
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=10)
        response.raise_for_status()
        
        with open(ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(MODEL_PATH))
            
    except Exception as e:
        logging.error(f"❌ Download failed: {e}")
        raise

# ========== 🚀 Load Model ==========
def load_model() -> LibraModel:
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = LibraModel().to(DEVICE)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model = torch.compile(model)
            model.eval()
            logging.info("🥂 Loaded existing model")
            return model
        except Exception as e:
            logging.warning(f"⚠️ Model loading failed: {str(e)}")
    
    # Download or create new model
    for attempt in range(3):
        try:
            download_model_from_cloudinary()
            model = LibraModel().to(DEVICE)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model = torch.compile(model)
            model.eval()
            logging.info("🥂 Downloaded and loaded model")
            return model
        except Exception as e:
            if attempt == 2:
                logging.info("🆕 Creating new model")
                model = LibraModel().to(DEVICE)
                torch.save(model.state_dict(), MODEL_PATH)
                if upload_model_with_retry():
                    logging.info("📤 Initial model uploaded")
                return model
            wait_time = 2 ** attempt
            logging.warning(f"⚠️ Download failed (attempt {attempt+1}): {str(e)}")
            time.sleep(wait_time)
    
    return LibraModel().to(DEVICE)

# ========== 🔮 Enhanced Predict Function ==========
def predict_ticks(model: LibraModel, ticks: List[float]) -> Dict:
    model.eval()
    xp = _get_array_module()
    
    # Branchless conversion
    returns = convert_to_log_returns(ticks)
    tensor_data = xp.asarray(returns) if USE_CUPY else np.asarray(returns)
    x = torch.tensor(tensor_data, dtype=torch.float32, device=DEVICE).view(1, 300, 1)

    with torch.inference_mode():
        mean, log_var = model(x)

    # Convert to numpy/cupy arrays
    mean = mean.squeeze().cpu().numpy()
    log_var = log_var.squeeze().cpu().numpy()

    # Branchless confidence calculation
    std = xp.exp(0.5 * log_var)
    ci_low = mean - 1.96 * std
    ci_high = mean + 1.96 * std

    # Vectorized decoding
    last_price = ticks[-1]
    predicted_prices = decode_log_returns(last_price, mean)
    ci_low_prices = decode_log_returns(last_price, ci_low)
    ci_high_prices = decode_log_returns(last_price, ci_high)
    confidence = 1 / (1 + std)

    return {
        "prices": predicted_prices,
        "confidence": confidence.tolist(),
        "ci_low": ci_low_prices,
        "ci_high": ci_high_prices
    }

# ========== 🔁 Enhanced Training Function ==========
def retrain_and_upload(model: LibraModel, 
                      x_data: List[List[float]], 
                      y_data: List[List[float]], 
                      epochs: int = 50,
                      patience: int = 5, 
                      peft_rank: int = 8) -> LibraModel:
    xp = _get_array_module()
    num_samples = len(x_data)
    
    # Preallocate with correct array type
    x_returns = xp.zeros((num_samples, 300), dtype=xp.float32)
    y_returns = xp.zeros((num_samples, 5), dtype=xp.float32)

    # Vectorized conversion
    for i in range(num_samples):
        x_returns[i] = convert_to_log_returns(x_data[i])
        y_returns[i] = convert_to_future_returns(x_data[i], y_data[i])

    # Split dataset
    split_idx = int(0.8 * num_samples)
    x_train, x_val = x_returns[:split_idx], x_returns[split_idx:]
    y_train, y_val = y_returns[:split_idx], y_returns[split_idx:]

    # Convert to tensors
    x_train_t = torch.tensor(x_train.get() if USE_CUPY else x_train, 
                             dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train.get() if USE_CUPY else y_train, 
                             dtype=torch.float32)
    x_val_t = torch.tensor(x_val.get() if USE_CUPY else x_val, 
                           dtype=torch.float32).unsqueeze(-1)
    y_val_t = torch.tensor(y_val.get() if USE_CUPY else y_val, 
                           dtype=torch.float32)

    # Create datasets
    train_dataset = TensorDataset(x_train_t, y_train_t)
    val_dataset = TensorDataset(x_val_t, y_val_t)

    # Dataloaders
    num_workers = min(os.cpu_count() or 1, 4)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                             pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=32, 
                           pin_memory=True, num_workers=num_workers)

    # Enable PEFT
    if peft_rank > 0:
        model = model.enable_peft(rank=peft_rank)
    model.to(DEVICE).train()
    
    # Optimization setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                mean, log_var = model(xb)
                var = log_var.exp()
                loss = F.gaussian_nll_loss(mean, yb, var)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * xb.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                mean, log_var = model(xb)
                var = log_var.exp()
                val_loss += F.gaussian_nll_loss(mean, yb, var).item() * xb.size(0)
        
        # Epoch metrics
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        # Check improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(f"🛑 Early stopping at epoch {epoch+1}")
                break

    # Final upload
    model.eval()
    if upload_model_with_retry():
        logging.info("🚀 Model uploaded successfully")
    return model
