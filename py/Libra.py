# Libra Pro_V5.3 by Sanne Karibo (Optimized and Future-Proofed for 2025)
import os
import requests
import zipfile
import torch
import logging
import math
import time
import jax.numpy as jnp
from jax import jit
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict

# Note: For production systems, consider a dedicated configuration library
# like Pydantic for managing settings from environment variables or files.
import cloudinary
import cloudinary.uploader
import cloudinary.api

# ========== 🔐 Cloudinary Config ==========
CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "dj4bwntzb")
API_KEY = os.getenv("CLOUDINARY_API_KEY", "354656419316393")
API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "M-Trl9ltKDHyo1dIP2AaLOG-WPM")
MODEL_URL = f"https://res.cloudinary.com/{CLOUD_NAME}/raw/upload/v1/model.pt.zip"
MODEL_PATH = "/tmp/model.pt"
ZIP_PATH = "/tmp/model.pt.zip"
CHECKPOINT_DIR = "/tmp/checkpoints"

# ========== ⚙️ Device Setup ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
logging.info(f"🔌 Using device: {DEVICE}")

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ==========  Vectorized Helper Functions ==========
@jit
def convert_to_log_returns(prices: List[float]) -> jnp.ndarray:
    """Branchless vectorized conversion to log returns"""
    prices_arr = jnp.asarray(prices, dtype=jnp.float32)
    # Avoid division by zero by ensuring shifted array is non-zero at division points
    shifted = jnp.roll(prices_arr, 1)
    shifted = shifted.at[0].set(0)  # First element has no prior price, log return is 0

    ratio = jnp.divide(prices_arr, shifted, out=jnp.ones_like(prices_arr), where=(shifted != 0))
    # Ensure log is only taken for positive ratios
    return jnp.log(ratio, out=jnp.zeros_like(prices_arr), where=(ratio > 0))


@jit
def convert_to_future_returns(input_prices: List[float], output_prices: List[float]) -> jnp.ndarray:
    """Branchless vectorized future log returns"""
    input_arr = jnp.asarray(input_prices, dtype=jnp.float32)
    output_arr = jnp.asarray(output_prices, dtype=jnp.float32)

    # Chain the last input price with the output prices for continuous calculation
    full_price_series = jnp.concatenate((jnp.array([input_arr[-1]]), output_arr))

    # Calculate log returns on the full series
    shifted = jnp.roll(full_price_series, 1)
    ratio = jnp.divide(full_price_series, shifted, out=jnp.ones_like(full_price_series), where=(shifted != 0))
    log_returns = jnp.log(ratio, out=jnp.zeros_like(full_price_series), where=(ratio > 0))

    return log_returns[1:]  # Return the 5 future log returns


@jit
def decode_log_returns(last_price: float, log_returns: jnp.ndarray) -> List[float]:
    """Vectorized conversion of log returns to price predictions"""
    # Cumulative sum of log returns is the log of the price ratio
    cum_log_returns = jnp.cumsum(log_returns)
    # Price = last_price * exp(cumulative_log_return)
    return (last_price * jnp.exp(cum_log_returns)).astype(jnp.float32).tolist()


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

        # Dynamic decay mask generation (branchless)
        indices = torch.arange(seq_len, device=x.device)
        time_diffs = torch.abs(indices.view(1, -1) - indices.view(-1, 1))
        decay_mask = (self.decay_factor ** time_diffs).unsqueeze(0).unsqueeze(0)

        # Project inputs
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply decay mask
        attn_scores = attn_scores * decay_mask

        # Get weighted values
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)

        # Combine heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class PEFTAdapter(nn.Module):
    """
    Parameter Efficient Fine-Tuning (PEFT) Adapter using Low-Rank Adaptation (LoRA).
    Note: For production, consider using established libraries like Hugging Face's 'peft'
    for more robust and feature-rich implementations.
    """

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
    def __init__(self, input_size: int = 1, hidden_size: int = 64, lstm_layers: int = 2, dropout: float = 0.2,
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

        # Main prediction layer (mean)
        self.fc_mean = nn.Linear(hidden_size, 5)

        # Confidence estimation layer (log variance)
        self.fc_var = nn.Linear(hidden_size, 5)

        logging.info(f"✅ LSTM: input_size={input_size}, hidden_size={hidden_size}, layers={lstm_layers}")
        logging.info(f"✅ TemporalDecayAttention: embed_dim={hidden_size}, heads={attn_heads}")
        logging.info(f"✅ Dual-head output: mean + variance estimation")

    def enable_peft(self, rank: int = 8):
        if rank <= 0:
            logging.info("⚠️ PEFT rank is 0 or less. Skipping PEFT setup.")
            return self
        logging.info(f"🔧 Enabling PEFT adapters with rank={rank}")
        # Note: Moving PEFTAdapter to the correct device is handled when the parent model is moved.
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

        # Dual outputs: mean and log variance
        mean = self.fc_mean(context)
        log_var = self.fc_var(context)

        return mean, log_var


# ========== ⬇️⬆️ Upload/Download Helpers ==========
def upload_model_with_retry(max_attempts: int = 3):
    """Upload model to Cloudinary with retry mechanism"""
    logging.info("📦 Zipping model for upload...")
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(MODEL_PATH, arcname="model.pt")

    # Configure Cloudinary
    cloudinary.config(
        cloud_name=CLOUD_NAME,
        api_key=API_KEY,
        api_secret=API_SECRET,
        secure=True
    )

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
            logging.warning(f"⚠️ Upload failed (attempt {attempt + 1}): {str(e)}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    logging.error(f"❌ Upload failed after {max_attempts} attempts")
    return False


def download_model_from_cloudinary():
    """Download model from Cloudinary"""
    logging.info("📥 Downloading model ZIP from Cloudinary...")

    try:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()

        with open(ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"✅ ZIP saved to {ZIP_PATH}")

        # Extract contents
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall("/tmp")
        logging.info(f"✅ Model extracted to {MODEL_PATH}")
    except Exception as e:
        logging.error(f"❌ Download failed: {e}")
        raise


# ========== 🚀 Load Model ==========
def load_model() -> LibraModel:
    """Load model with retry and fallback"""
    if not os.path.exists(MODEL_PATH):
        for attempt in range(3):
            try:
                download_model_from_cloudinary()
                print("🥂 Resuming With Previous Trained Model")
                break
            except Exception as e:
                if attempt == 2:
                    logging.error("❌ Download failed 3 times. Creating fresh model...")
                    model = LibraModel().to(DEVICE)
                    torch.save(model.state_dict(), MODEL_PATH)
                    print("😔 Saved New Model")
                    if upload_model_with_retry():
                        logging.info("🆕 Fresh model uploaded to Cloudinary")
                    return model
                wait_time = 2 ** attempt
                logging.warning(f"⚠️ Download failed (attempt {attempt + 1}): {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

    # Load existing model
    model = LibraModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = torch.compile(model)
    model.eval()
    return model


# ========== 🔮 Enhanced Predict Function ==========
def predict_ticks(model: LibraModel, ticks: List[float]) -> Dict:
    """Predict next 5 ticks with confidence estimation"""
    model.eval()  # Ensure model is in evaluation mode

    # Vectorized conversion to log returns
    returns = convert_to_log_returns(ticks)
    # The conversion to numpy is necessary because JAX arrays don't directly convert to torch tensors
    x = torch.tensor(np.array(returns), dtype=torch.float32).view(1, 300, 1).to(DEVICE)

    # Use torch.inference_mode() for better performance than torch.no_grad()
    with torch.inference_mode():
        mean, log_var = model(x)

    # Convert to numpy arrays
    mean = mean.squeeze().cpu().numpy()
    log_var = log_var.squeeze().cpu().numpy()

    # Calculate confidence intervals (95% CI)
    std = jnp.exp(0.5 * log_var)
    ci_low = mean - 1.96 * std
    ci_high = mean + 1.96 * std

    # Vectorized decoding
    last_price = ticks[-1]
    predicted_prices = decode_log_returns(last_price, mean)
    ci_low_prices = decode_log_returns(last_price, ci_low)
    ci_high_prices = decode_log_returns(last_price, ci_high)

    # Calculate confidence scores
    confidence = 1 / (1 + std)

    logging.info(f"📈 Predicted prices: {predicted_prices}")
    logging.info(f"🛡️ Confidence: {confidence.tolist()}")

    return {
        "prices": predicted_prices,
        "confidence": confidence.tolist(),
        "ci_low": ci_low_prices,
        "ci_high": ci_high_prices
    }


# ========== 🔁 Enhanced Training Function ==========
def retrain_and_upload(model: LibraModel, x_data: List[List[float]], y_data: List[List[float]], epochs: int = 50,
                       patience: int = 5, peft_rank: int = 8) -> LibraModel:
    """Enhanced training with early stopping, metrics, and PEFT."""
    # Preallocate arrays for vectorized processing
    num_samples = len(x_data)
    # Use jnp.zeros for consistency with JAX functions
    x_returns = jnp.zeros((num_samples, 300), dtype=jnp.float32)
    y_returns = jnp.zeros((num_samples, 5), dtype=jnp.float32)

    # Vectorized data conversion
    for i in range(num_samples):
        x_returns = x_returns.at[i].set(convert_to_log_returns(x_data[i]))
        y_returns = y_returns.at[i].set(convert_to_future_returns(x_data[i], y_data[i]))

    # Split into train and validation (80/20)
    split_idx = int(0.8 * num_samples)
    x_train, x_val = x_returns[:split_idx], x_returns[split_idx:]
    y_train, y_val = y_returns[:split_idx], y_returns[split_idx:]

    # Convert to tensors - JAX arrays must be converted to numpy first
    x_train_t = torch.tensor(np.array(x_train), dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(np.array(y_train), dtype=torch.float32)
    x_val_t = torch.tensor(np.array(x_val), dtype=torch.float32).unsqueeze(-1)
    y_val_t = torch.tensor(np.array(y_val), dtype=torch.float32)

    # Safety check for data shape mismatches before creating TensorDataset
    assert len(x_train_t) == len(y_train_t), f"❌ Training data/label mismatch: {len(x_train_t)} vs {len(y_train_t)}"
    assert len(x_val_t) == len(y_val_t), f"❌ Validation data/label mismatch: {len(x_val_t)} vs {len(y_val_t)}"

    # Create datasets
    train_dataset = TensorDataset(x_train_t, y_train_t)
    val_dataset = TensorDataset(x_val_t, y_val_t)

    # Create dataloaders with optimized settings
    use_gpu = torch.cuda.is_available()
    # Set num_workers safely; os.cpu_count() can be None.
    max_workers = os.cpu_count() or 1
    num_workers = min(max_workers, 4)  # Use up to 4 workers if available

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=use_gpu,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        pin_memory=use_gpu,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0)
    )

    # Enable PEFT
    if peft_rank > 0:
        model = model.enable_peft(rank=peft_rank)
    model.to(DEVICE) # No need to call .train() yet, will be done in loop

    # Loss function (Gaussian NLL)
    loss_fn = F.gaussian_nll_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # Training variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_gpu)

    # Training loop
    for epoch in range(epochs):
        # --- Training phase ---
        model.train()
        train_loss = 0
        correct_direction = 0
        total_points = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_gpu):
                mean, log_var = model(xb)
                var = log_var.exp()
                loss = loss_fn(mean, yb, var)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Metrics
            train_loss += loss.item() * xb.size(0)
            pred_direction = torch.sign(mean)
            true_direction = torch.sign(yb)
            correct_direction += (pred_direction == true_direction).sum().item()
            total_points += yb.numel()

        # --- Validation phase ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.inference_mode():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                mean, log_var = model(xb)
                var = log_var.exp()
                val_loss += loss_fn(mean, yb, var).item() * xb.size(0)

                # Directional accuracy
                pred_direction = torch.sign(mean)
                true_direction = torch.sign(yb)
                val_correct += (pred_direction == true_direction).sum().item()
                val_total += yb.numel()

        # Calculate epoch metrics
        train_loss /= len(train_loader.dataset)
        train_acc = correct_direction / total_points
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

        # Update scheduler
        scheduler.step(val_loss)

        # Log metrics
        logging.info(f"📚 Epoch {epoch + 1}/{epochs}:")
        logging.info(f"  Train Loss: {train_loss:.6f} | Dir Acc: {train_acc:.4f}")
        logging.info(f"  Val Loss:   {val_loss:.6f} | Dir Acc: {val_acc:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_val_loss_{val_loss:.4f}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            torch.save(model.state_dict(), MODEL_PATH)
            logging.info(f"💾 New best model saved: {checkpoint_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"⏳ No improvement: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                logging.info(f"🛑 Early stopping at epoch {epoch + 1}")
                break

    logging.info(f"🏆 Training finished. Best validation loss: {best_val_loss:.6f}")

    # Upload the best model found during training
    if upload_model_with_retry():
        logging.info("🚀 Best retrained model uploaded to Cloudinary")
    else:
        logging.error("❌ Failed to upload retrained model")

    return model
