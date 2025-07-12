#Libra Pro_V5 by Sanne Karibo
import os, requests, zipfile, torch, logging, math, time, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
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

# ==========  Helper Functions ==========
def convert_to_log_returns(prices):
    """Convert price sequence to log returns with initial zero"""
    returns = [0.0]  # Initial zero return
    for i in range(1, len(prices)):
        returns.append(math.log(prices[i] / prices[i-1]))
    return returns

def convert_to_future_returns(input_prices, output_prices):
    """Convert input/output prices to future log returns"""
    returns = []
    # First return connects input to output
    returns.append(math.log(output_prices[0] / input_prices[-1]))
    # Subsequent returns between output prices
    for i in range(1, len(output_prices)):
        returns.append(math.log(output_prices[i] / output_prices[i-1]))
    return returns

def decode_log_returns(last_price, log_returns):
    """Convert log returns back to price predictions"""
    prices = [last_price]
    for r in log_returns:
        prices.append(prices[-1] * math.exp(r))
    return prices[1:]  # Return only future prices

# ========== 🧠 Enhanced Model Definition ==========
class TemporalDecayAttention(nn.Module):
    """Attention with temporal decay weighting"""
    def __init__(self, embed_dim, num_heads, decay_factor=0.95):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.decay_factor = decay_factor
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project inputs
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Create temporal decay mask
        time_diffs = torch.abs(torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0))
        decay_mask = self.decay_factor ** time_diffs
        decay_mask = decay_mask.to(DEVICE).unsqueeze(0).unsqueeze(0)
        
        # Apply decay mask to attention scores
        attn_scores = attn_scores * decay_mask
        
        # Apply softmax and get weighted values
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        
        # Combine heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)

class PEFTAdapter(nn.Module):
    """Parameter Efficient Fine-Tuning Adapter"""
    def __init__(self, layer, rank=8):
        super().__init__()
        self.layer = layer
        self.rank = rank
        
        # Freeze original parameters
        for param in self.layer.parameters():
            param.requires_grad = False
        
        # Add low-rank adapters
        in_features = layer.in_features
        out_features = layer.out_features
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
    def forward(self, x):
        base_output = self.layer(x)
        lora_output = x @ self.lora_A @ self.lora_B
        return base_output + lora_output

class LibraModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, lstm_layers=2, dropout=0.2, attn_heads=4):
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

    def enable_peft(self, rank=8):
        if rank <= 0:
           logging.info("⚠️ PEFT rank is 0 or less. Skipping PEFT setup.")
           return self
        logging.info("🔧 Enabling PEFT adapters")
        self.fc_mean = PEFTAdapter(self.fc_mean, rank)
        self.fc_var = PEFTAdapter(self.fc_var, rank)
        return self


    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Attention with temporal decay
        attn_out = self.attn(lstm_out)
        
        # Last timestep features
        context = self.dropout(attn_out[:, -1, :])
        
        # Dual outputs: mean and log variance
        mean = self.fc_mean(context)
        log_var = self.fc_var(context)
        
        # Return both predictions and uncertainty
        return mean, log_var

# ========== ⬇️⬆️ Upload/Download Helpers ==========
def upload_model_with_retry(max_attempts=3):
    """Upload model to Cloudinary with retry mechanism using Cloudinary SDK"""
    logging.info("📦 Zipping model for upload...")
    with zipfile.ZipFile(ZIP_PATH, 'w') as zipf:
        zipf.write(MODEL_PATH, arcname="model.pt")

    # Configure Cloudinary if not already set
    cloudinary.config(
        cloud_name=CLOUD_NAME,
        api_key=API_KEY,
        api_secret=API_SECRET,
        secure=True
    )

    attempts = 0
    while attempts < max_attempts:
        try:
            result = cloudinary.uploader.upload(
                ZIP_PATH,
                resource_type='raw',
                public_id="model.pt",
                overwrite=True,
                use_filename=True
            )
            logging.info(f"✅ Upload successful: {result.get('secure_url')}")
            return True
        except Exception as e:
            attempts += 1
            if attempts < max_attempts:
                wait_time = 2 ** attempts
                logging.warning(f"⚠️ Upload failed (attempt {attempts}): {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.error(f"❌ Upload failed after {max_attempts} attempts: {str(e)}")
                return False


def download_model_from_cloudinary():
    """Download model from Cloudinary using signed URL and extract it"""
    logging.info("📥 Downloading model ZIP from Cloudinary...")

    # Ensure Cloudinary is configured
    cloudinary.config(
        cloud_name=CLOUD_NAME,
        api_key=API_KEY,
        api_secret=API_SECRET,
        secure=True
    )

    try:
        # Generate signed URL valid for 10 minutes
        url, options = cloudinary.utils.cloudinary_url(
            "model.pt",
            resource_type='raw',
            type='upload',
            sign_url=True,
            expires_at=int(time.time()) + 600
        )

        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download model: HTTP {response.status_code}")

        with open(ZIP_PATH, "wb") as f:
            f.write(response.content)
        logging.info(f"✅ ZIP saved to {ZIP_PATH}")

        # Extract contents to /tmp
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall("/tmp")
        logging.info("✅ Model extracted to /tmp")
    except Exception as e:
        logging.error(f"❌ Download failed: {e}")
        raise


# ========== 🚀 Load Model ==========
def load_model():
    """Load model with retry and fallback to new model"""
    # Try to download if local model doesn't exist
    if not os.path.exists(MODEL_PATH):
        attempts = 0
        while attempts < 3:
            try:
                download_model_from_cloudinary()
                break
            except Exception as e:
                attempts += 1
                if attempts == 3:
                    logging.error("❌ Download failed 3 times. Creating fresh model...")
                    model = LibraModel().to(DEVICE)
                    torch.save(model.state_dict(), MODEL_PATH)
                    if upload_model_with_retry():
                        logging.info("🆕 Fresh model uploaded to Cloudinary")
                    return model
                logging.warning(f"⚠️ Download failed (attempt {attempts}): {str(e)}. Retrying...")
                time.sleep(2 ** attempts)

    # Load existing model
    model = LibraModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    logging.info("✅ Model loaded from disk.")
    return model

# ========== 🔮 Enhanced Predict Function ==========
def predict_ticks(model, ticks):
    """Predict next 5 ticks with confidence estimation"""
    # Convert prices to log returns
    returns = convert_to_log_returns(ticks)
    x = torch.tensor(returns, dtype=torch.float32).view(1, 300, 1).to(DEVICE)
    
    with torch.no_grad():
        mean, log_var = model(x)
    
    # Convert to numpy arrays
    mean = mean.squeeze().cpu().numpy()
    log_var = log_var.squeeze().cpu().numpy()
    
    # Calculate confidence intervals (95% CI)
    std = np.exp(0.5 * log_var)
    ci_low = mean - 1.96 * std
    ci_high = mean + 1.96 * std
    
    # Decode log returns to price predictions
    last_price = ticks[-1]
    predicted_prices = decode_log_returns(last_price, mean)
    ci_low_prices = decode_log_returns(last_price, ci_low)
    ci_high_prices = decode_log_returns(last_price, ci_high)
    
    # Calculate confidence scores (1 - coefficient of variation)
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
def retrain_and_upload(model, x_data, y_data, epochs=50, patience=5, peft_rank=8):
    """Enhanced training with early stopping, metrics, and sparse checkpointing"""
    # Prepare data
    logging.info("🔄 Converting data to log returns...")
    x_returns, y_returns = [], []
    
    for i in range(len(x_data)):
        x_returns.append(convert_to_log_returns(x_data[i]))
        y_returns.append(convert_to_future_returns(x_data[i], y_data[i]))
    
    # Split into train and validation (80/20)
    split_idx = int(0.8 * len(x_returns))
    x_train, x_val = x_returns[:split_idx], x_returns[split_idx:]
    y_train, y_val = y_returns[:split_idx], y_returns[split_idx:]
    
    # Convert to tensors
    x_train_t = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).unsqueeze(-1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    
    # Create datasets
    train_dataset = TensorDataset(x_train_t, y_train_t)
    val_dataset = TensorDataset(x_val_t, y_val_t)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Enable PEFT
    if peft_rank > 0:
    model = model.enable_peft(peft_rank)
    model.train().to(DEVICE)
    
    # Loss function (Gaussian NLL)
    def loss_fn(mean, log_var, target):
        var = torch.exp(log_var)
        return 0.5 * (log_var + ((target - mean) ** 2) / var).mean()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # Training variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    training_history = []
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct_direction = 0
        total = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            mean, log_var = model(xb)
            
            # Calculate loss and backpropagate
            loss = loss_fn(mean, log_var, yb)
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item() * xb.size(0)
            
            # Directional accuracy
            pred_direction = torch.sign(mean)
            true_direction = torch.sign(yb)
            correct_direction += (pred_direction == true_direction).sum().item()
            total += yb.numel()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                mean, log_var = model(xb)
                val_loss += loss_fn(mean, log_var, yb).item() * xb.size(0)
                
                # Directional accuracy
                pred_direction = torch.sign(mean)
                true_direction = torch.sign(yb)
                val_correct += (pred_direction == true_direction).sum().item()
                val_total += yb.numel()
        
        # Calculate epoch metrics
        train_loss /= len(train_loader.dataset)
        train_acc = correct_direction / total
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        logging.info(f"📚 Epoch {epoch+1}/{epochs}:")
        logging.info(f"  Train Loss: {train_loss:.6f} | Dir Acc: {train_acc:.4f}")
        logging.info(f"  Val Loss:   {val_loss:.6f} | Dir Acc: {val_acc:.4f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # Save sparse checkpoint (every 5 improvements)
            if epoch % 5 == 0:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                logging.info(f"💾 Saved checkpoint: {checkpoint_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"⏳ No improvement: {epochs_no_improve}/{patience}")
            
            # Early stopping check
            if epochs_no_improve >= patience:
                logging.info(f"🛑 Early stopping at epoch {epoch+1}")
                break
    
    # Save best model
    final_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"🏆 Best model saved: {final_model_path}")
    
    # Copy to main model path
    torch.save(model.state_dict(), MODEL_PATH)
    
    # Upload to cloud
    if upload_model_with_retry():
        logging.info("🚀 Retrained model uploaded to Cloudinary")
    else:
        logging.error("❌ Failed to upload retrained model")
    
    return model
