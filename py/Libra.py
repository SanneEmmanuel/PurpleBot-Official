#Libra Pro_V5.3 by Sanne Karibo (Optimized)
import os, requests, zipfile, torch, logging, math, time, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO
from torch.utils.data import DataLoader, TensorDataset
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
def convert_to_log_returns(prices):
    """Branchless vectorized conversion to log returns"""
    prices_arr = np.asarray(prices, dtype=np.float32)
    shifted = np.roll(prices_arr, 1)
    ratio = np.divide(prices_arr, shifted, out=np.ones_like(prices_arr), where=(shifted!=0)&(prices_arr!=0))
    return np.log(ratio, out=np.zeros_like(prices_arr), where=(ratio>0))

def convert_to_future_returns(input_prices, output_prices):
    """Branchless vectorized future log returns"""
    input_arr = np.asarray(input_prices, dtype=np.float32)
    output_arr = np.asarray(output_prices, dtype=np.float32)
    
    # First return (output[0] / input[-1])
    initial_ratio = np.divide(output_arr[0], input_arr[-1], 
                             out=np.ones(1), where=(input_arr[-1]!=0))
    initial_return = np.log(initial_ratio, out=np.zeros(1))
    
    # Subsequent returns (output[1:] / output[:-1])
    shifted_out = np.roll(output_arr, 1)
    ratio = np.divide(output_arr, shifted_out, out=np.ones_like(output_arr), 
                    where=(shifted_out!=0)&(output_arr!=0))
    subsequent_returns = np.log(ratio, out=np.zeros_like(output_arr))
    
    # Combine results
    returns = np.empty_like(output_arr)
    returns[0] = initial_return
    returns[1:] = subsequent_returns[1:]
    return returns

def decode_log_returns(last_price, log_returns):
    """Vectorized conversion of log returns to price predictions"""
    cum_returns = np.cumsum(log_returns)
    return (last_price * np.exp(cum_returns)).astype(np.float32).tolist()

# ========== 🧠 Enhanced Model Definition ==========
class TemporalDecayAttention(nn.Module):
    """Branchless attention with temporal decay"""
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
    """Parameter Efficient Fine-Tuning Adapter"""
    def __init__(self, layer, rank=8):
        super().__init__()
        self.layer = layer
        self.rank = rank
        
        # Freeze original parameters
        for param in self.layer.parameters():
            param.requires_grad = False
        
        # Add low-rank adapters
        self.lora_A = nn.Parameter(torch.randn(layer.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, layer.out_features))
        
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
        ).to(DEVICE)  # Direct device placement
        
        # Temporal Decay Attention
        self.attn = TemporalDecayAttention(
            embed_dim=hidden_size,
            num_heads=attn_heads,
            decay_factor=0.95
        ).to(DEVICE)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Main prediction layer (mean)
        self.fc_mean = nn.Linear(hidden_size, 5).to(DEVICE)
        
        # Confidence estimation layer (log variance)
        self.fc_var = nn.Linear(hidden_size, 5).to(DEVICE)
        
        logging.info(f"✅ LSTM: input_size={input_size}, hidden_size={hidden_size}, layers={lstm_layers}")
        logging.info(f"✅ TemporalDecayAttention: embed_dim={hidden_size}, heads={attn_heads}")
        logging.info(f"✅ Dual-head output: mean + variance estimation")

    def enable_peft(self, rank=8):
        if rank <= 0:
            logging.info("⚠️ PEFT rank is 0 or less. Skipping PEFT setup.")
            return self
        logging.info("🔧 Enabling PEFT adapters")
        self.fc_mean = PEFTAdapter(self.fc_mean, rank).to(DEVICE)
        self.fc_var = PEFTAdapter(self.fc_var, rank).to(DEVICE)
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
        
        return mean, log_var

# ========== ⬇️⬆️ Upload/Download Helpers ==========
def upload_model_with_retry(max_attempts=3):
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
                overwrite=True,
                use_filename=True
            )
            logging.info(f"✅ Upload successful: {result.get('secure_url')}")
            return True
        except Exception as e:
            wait_time = 2 ** attempt
            logging.warning(f"⚠️ Upload failed (attempt {attempt+1}): {str(e)}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    logging.error(f"❌ Upload failed after {max_attempts} attempts")
    return False

def download_model_from_cloudinary():
    """Download model from Cloudinary using signed URL"""
    logging.info("📥 Downloading model ZIP from Cloudinary...")
    
    # Configure Cloudinary
    cloudinary.config(
        cloud_name=CLOUD_NAME,
        api_key=API_KEY,
        api_secret=API_SECRET,
        secure=True
    )

    try:
        # Generate signed URL
        url, _ = cloudinary.utils.cloudinary_url(
            "model.pt.zip",
            resource_type='raw',
            type='upload',
            sign_url=True,
            expires_at=int(time.time()) + 600
        )

        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"✅ ZIP saved to {ZIP_PATH}")

        # Extract contents
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall("/tmp")
        logging.info("✅ Model extracted to /tmp")
    except Exception as e:
        logging.error(f"❌ Download failed: {e}")
        raise

# ========== 🚀 Load Model ==========
def load_model():
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
                logging.warning(f"⚠️ Download failed (attempt {attempt+1}): {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

    # Load existing model
    model = LibraModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# ========== 🔮 Enhanced Predict Function ==========
def predict_ticks(model, ticks):
    """Predict next 5 ticks with confidence estimation"""
    # Vectorized conversion to log returns
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
def retrain_and_upload(model, x_data, y_data, epochs=50, patience=5, peft_rank=8):
    """Enhanced training with early stopping and metrics"""
    # Preallocate arrays for vectorized processing
    num_samples = len(x_data)
    x_returns = np.zeros((num_samples, 300), dtype=np.float32)
    y_returns = np.zeros((num_samples, 5), dtype=np.float32)
    
    # Vectorized data conversion
    for i in range(num_samples):
        x_returns[i] = convert_to_log_returns(x_data[i])
        y_returns[i] = convert_to_future_returns(x_data[i], y_data[i])
    
    # Split into train and validation (80/20)
    split_idx = int(0.8 * num_samples)
    x_train, x_val = x_returns[:split_idx], x_returns[split_idx:]
    y_train, y_val = y_returns[:split_idx], y_returns[split_idx:]
    
    # Convert to tensors
    x_train_t = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    
    # Create datasets
    train_dataset = TensorDataset(x_train_t, y_train_t)
    val_dataset = TensorDataset(x_val_t, y_val_t)
    
    # Create dataloaders with optimized settings
    use_gpu = torch.cuda.is_available()
    num_workers = 4 if use_gpu else 0
    persistent = use_gpu and num_workers > 0
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        pin_memory=use_gpu,
        num_workers=num_workers,
        persistent_workers=persistent
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32,
        pin_memory=use_gpu,
        num_workers=num_workers,
        persistent_workers=persistent
    )
    
    # Enable PEFT
    if peft_rank > 0:
        model = model.enable_peft(peft_rank)
    model.train()
    
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
        # Training phase
        model.train()
        train_loss = 0
        correct_direction = 0
        total = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=use_gpu):
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
            total += yb.numel()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
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
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"💾 Saved checkpoint: {checkpoint_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"⏳ No improvement: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                logging.info(f"🛑 Early stopping at epoch {epoch+1}")
                break
    
    # Save best model
    final_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    torch.save(model.state_dict(), final_model_path)
    torch.save(model.state_dict(), MODEL_PATH)
    logging.info(f"🏆 Best model saved: {final_model_path}")
    
    # Upload to cloud
    if upload_model_with_retry():
        logging.info("🚀 Retrained model uploaded to Cloudinary")
    else:
        logging.error("❌ Failed to upload retrained model")
    
    return model
