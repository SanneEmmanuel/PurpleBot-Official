# Libra Pro_V5 (Optimized, TorchScript-Free)
import os, requests, zipfile, torch, logging, math, time, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO
from torch.utils.data import DataLoader, TensorDataset
import cloudinary, cloudinary.uploader, cloudinary.api

CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "dj4bwntzb")
API_KEY = os.getenv("CLOUDINARY_API_KEY", "354656419316393")
API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "M-Trl9ltKDHyo1dIP2AaLOG-WPM")
MODEL_URL = f"https://res.cloudinary.com/{CLOUD_NAME}/raw/upload/v1/model.pt.zip"
MODEL_PATH = "/tmp/model.pt"
ZIP_PATH = "/tmp/model.pt.zip"
CHECKPOINT_DIR = "/tmp/checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logging.info(f"🔌 Using device: {DEVICE}")

def convert_to_log_returns(prices: np.ndarray) -> np.ndarray:
    """Convert price sequence to log returns with initial zero using NumPy for speed."""
    prices_np = np.asarray(prices, dtype=np.float64)
    log_returns = np.zeros_like(prices_np, dtype=np.float64)
    if len(prices_np) > 1:
        log_returns[1:] = np.log(prices_np[1:] / prices_np[:-1])
    return log_returns

def convert_to_future_returns(input_prices: np.ndarray, output_prices: np.ndarray) -> np.ndarray:
    """Convert input/output prices to future log returns using NumPy."""
    input_prices_np = np.asarray(input_prices, dtype=np.float64)
    output_prices_np = np.asarray(output_prices, dtype=np.float64)

    returns = np.zeros(len(output_prices_np), dtype=np.float64)
    if len(output_prices_np) > 0:
        returns[0] = np.log(output_prices_np[0] / input_prices_np[-1])
        if len(output_prices_np) > 1:
            returns[1:] = np.log(output_prices_np[1:] / output_prices_np[:-1])
    return returns

def decode_log_returns(last_price: float, log_returns: np.ndarray) -> np.ndarray:
    """Convert log returns back to price predictions using NumPy for speed and robustness.
    Handles cases where last_price might be zero by ensuring non-negative predictions.
    """
    log_returns_np = np.asarray(log_returns, dtype=np.float64)
    multiplicative_factors = np.exp(np.cumsum(log_returns_np))
    prices = last_price * multiplicative_factors
    prices[prices < 0] = 0.0 
    return prices

class TemporalDecayAttention(nn.Module):
    """Attention with temporal decay weighting optimized for PyTorch 2.x."""
    def __init__(self, embed_dim: int, num_heads: int, decay_factor: float = 0.95):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

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
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        time_diffs = torch.abs(torch.arange(seq_len, device=DEVICE).unsqueeze(1) - torch.arange(seq_len, device=DEVICE).unsqueeze(0))
        decay_mask = self.decay_factor ** time_diffs
        decay_mask = decay_mask.unsqueeze(0).unsqueeze(0) 
        
        attn_scores = attn_scores * decay_mask
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)

class PEFTAdapter(nn.Module):
    """Parameter Efficient Fine-Tuning Adapter for nn.Linear layers."""
    def __init__(self, layer: nn.Module, rank: int = 8):
        super().__init__()
        if not isinstance(layer, nn.Linear):
            raise TypeError("PEFTAdapter currently only supports nn.Linear layers.")
        
        self.layer = layer
        self.rank = rank
        
        for param in self.layer.parameters():
            param.requires_grad = False
        
        in_features = layer.in_features
        out_features = layer.out_features
        
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) 
        
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features)) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.layer(x)
        lora_output = x @ self.lora_A @ self.lora_B
        return base_output + lora_output

class LibraModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, lstm_layers: int = 2, dropout: float = 0.2, attn_heads: int = 4):
        super().__init__()
        logging.info("🧠 Initializing Enhanced LibraModel...")
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.attn = TemporalDecayAttention(
            embed_dim=hidden_size,
            num_heads=attn_heads,
            decay_factor=0.95
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc_mean = nn.Linear(hidden_size, 5)
        self.fc_var = nn.Linear(hidden_size, 5)
        
        logging.info(f"✅ LSTM: input_size={input_size}, hidden_size={hidden_size}, layers={lstm_layers}")
        logging.info(f"✅ TemporalDecayAttention: embed_dim={hidden_size}, heads={attn_heads}")
        logging.info(f"✅ Dual-head output: mean + variance estimation")

    def enable_peft(self, rank: int = 8):
        """Enables PEFT adapters on the final linear layers (fc_mean and fc_var)."""
        if rank <= 0:
           logging.info("⚠️ PEFT rank is 0 or less. Skipping PEFT setup.")
           return self
        logging.info("🔧 Enabling PEFT adapters")
        self.fc_mean = PEFTAdapter(self.fc_mean, rank)
        self.fc_var = PEFTAdapter(self.fc_var, rank)
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        attn_out = self.attn(lstm_out)
        context = self.dropout(attn_out[:, -1, :])
        mean = self.fc_mean(context)
        log_var = self.fc_var(context)
        return mean, log_var

def upload_model_with_retry(max_attempts: int = 3) -> bool:
    """Upload model to Cloudinary with retry mechanism using Cloudinary SDK."""
    logging.info("📦 Zipping model for upload...")
    os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as zipf: 
        zipf.write(MODEL_PATH, arcname="model.pt")

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
                public_id="model.pt.zip",
                overwrite=True,
                use_filename=True,
                tags=["libra_model"]
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
    """Download model from Cloudinary using signed URL and extract it."""
    logging.info("📥 Downloading model ZIP from Cloudinary...")

    cloudinary.config(
        cloud_name=CLOUD_NAME,
        api_key=API_KEY,
        api_secret=API_SECRET,
        secure=True
    )

    try:
        url, _ = cloudinary.utils.cloudinary_url( 
            "model.pt.zip",
            resource_type='raw',
            type='upload',
            sign_url=True,
            expires_at=int(time.time()) + 600
        )

        response = requests.get(url, stream=True) 
        response.raise_for_status()

        os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
        with open(ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192): 
                f.write(chunk)
        logging.info(f"✅ ZIP saved to {ZIP_PATH}")

        os.makedirs("/tmp", exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall("/tmp")
        logging.info("✅ Model extracted to /tmp")
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Download request failed: {e}")
        raise
    except zipfile.BadZipFile:
        logging.error("❌ Downloaded file is not a valid zip file. It might be corrupted.")
        raise
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred during download or extraction: {e}")
        raise

def load_model() -> LibraModel:
    """Load model with retry mechanism and fallback to creating a new model if download/load fails."""
    if not os.path.exists(MODEL_PATH):
        attempts = 0
        while attempts < 3:
            try:
                download_model_from_cloudinary()
                logging.info("🥂 Resuming With Previously Trained Model")
                break
            except Exception as e:
                attempts += 1
                logging.warning(f"⚠️ Download failed (attempt {attempts}): {str(e)}. Retrying...")
                time.sleep(2 ** attempts)
        else:
            logging.error("❌ Download failed after multiple attempts. Creating a fresh model...")
            model = LibraModel().to(DEVICE)
            torch.save(model.state_dict(), MODEL_PATH)
            logging.info("😔 Saved New Model State Locally")
            if upload_model_with_retry():
                logging.info("🆕 Fresh model uploaded to Cloudinary")
            return model

    model = LibraModel().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        logging.info("✅ Model loaded successfully from local path.")
    except Exception as e:
        logging.error(f"❌ Failed to load model from {MODEL_PATH}: {e}. Initializing a fresh model instead.")
        model = LibraModel().to(DEVICE)
        torch.save(model.state_dict(), MODEL_PATH)
        if upload_model_with_retry():
            logging.info("🆕 Fresh model initialized and uploaded to Cloudinary due to load failure.")
    
    return model

@torch.no_grad()
def predict_ticks(model: LibraModel, ticks: list[float]) -> dict:
    """Predict next 5 ticks with confidence estimation, optimized for speed."""
    ticks_np = np.asarray(ticks, dtype=np.float64)
    returns = convert_to_log_returns(ticks_np)
    x = torch.tensor(returns, dtype=torch.float32).view(1, -1, 1).to(DEVICE) 
    
    mean_log_returns, log_var_log_returns = model(x) # Directly call model forward
    
    mean_log_returns_np = mean_log_returns.squeeze().cpu().numpy()
    log_var_log_returns_np = log_var_log_returns.squeeze().cpu().numpy()
    
    std_log_returns_np = np.exp(0.5 * log_var_log_returns_np)
    
    ci_low_log_returns_np = mean_log_returns_np - 1.96 * std_log_returns_np
    ci_high_log_returns_np = mean_log_returns_np + 1.96 * std_log_returns_np
    
    last_price = ticks_np[-1]
    predicted_prices = decode_log_returns(last_price, mean_log_returns_np)
    ci_low_prices = decode_log_returns(last_price, ci_low_log_returns_np)
    ci_high_prices = decode_log_returns(last_price, ci_high_log_returns_np)
    
    confidence_np = 1 / (1 + std_log_returns_np) 
    
    logging.info(f"📈 Predicted prices: {predicted_prices.tolist()}")
    logging.info(f"🛡️ Confidence: {confidence_np.tolist()}")
    
    return {
        "prices": predicted_prices.tolist(), 
        "confidence": confidence_np.tolist(),
        "ci_low": ci_low_prices.tolist(),
        "ci_high": ci_high_prices.tolist()
    }

def retrain_and_upload(model: LibraModel, x_data: list[list[float]], y_data: list[list[float]], epochs: int = 50, patience: int = 5, peft_rank: int = 8) -> LibraModel:
    """Enhanced training function with early stopping, metrics, gradient clipping, and sparse checkpointing."""
    logging.info("🔄 Converting data to log returns using optimized NumPy functions...")
    
    x_returns = [convert_to_log_returns(np.asarray(x_seq, dtype=np.float64)) for x_seq in x_data]
    y_returns = [convert_to_future_returns(np.asarray(x_data[i], dtype=np.float64), np.asarray(y_data[i], dtype=np.float64)) for i in range(len(x_data))]
    
    x_returns_np = np.stack(x_returns, axis=0)
    y_returns_np = np.stack(y_returns, axis=0)

    split_idx = int(0.8 * len(x_returns_np))
    x_train, x_val = x_returns_np[:split_idx], x_returns_np[split_idx:]
    y_train, y_val = y_returns_np[:split_idx], y_returns_np[split_idx:]
    
    x_train_t = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).unsqueeze(-1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(x_train_t, y_train_t)
    val_dataset = TensorDataset(x_val_t, y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)
    val_loader = DataLoader(val_dataset, batch_size=32, pin_memory=True, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)
    
    if peft_rank > 0:
        model = model.enable_peft(peft_rank)
    model.train().to(DEVICE)
    
    def loss_fn(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        var = torch.exp(log_var)
        return 0.5 * (log_var + ((target - mean) ** 2) / var).mean()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_direction = 0
        total_elements = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            
            mean, log_var = model(xb)
            
            loss = loss_fn(mean, log_var, yb)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            
            optimizer.step()
            
            train_loss += loss.item() * xb.size(0)
            
            pred_direction = torch.sign(mean)
            true_direction = torch.sign(yb)
            correct_direction += (pred_direction == true_direction).float().sum().item()
            total_elements += yb.numel()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total_elements = 0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                mean, log_var = model(xb)
                val_loss += loss_fn(mean, log_var, yb).item() * xb.size(0)
                
                pred_direction = torch.sign(mean)
                true_direction = torch.sign(yb)
                val_correct += (pred_direction == true_direction).float().sum().item()
                val_total_elements += yb.numel()
        
        train_loss /= len(train_loader.dataset)
        train_acc = correct_direction / total_elements if total_elements > 0 else 0.0
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total_elements if val_total_elements > 0 else 0.0
        
        scheduler.step(val_loss)
        
        logging.info(f"📚 Epoch {epoch+1}/{epochs}:")
        logging.info(f"  Train Loss: {train_loss:.6f} | Dir Acc: {train_acc:.4f}")
        logging.info(f"  Val Loss:   {val_loss:.6f} | Dir Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            if (epoch % 5 == 0) or (epoch == 0): 
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}_val_loss_{val_loss:.4f}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                logging.info(f"💾 Saved checkpoint: {checkpoint_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"⏳ No improvement for {epochs_no_improve}/{patience} epochs.")
            
            if epochs_no_improve >= patience:
                logging.info(f"🛑 Early stopping at epoch {epoch+1}")
                break
    
    final_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"🏆 Best model (or last trained) saved to: {final_model_path}")
    
    torch.save(model.state_dict(), MODEL_PATH)
    
    if upload_model_with_retry():
        logging.info("🚀 Retrained model uploaded to Cloudinary")
    else:
        logging.error("❌ Failed to upload retrained model")
    
    return model
