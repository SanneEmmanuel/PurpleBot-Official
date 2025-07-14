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
# Ensure to use torch.device for explicit device placement
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
logging.info(f"🔌 Using device: {DEVICE}")

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==========  Helper Functions ==========
def convert_to_log_returns(prices: np.ndarray) -> np.ndarray:
    """Convert price sequence to log returns with initial zero using NumPy for speed."""
    # Ensure prices are a NumPy array for efficient operations
    prices_np = np.asarray(prices, dtype=np.float64)
    # Using np.diff and np.log for vectorized computation
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
        # First return connects input to output
        returns[0] = np.log(output_prices_np[0] / input_prices_np[-1])
        # Subsequent returns between output prices
        if len(output_prices_np) > 1:
            returns[1:] = np.log(output_prices_np[1:] / output_prices_np[:-1])
    return returns

def decode_log_returns(last_price: float, log_returns: np.ndarray) -> np.ndarray:
    """Convert log returns back to price predictions using NumPy for speed and branchless ops."""
    log_returns_np = np.asarray(log_returns, dtype=np.float64)
    
    # Using np.cumsum and np.exp for vectorized, branchless computation
    # The initial price is last_price, and subsequent prices are derived by exp(log_return) * previous_price
    # np.exp(np.cumsum(log_returns_np)) gives the multiplicative factor relative to the first element's exp.
    # To correctly apply this, we consider the base price (last_price) and accumulate from there.
    
    # Calculate cumulative product of (1 + return) where returns are exp(log_returns) - 1
    # Or more directly, cumulative product of exp(log_returns)
    multiplicative_factors = np.exp(np.cumsum(log_returns_np))
    prices = last_price * multiplicative_factors
    
    return prices

# ========== 🧠 Enhanced Model Definition ==========
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
        
        # Using nn.Linear is efficient and standard
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project inputs and reshape for multi-head attention
        # Using .transpose() is efficient
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores. torch.matmul is highly optimized.
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Create temporal decay mask
        # Using broadcasting and .to(DEVICE) for efficiency
        time_diffs = torch.abs(torch.arange(seq_len, device=DEVICE).unsqueeze(1) - torch.arange(seq_len, device=DEVICE).unsqueeze(0))
        decay_mask = self.decay_factor ** time_diffs
        # Expand dimensions to match attention scores for broadcasting
        decay_mask = decay_mask.unsqueeze(0).unsqueeze(0) # Shape: [1, 1, seq_len, seq_len]
        
        # Apply decay mask to attention scores using element-wise multiplication
        # This is branchless and efficient.
        attn_scores = attn_scores * decay_mask
        
        # Apply softmax and get weighted values
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        
        # Combine heads and project back to original embedding dimension
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
        
        # Freeze original parameters
        for param in self.layer.parameters():
            param.requires_grad = False
        
        # Add low-rank adapters
        in_features = layer.in_features
        out_features = layer.out_features
        
        # Initialize lora_A with Kaiming uniform and lora_B with zeros, common practice
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) # Kaiming init
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features)) # Zero init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute base output first
        base_output = self.layer(x)
        
        # Compute LoRA output: x @ lora_A @ lora_B
        # Torch's matmul operator is optimized for this.
        lora_output = x @ self.lora_A @ self.lora_B
        
        # Add the LoRA output to the base output (branchless)
        return base_output + lora_output

class LibraModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, lstm_layers: int = 2, dropout: float = 0.2, attn_heads: int = 4):
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
        """Enables PEFT adapters on the final linear layers."""
        if rank <= 0:
           logging.info("⚠️ PEFT rank is 0 or less. Skipping PEFT setup.")
           return self
        logging.info("🔧 Enabling PEFT adapters")
        # Replace original layers with PEFT adapted layers
        self.fc_mean = PEFTAdapter(self.fc_mean, rank)
        self.fc_var = PEFTAdapter(self.fc_var, rank)
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Attention with temporal decay
        attn_out = self.attn(lstm_out)
        
        # Last timestep features (efficient slicing)
        context = self.dropout(attn_out[:, -1, :])
        
        # Dual outputs: mean and log variance
        mean = self.fc_mean(context)
        log_var = self.fc_var(context)
        
        # Return both predictions and uncertainty
        return mean, log_var

# ========== ⬇️⬆️ Upload/Download Helpers ==========
def upload_model_with_retry(max_attempts: int = 3) -> bool:
    """Upload model to Cloudinary with retry mechanism using Cloudinary SDK."""
    logging.info("📦 Zipping model for upload...")
    # Ensure the directory exists before attempting to write the zip file
    os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as zipf: # Added compression
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
                public_id="model.pt.zip",
                overwrite=True,
                use_filename=True, # Recommended to maintain original filename in Cloudinary
                # Add tags or folder for better organization if needed
                tags=["libra_model"]
            )
            logging.info(f"✅ Upload successful: {result.get('secure_url')}")
            return True
        except Exception as e:
            attempts += 1
            if attempts < max_attempts:
                wait_time = 2 ** attempts # Exponential backoff
                logging.warning(f"⚠️ Upload failed (attempt {attempts}): {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.error(f"❌ Upload failed after {max_attempts} attempts: {str(e)}")
                return False

def download_model_from_cloudinary():
    """Download model from Cloudinary using signed URL and extract it."""
    logging.info("📥 Downloading model ZIP from Cloudinary...")

    # Ensure Cloudinary is configured
    cloudinary.config(
        cloud_name=CLOUD_NAME,
        api_key=API_KEY,
        api_secret=API_SECRET,
        secure=True
    )

    try:
        # Generate signed URL valid for 10 minutes (600 seconds)
        url, _ = cloudinary.utils.cloudinary_url( # Ignoring options since we only need the URL
            "model.pt.zip",
            resource_type='raw',
            type='upload',
            sign_url=True,
            expires_at=int(time.time()) + 600
        )

        response = requests.get(url, stream=True) # Use stream for potentially large files
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)

        # Ensure the directory exists before writing the file
        os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
        with open(ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192): # Iterate in chunks
                f.write(chunk)
        logging.info(f"✅ ZIP saved to {ZIP_PATH}")

        # Extract contents to /tmp
        # Ensure the target directory for extraction exists
        os.makedirs("/tmp", exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall("/tmp")
        logging.info("✅ Model extracted to /tmp")
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Download request failed: {e}")
        raise
    except zipfile.BadZipFile:
        logging.error("❌ Downloaded file is not a valid zip file.")
        raise
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred during download or extraction: {e}")
        raise


# ========== 🚀 Load Model ==========
def load_model() -> LibraModel:
    """Load model with retry and fallback to new model."""
    # Try to download if local model doesn't exist
    if not os.path.exists(MODEL_PATH):
        attempts = 0
        while attempts < 3:
            try:
                download_model_from_cloudinary()
                logging.info("🥂 Resuming With Previously Trained Model")
                break # Exit loop if download successful
            except Exception as e:
                attempts += 1
                logging.warning(f"⚠️ Download failed (attempt {attempts}): {str(e)}. Retrying...")
                time.sleep(2 ** attempts) # Exponential backoff
        else: # This else block executes if the while loop completes without a 'break'
            logging.error("❌ Download failed 3 times. Creating fresh model...")
            model = LibraModel().to(DEVICE)
            # Save the new model state dictionary
            torch.save(model.state_dict(), MODEL_PATH)
            logging.info("😔 Saved New Model State Locally")
            if upload_model_with_retry():
                logging.info("🆕 Fresh model uploaded to Cloudinary")
            return model

    # If MODEL_PATH exists (either downloaded or previously existed), load it
    model = LibraModel().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval() # Set model to evaluation mode
        logging.info("✅ Model loaded successfully from local path.")
    except Exception as e:
        logging.error(f"❌ Failed to load model from {MODEL_PATH}: {e}. Initializing a fresh model.")
        model = LibraModel().to(DEVICE)
        # Optionally, save this new model state
        torch.save(model.state_dict(), MODEL_PATH)
        if upload_model_with_retry():
            logging.info("🆕 Fresh model initialized and uploaded to Cloudinary due to load failure.")
    
    return model

# ========== 🔮 Enhanced Predict Function ==========
@torch.no_grad() # Decorator to disable gradient calculations for inference
@torch.jit.script # Use torch.jit.script for JIT compilation for performance
def _predict_compiled(model: LibraModel, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """JIT compiled helper for prediction."""
    return model(x)

def predict_ticks(model: LibraModel, ticks: list[float]) -> dict:
    """Predict next 5 ticks with confidence estimation, optimized for speed."""
    # Ensure ticks are in a NumPy array for efficient processing
    ticks_np = np.asarray(ticks, dtype=np.float64)
    
    # Convert prices to log returns using optimized NumPy function
    returns = convert_to_log_returns(ticks_np)
    
    # Prepare tensor for model input
    # Ensure tensor is float32 as expected by the model
    x = torch.tensor(returns, dtype=torch.float32).view(1, -1, 1).to(DEVICE) # -1 for flexible sequence length
    
    # Use the JIT compiled prediction helper
    mean_log_returns, log_var_log_returns = _predict_compiled(model, x)
    
    # Convert to NumPy arrays immediately after computation on device to release GPU memory
    # Squeeze to remove batch dimension if it's 1
    mean_log_returns_np = mean_log_returns.squeeze().cpu().numpy()
    log_var_log_returns_np = log_var_log_returns.squeeze().cpu().numpy()
    
    # Calculate standard deviation from log variance using np.exp and element-wise ops (branchless)
    std_log_returns_np = np.exp(0.5 * log_var_log_returns_np)
    
    # Calculate confidence intervals (95% CI) for log returns (branchless)
    # 1.96 is for 95% confidence interval for a normal distribution
    ci_low_log_returns_np = mean_log_returns_np - 1.96 * std_log_returns_np
    ci_high_log_returns_np = mean_log_returns_np + 1.96 * std_log_returns_np
    
    # Decode log returns to price predictions using optimized NumPy function
    last_price = ticks_np[-1]
    predicted_prices = decode_log_returns(last_price, mean_log_returns_np)
    ci_low_prices = decode_log_returns(last_price, ci_low_log_returns_np)
    ci_high_prices = decode_log_returns(last_price, ci_high_log_returns_np)
    
    # Calculate confidence scores (1 - coefficient of variation)
    # Avoid division by zero if std is extremely small, add a small epsilon if needed
    confidence_np = 1 / (1 + std_log_returns_np) # This is a direct, branchless calculation.
    
    logging.info(f"📈 Predicted prices: {predicted_prices.tolist()}")
    logging.info(f"🛡️ Confidence: {confidence_np.tolist()}")
    
    return {
        "prices": predicted_prices.tolist(), # Convert back to list for JSON compatibility
        "confidence": confidence_np.tolist(),
        "ci_low": ci_low_prices.tolist(),
        "ci_high": ci_high_prices.tolist()
    }

# ========== 🔁 Enhanced Training Function ==========
def retrain_and_upload(model: LibraModel, x_data: list[list[float]], y_data: list[list[float]], epochs: int = 50, patience: int = 5, peft_rank: int = 8) -> LibraModel:
    """Enhanced training with early stopping, metrics, and sparse checkpointing."""
    # Prepare data
    logging.info("🔄 Converting data to log returns using optimized NumPy functions...")
    
    x_returns = [convert_to_log_returns(np.asarray(x_seq, dtype=np.float64)) for x_seq in x_data]
    y_returns = [convert_to_future_returns(np.asarray(x_data[i], dtype=np.float64), np.asarray(y_data[i], dtype=np.float64)) for i in range(len(x_data))]
    
    # Convert lists of NumPy arrays to single NumPy arrays for splitting
    x_returns_np = np.stack(x_returns, axis=0)
    y_returns_np = np.stack(y_returns, axis=0)

    # Split into train and validation (80/20) - using NumPy for efficient slicing
    split_idx = int(0.8 * len(x_returns_np))
    x_train, x_val = x_returns_np[:split_idx], x_returns_np[split_idx:]
    y_train, y_val = y_returns_np[:split_idx], y_returns_np[split_idx:]
    
    # Convert to tensors
    # Ensure correct dimensions for LSTM: (batch, sequence_length, input_size)
    x_train_t = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).unsqueeze(-1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    
    # Create datasets
    train_dataset = TensorDataset(x_train_t, y_train_t)
    val_dataset = TensorDataset(x_val_t, y_val_t)
    
    # Create dataloaders
    # Pin memory for faster data transfer to GPU
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)
    val_loader = DataLoader(val_dataset, batch_size=32, pin_memory=True, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)
    
    # Enable PEFT
    if peft_rank > 0:
        model = model.enable_peft(peft_rank)
    model.train().to(DEVICE) # Ensure model is in training mode and on the correct device
    
    # Loss function (Gaussian NLL) - as per original, it's efficient
    def loss_fn(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        var = torch.exp(log_var)
        # Using torch.mean for scalar loss
        return 0.5 * (log_var + ((target - mean) ** 2) / var).mean()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Consider more advanced schedulers if needed, but ReduceLROnPlateau is good.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    
    # Training variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train() # Set to training mode
        train_loss = 0.0
        correct_direction = 0
        total_elements = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad() # Zero gradients before backward pass
            
            mean, log_var = model(xb)
            
            # Calculate loss and backpropagate
            loss = loss_fn(mean, log_var, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()
            
            # Metrics
            train_loss += loss.item() * xb.size(0) # Accumulate batch loss
            
            # Directional accuracy (branchless using torch.sign and equality)
            # torch.sign returns -1 for negative, 0 for zero, 1 for positive
            pred_direction = torch.sign(mean)
            true_direction = torch.sign(yb)
            # Use .float() for boolean to float conversion for summation
            correct_direction += (pred_direction == true_direction).float().sum().item()
            total_elements += yb.numel() # Total number of elements in the target batch

        # Validation phase
        model.eval() # Set to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total_elements = 0
        
        with torch.no_grad(): # Disable gradient calculations
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                mean, log_var = model(xb)
                val_loss += loss_fn(mean, log_var, yb).item() * xb.size(0)
                
                # Directional accuracy
                pred_direction = torch.sign(mean)
                true_direction = torch.sign(yb)
                val_correct += (pred_direction == true_direction).float().sum().item()
                val_total_elements += yb.numel()
        
        # Calculate epoch metrics
        train_loss /= len(train_loader.dataset)
        train_acc = correct_direction / total_elements if total_elements > 0 else 0.0
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total_elements if val_total_elements > 0 else 0.0
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        logging.info(f"📚 Epoch {epoch+1}/{epochs}:")
        logging.info(f"  Train Loss: {train_loss:.6f} | Dir Acc: {train_acc:.4f}")
        logging.info(f"  Val Loss:   {val_loss:.6f} | Dir Acc: {val_acc:.4f}")
        
        # Check for improvement and save sparse checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # Save sparse checkpoint (every 5 improvements or the very first improvement)
            if (epoch % 5 == 0) or (epoch == 0): # Save on first improvement as well
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}_val_loss_{val_loss:.4f}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                logging.info(f"💾 Saved checkpoint: {checkpoint_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"⏳ No improvement for {epochs_no_improve}/{patience} epochs.")
            
            # Early stopping check
            if epochs_no_improve >= patience:
                logging.info(f"🛑 Early stopping at epoch {epoch+1}")
                break
    
    # Save the final best model (or the model at early stopping)
    final_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"🏆 Best model (or last trained) saved to: {final_model_path}")
    
    # Copy to main model path for deployment/loading
    torch.save(model.state_dict(), MODEL_PATH)
    
    # Upload to cloud
    if upload_model_with_retry():
        logging.info("🚀 Retrained model uploaded to Cloudinary")
    else:
        logging.error("❌ Failed to upload retrained model")
    
    return model
