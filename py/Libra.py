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
        # Calculate log returns for subsequent prices relative to previous ones
        log_returns[1:] = np.log(prices_np[1:] / prices_np[:-1])
    return log_returns

def convert_to_future_returns(input_prices: np.ndarray, output_prices: np.ndarray) -> np.ndarray:
    """Convert input/output prices to future log returns using NumPy."""
    input_prices_np = np.asarray(input_prices, dtype=np.float64)
    output_prices_np = np.asarray(output_prices, dtype=np.float64)

    returns = np.zeros(len(output_prices_np), dtype=np.float64)
    if len(output_prices_np) > 0:
        # First return connects the last input price to the first output price
        returns[0] = np.log(output_prices_np[0] / input_prices_np[-1])
        # Subsequent returns are between consecutive output prices
        if len(output_prices_np) > 1:
            returns[1:] = np.log(output_prices_np[1:] / output_prices_np[:-1])
    return returns

def decode_log_returns(last_price: float, log_returns: np.ndarray) -> np.ndarray:
    """Convert log returns back to price predictions using NumPy for speed and robustness.
    Handles cases where last_price might be zero by ensuring non-negative predictions.
    """
    log_returns_np = np.asarray(log_returns, dtype=np.float64)
    
    # Calculate cumulative product of exp(log_returns)
    # This effectively computes (exp(r1), exp(r1+r2), exp(r1+r2+r3), ...)
    multiplicative_factors = np.exp(np.cumsum(log_returns_np))
    
    # Apply to last_price to get the predicted prices
    prices = last_price * multiplicative_factors
    
    # Ensure prices are non-negative. This is a safeguard against
    # potential floating-point inaccuracies or model predictions that
    # might theoretically drop below zero in a financial context.
    prices[prices < 0] = 0.0 
    
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
        
        # Linear projections for Query, Key, Value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project inputs and reshape for multi-head attention
        # Transpose for (batch_size, num_heads, seq_len, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores: (Q @ K^T) / sqrt(head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Create temporal decay mask
        # Generates a matrix of absolute time differences: |i - j|
        time_diffs = torch.abs(torch.arange(seq_len, device=DEVICE).unsqueeze(1) - torch.arange(seq_len, device=DEVICE).unsqueeze(0))
        # Compute decay factors: decay_factor ^ time_diffs
        decay_mask = self.decay_factor ** time_diffs
        # Expand dimensions to match attention scores for broadcasting: [1, 1, seq_len, seq_len]
        decay_mask = decay_mask.unsqueeze(0).unsqueeze(0) 
        
        # Apply decay mask to attention scores
        attn_scores = attn_scores * decay_mask
        
        # Apply softmax to get attention probabilities and then multiply with Value
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        
        # Combine heads and project back to original embedding dimension
        # Reshape from (batch_size, num_heads, seq_len, head_dim) to (batch_size, seq_len, embed_dim)
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
        
        # Freeze original parameters of the linear layer
        for param in self.layer.parameters():
            param.requires_grad = False
        
        # Add low-rank adapters (LoRA)
        in_features = layer.in_features
        out_features = layer.out_features
        
        # Initialize lora_A with Kaiming uniform (suitable for ReLU-like activations)
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) # 'a' for leaky_relu, but common for ReLU too
        
        # Initialize lora_B with zeros, so initial LoRA output is zero
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features)) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute base output from the original frozen layer
        base_output = self.layer(x)
        
        # Compute LoRA output: (input @ lora_A) @ lora_B
        lora_output = x @ self.lora_A @ self.lora_B
        
        # Add the LoRA output to the base output
        return base_output + lora_output

class LibraModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, lstm_layers: int = 2, dropout: float = 0.2, attn_heads: int = 4):
        super().__init__()
        logging.info("🧠 Initializing Enhanced LibraModel...")
        
        # LSTM Encoder: Processes sequential input data
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=True # Input and output tensors are provided as (batch, seq, feature)
        )
        
        # Temporal Decay Attention: Applies attention with a decay based on time difference
        self.attn = TemporalDecayAttention(
            embed_dim=hidden_size, # LSTM hidden size is the embedding dimension for attention
            num_heads=attn_heads,
            decay_factor=0.95
        )
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Main prediction layer for the mean of the predicted log returns
        self.fc_mean = nn.Linear(hidden_size, 5) # Predicts 5 future log returns
        
        # Layer for estimating the log variance (uncertainty) of predictions
        self.fc_var = nn.Linear(hidden_size, 5) # Predicts 5 log variances
        
        logging.info(f"✅ LSTM: input_size={input_size}, hidden_size={hidden_size}, layers={lstm_layers}")
        logging.info(f"✅ TemporalDecayAttention: embed_dim={hidden_size}, heads={attn_heads}")
        logging.info(f"✅ Dual-head output: mean + variance estimation")

    def enable_peft(self, rank: int = 8):
        """Enables PEFT adapters on the final linear layers (fc_mean and fc_var)."""
        if rank <= 0:
           logging.info("⚠️ PEFT rank is 0 or less. Skipping PEFT setup.")
           return self
        logging.info("🔧 Enabling PEFT adapters")
        # Replace original linear layers with PEFT adapted versions
        self.fc_mean = PEFTAdapter(self.fc_mean, rank)
        self.fc_var = PEFTAdapter(self.fc_var, rank)
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM processing: lstm_out shape (batch_size, sequence_length, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # Attention with temporal decay
        attn_out = self.attn(lstm_out)
        
        # Extract features from the last timestep of the attention output
        # context shape: (batch_size, hidden_size)
        context = self.dropout(attn_out[:, -1, :])
        
        # Dual outputs: mean and log variance for the 5 future predictions
        mean = self.fc_mean(context)
        log_var = self.fc_var(context)
        
        # Return both predictions (mean) and their associated uncertainty (log_var)
        return mean, log_var

# ========== ⬇️⬆️ Upload/Download Helpers ==========
def upload_model_with_retry(max_attempts: int = 3) -> bool:
    """Upload model to Cloudinary with retry mechanism using Cloudinary SDK."""
    logging.info("📦 Zipping model for upload...")
    # Ensure the directory for the ZIP file exists
    os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
    # Create a zip file with DEFLATED compression for smaller size
    with zipfile.ZipFile(ZIP_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as zipf: 
        zipf.write(MODEL_PATH, arcname="model.pt") # Add model.pt to the zip archive

    # Configure Cloudinary using credentials from environment variables
    cloudinary.config(
        cloud_name=CLOUD_NAME,
        api_key=API_KEY,
        api_secret=API_SECRET,
        secure=True
    )

    attempts = 0
    while attempts < max_attempts:
        try:
            # Upload the zip file to Cloudinary
            result = cloudinary.uploader.upload(
                ZIP_PATH,
                resource_type='raw', # Specify raw resource type for non-image files
                public_id="model.pt.zip", # Public ID for easy access
                overwrite=True, # Overwrite if a file with the same public_id exists
                use_filename=True, # Use the original filename in Cloudinary
                tags=["libra_model"] # Add tags for better organization and searchability
            )
            logging.info(f"✅ Upload successful: {result.get('secure_url')}")
            return True # Return True on successful upload
        except Exception as e:
            attempts += 1
            if attempts < max_attempts:
                wait_time = 2 ** attempts # Exponential backoff for retries
                logging.warning(f"⚠️ Upload failed (attempt {attempts}): {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.error(f"❌ Upload failed after {max_attempts} attempts: {str(e)}")
                return False # Return False if all attempts fail


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
        # Generate a signed URL for secure download, valid for 10 minutes (600 seconds)
        url, _ = cloudinary.utils.cloudinary_url( 
            "model.pt.zip",
            resource_type='raw',
            type='upload',
            sign_url=True,
            expires_at=int(time.time()) + 600
        )

        # Make a GET request to download the file, streaming for efficiency with large files
        response = requests.get(url, stream=True) 
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        # Ensure the directory for the ZIP file exists before writing
        os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
        with open(ZIP_PATH, "wb") as f:
            # Write content in chunks to handle potentially large files efficiently
            for chunk in response.iter_content(chunk_size=8192): 
                f.write(chunk)
        logging.info(f"✅ ZIP saved to {ZIP_PATH}")

        # Ensure the target directory for extraction exists (e.g., /tmp)
        os.makedirs("/tmp", exist_ok=True)
        # Extract the contents of the zip file to the /tmp directory
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall("/tmp")
        logging.info("✅ Model extracted to /tmp")
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Download request failed: {e}")
        raise # Re-raise the exception for upstream handling
    except zipfile.BadZipFile:
        logging.error("❌ Downloaded file is not a valid zip file. It might be corrupted.")
        raise
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred during download or extraction: {e}")
        raise


# ========== 🚀 Load Model ==========
def load_model() -> LibraModel:
    """Load model with retry mechanism and fallback to creating a new model if download/load fails."""
    # Check if the model already exists locally
    if not os.path.exists(MODEL_PATH):
        attempts = 0
        while attempts < 3: # Try downloading up to 3 times
            try:
                download_model_from_cloudinary()
                logging.info("🥂 Resuming With Previously Trained Model")
                break # Exit loop if download is successful
            except Exception as e:
                attempts += 1
                logging.warning(f"⚠️ Download failed (attempt {attempts}): {str(e)}. Retrying...")
                time.sleep(2 ** attempts) # Exponential backoff between retries
        else: # This 'else' block executes if the 'while' loop completes without a 'break'
            logging.error("❌ Download failed after multiple attempts. Creating a fresh model...")
            model = LibraModel().to(DEVICE) # Initialize a new model
            torch.save(model.state_dict(), MODEL_PATH) # Save its initial state locally
            logging.info("😔 Saved New Model State Locally")
            if upload_model_with_retry(): # Attempt to upload the new model to Cloudinary
                logging.info("🆕 Fresh model uploaded to Cloudinary")
            return model # Return the newly created model

    # If MODEL_PATH exists (either downloaded or was already present), attempt to load it
    model = LibraModel().to(DEVICE)
    try:
        # Load the model's state dictionary, mapping to the correct device
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval() # Set the model to evaluation mode (disables dropout, batchnorm updates)
        logging.info("✅ Model loaded successfully from local path.")
    except Exception as e:
        logging.error(f"❌ Failed to load model from {MODEL_PATH}: {e}. Initializing a fresh model instead.")
        model = LibraModel().to(DEVICE) # Initialize a new model if loading fails
        torch.save(model.state_dict(), MODEL_PATH) # Save its initial state locally
        if upload_model_with_retry(): # Attempt to upload the new model
            logging.info("🆕 Fresh model initialized and uploaded to Cloudinary due to load failure.")
    
    return model

# ========== 🔮 Enhanced Predict Function ==========
@torch.no_grad() # Decorator to disable gradient calculations for inference, saving memory and speeding up.
@torch.jit.script # Decorator to compile this function into TorchScript for optimized execution.
def _predict_compiled(model: LibraModel, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """JIT compiled helper for model prediction. This function is optimized by TorchScript."""
    return model(x)

def predict_ticks(model: LibraModel, ticks: list[float]) -> dict:
    """Predict next 5 ticks with confidence estimation, optimized for speed."""
    # Convert input list of ticks to a NumPy array for efficient numerical operations
    ticks_np = np.asarray(ticks, dtype=np.float64)
    
    # Convert prices to log returns using the optimized NumPy helper function
    returns = convert_to_log_returns(ticks_np)
    
    # Prepare the input tensor for the model.
    # .view(1, -1, 1) reshapes to (batch_size=1, sequence_length, input_feature_size=1).
    # .to(DEVICE) moves the tensor to the appropriate computing device (CPU/GPU).
    x = torch.tensor(returns, dtype=torch.float32).view(1, -1, 1).to(DEVICE) 
    
    # Use the JIT compiled prediction helper function to get mean and log variance
    mean_log_returns, log_var_log_returns = _predict_compiled(model, x)
    
    # Move results back to CPU and convert to NumPy arrays to release GPU memory and for further NumPy ops.
    # .squeeze() removes any dimensions of size 1 (e.g., the batch dimension if it was 1).
    mean_log_returns_np = mean_log_returns.squeeze().cpu().numpy()
    log_var_log_returns_np = log_var_log_returns.squeeze().cpu().numpy()
    
    # Calculate standard deviation from log variance. std = exp(0.5 * log_var)
    std_log_returns_np = np.exp(0.5 * log_var_log_returns_np)
    
    # Calculate 95% Confidence Intervals (CI) for log returns. CI = mean +/- 1.96 * std
    ci_low_log_returns_np = mean_log_returns_np - 1.96 * std_log_returns_np
    ci_high_log_returns_np = mean_log_returns_np + 1.96 * std_log_returns_np
    
    # Decode log returns back to actual price predictions using the optimized NumPy helper function
    last_price = ticks_np[-1] # The last known price is the base for future predictions
    predicted_prices = decode_log_returns(last_price, mean_log_returns_np)
    ci_low_prices = decode_log_returns(last_price, ci_low_log_returns_np)
    ci_high_prices = decode_log_returns(last_price, ci_high_log_returns_np)
    
    # Calculate confidence scores (e.g., 1 / (1 + coefficient of variation or standard deviation)
    # A higher std_log_returns_np implies lower confidence.
    confidence_np = 1 / (1 + std_log_returns_np) 
    
    logging.info(f"📈 Predicted prices: {predicted_prices.tolist()}")
    logging.info(f"🛡️ Confidence: {confidence_np.tolist()}")
    
    # Return results as a dictionary, converting NumPy arrays back to Python lists for broader compatibility (e.g., JSON serialization)
    return {
        "prices": predicted_prices.tolist(), 
        "confidence": confidence_np.tolist(),
        "ci_low": ci_low_prices.tolist(),
        "ci_high": ci_high_prices.tolist()
    }

# ========== 🔁 Enhanced Training Function ==========
def retrain_and_upload(model: LibraModel, x_data: list[list[float]], y_data: list[list[float]], epochs: int = 50, patience: int = 5, peft_rank: int = 8) -> LibraModel:
    """Enhanced training function with early stopping, metrics, gradient clipping, and sparse checkpointing."""
    logging.info("🔄 Converting data to log returns using optimized NumPy functions...")
    
    # Convert input price data to log returns using list comprehensions and NumPy
    x_returns = [convert_to_log_returns(np.asarray(x_seq, dtype=np.float64)) for x_seq in x_data]
    y_returns = [convert_to_future_returns(np.asarray(x_data[i], dtype=np.float64), np.asarray(y_data[i], dtype=np.float64)) for i in range(len(x_data))]
    
    # Stack the list of NumPy arrays into single NumPy arrays for efficient splitting and tensor conversion
    x_returns_np = np.stack(x_returns, axis=0)
    y_returns_np = np.stack(y_returns, axis=0)

    # Split data into training and validation sets (80/20 split)
    split_idx = int(0.8 * len(x_returns_np))
    x_train, x_val = x_returns_np[:split_idx], x_returns_np[split_idx:]
    y_train, y_val = y_returns_np[:split_idx], y_returns_np[split_idx:]
    
    # Convert NumPy arrays to PyTorch tensors.
    # .unsqueeze(-1) adds a feature dimension of size 1, required by LSTM for single-feature input.
    x_train_t = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).unsqueeze(-1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    
    # Create PyTorch TensorDatasets from the tensors
    train_dataset = TensorDataset(x_train_t, y_train_t)
    val_dataset = TensorDataset(x_val_t, y_val_t)
    
    # Create DataLoaders for batching and shuffling data.
    # pin_memory=True: Speeds up data transfer to GPU.
    # num_workers: Enables parallel data loading (set to half CPU cores for balance).
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)
    val_loader = DataLoader(val_dataset, batch_size=32, pin_memory=True, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)
    
    # Enable PEFT (Parameter Efficient Fine-Tuning) if rank is greater than 0
    if peft_rank > 0:
        model = model.enable_peft(peft_rank)
    model.train().to(DEVICE) # Set model to training mode and move to the specified device
    
    # Define the loss function (Gaussian Negative Log Likelihood)
    def loss_fn(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        var = torch.exp(log_var) # Calculate variance from log variance
        # Computes 0.5 * (log(var) + ((target - mean)^2 / var)) and takes the mean over all elements
        return 0.5 * (log_var + ((target - mean) ** 2) / var).mean()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # Adam optimizer
    # Learning rate scheduler: Reduces LR when validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    
    # Variables for early stopping and tracking best model
    best_val_loss = float('inf') # Initialize with a very high value
    epochs_no_improve = 0 # Counter for epochs without validation loss improvement
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train() # Set model to training mode
        train_loss = 0.0
        correct_direction = 0
        total_elements = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE) # Move batch to device
            optimizer.zero_grad() # Clear gradients from previous step
            
            mean, log_var = model(xb) # Forward pass
            
            # Calculate loss
            loss = loss_fn(mean, log_var, yb)
            loss.backward() # Backpropagation: computes gradients
            
            # Gradient clipping: clips gradients by their norm to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            
            optimizer.step() # Update model parameters
            
            # Metrics calculation for training phase
            train_loss += loss.item() * xb.size(0) # Accumulate batch loss, scaled by batch size
            
            # Directional accuracy: checks if predicted direction matches true direction
            pred_direction = torch.sign(mean) # -1, 0, or 1
            true_direction = torch.sign(yb)
            # Sum correctly predicted directions. .float() converts boolean (True/False) to (1.0/0.0)
            correct_direction += (pred_direction == true_direction).float().sum().item()
            total_elements += yb.numel() # Total number of elements in the target tensor for accuracy calculation

        # Validation phase
        model.eval() # Set model to evaluation mode (disables dropout, etc.)
        val_loss = 0.0
        val_correct = 0
        val_total_elements = 0
        
        with torch.no_grad(): # Disable gradient calculations for validation
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                mean, log_var = model(xb)
                val_loss += loss_fn(mean, log_var, yb).item() * xb.size(0)
                
                # Directional accuracy for validation
                pred_direction = torch.sign(mean)
                true_direction = torch.sign(yb)
                val_correct += (pred_direction == true_direction).float().sum().item()
                val_total_elements += yb.numel()
        
        # Calculate epoch-level average metrics
        train_loss /= len(train_loader.dataset)
        train_acc = correct_direction / total_elements if total_elements > 0 else 0.0
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total_elements if val_total_elements > 0 else 0.0
        
        # Update learning rate scheduler based on validation loss
        scheduler.step(val_loss)
        
        # Log epoch metrics
        logging.info(f"📚 Epoch {epoch+1}/{epochs}:")
        logging.info(f"  Train Loss: {train_loss:.6f} | Dir Acc: {train_acc:.4f}")
        logging.info(f"  Val Loss:   {val_loss:.6f} | Dir Acc: {val_acc:.4f}")
        
        # Early stopping logic and sparse checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0 # Reset counter if improvement seen
            
            # Save checkpoint: every 5 improvements or on the very first improvement (epoch 0)
            if (epoch % 5 == 0) or (epoch == 0): 
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}_val_loss_{val_loss:.4f}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                logging.info(f"💾 Saved checkpoint: {checkpoint_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"⏳ No improvement for {epochs_no_improve}/{patience} epochs.")
            
            # Check for early stopping condition
            if epochs_no_improve >= patience:
                logging.info(f"🛑 Early stopping at epoch {epoch+1}")
                break # Exit training loop
    
    # Save the final best model (or the model at the point of early stopping)
    final_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"🏆 Best model (or last trained) saved to: {final_model_path}")
    
    # Copy the best model to the main model path for deployment/loading
    torch.save(model.state_dict(), MODEL_PATH)
    
    # Attempt to upload the retrained model to Cloudinary
    if upload_model_with_retry():
        logging.info("🚀 Retrained model uploaded to Cloudinary")
    else:
        logging.error("❌ Failed to upload retrained model")
    
    return model
