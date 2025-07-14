# Libra Pro_V56 (Classification, CrossEntropy, Optimized for Patterns & Speed)
import os, requests, zipfile, torch, logging, math, time, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO
from torch.utils.data import DataLoader, TensorDataset
import cloudinary, cloudinary.uploader, cloudinary.api

CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "dj4bwntzb")
API_KEY = os.getenv("CLOUDINARY_API_KEY", "354656419316393")
API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "M-Trl9ltKDHyo1dIP2AaLOG-WPM")
MODEL_URL = f"https://res.cloudinary.com/{CLOUD_NAME}/raw/upload/v1/libra_v56.pt.zip"
MODEL_PATH = "/tmp/libra_v56.pt"
ZIP_PATH = "/tmp/libra_v56.pt.zip"
CHECKPOINT_DIR = "/tmp/checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logging.info(f"🔌 Using device: {DEVICE}")

TICK_CLASSES = 9        # classes from -4 to +4
TICK_RANGE = np.arange(-4, 5)  # [-4, -3, ..., 4]

class MultiDecayTemporalAttention(nn.Module):
    """
    Multi-decay temporal attention with trainable decay factors.
    Each head can have its own decay parameter, which can be learned.
    Supports multiple decay mechanisms (exponential, linear, etc. if desired).
    """
    def __init__(self, embed_dim: int, num_heads: int, num_decays: int = 2, trainable_decay: bool = True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_decays = num_decays

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Multi decay: for each head, multiple decay factors (e.g. exponential, linear)
        # Exponential decay: decay_param in (0, 1), initialized close to 1 for long memory
        if trainable_decay:
            self.decay_params = nn.Parameter(torch.full((num_heads, num_decays), 0.95))
        else:
            self.register_buffer("decay_params", torch.full((num_heads, num_decays), 0.95))

        # Mix weights: how much each decay is weighted per head (softmax normalized)
        self.mix_weights = nn.Parameter(torch.zeros(num_heads, num_decays))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Compute all decay masks: shape (num_heads, num_decays, seq_len, seq_len)
        time_diffs = torch.abs(torch.arange(seq_len, device=x.device).unsqueeze(1) - torch.arange(seq_len, device=x.device).unsqueeze(0))  # (seq_len, seq_len)
        # (num_heads, num_decays, seq_len, seq_len)
        decay_masks = []
        for i in range(self.num_decays):
            # Clamp decay param for numerical stability
            decay_param_i = torch.clamp(self.decay_params[:, i], 0.01, 0.9999)  # (num_heads,)
            # decay_param_i: (num_heads, 1, 1) so broadcast over seq_len
            exp_decay = decay_param_i[:, None, None] ** time_diffs[None, :, :]
            decay_masks.append(exp_decay)
            # Other decay types (e.g., linear) can be added here if desired

        decay_masks = torch.stack(decay_masks, dim=1)  # (num_heads, num_decays, seq_len, seq_len)
        # Normalize mix weights over decays per head (softmax)
        mix_softmax = F.softmax(self.mix_weights, dim=-1)  # (num_heads, num_decays)
        # Weighted sum over decays
        decay_mask = (decay_masks * mix_softmax[:, :, None, None]).sum(dim=1)  # (num_heads, seq_len, seq_len)
        decay_mask = decay_mask.unsqueeze(0)  # (1, num_heads, seq_len, seq_len) for broadcasting over batch

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
    def __init__(self, input_size: int = 1, hidden_size: int = 64, lstm_layers: int = 2, dropout: float = 0.2, attn_heads: int = 4, num_attn_decays: int = 2):
        super().__init__()
        logging.info("🧠 Initializing LibraModel V56 (Classification/Pattern, MultiDecay Temporal Attention)...")

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=True
        )

        self.attn = MultiDecayTemporalAttention(
            embed_dim=hidden_size,
            num_heads=attn_heads,
            num_decays=num_attn_decays,
            trainable_decay=True
        )

        self.dropout = nn.Dropout(dropout)
        # Single output head: predicts logits for 5 steps, each with 9 classes
        self.fc_out = nn.Linear(hidden_size, 5 * TICK_CLASSES)

        logging.info(f"✅ Output: (batch, 5, 9) tick class logits (multi-decay, trainable)")

    def enable_peft(self, rank: int = 8):
        """Enables PEFT adapters on the final linear layer (fc_out)."""
        if rank <= 0:
           logging.info("⚠️ PEFT rank is 0 or less. Skipping PEFT setup.")
           return self
        logging.info("🔧 Enabling PEFT adapter on fc_out")
        self.fc_out = PEFTAdapter(self.fc_out, rank)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_out = self.attn(lstm_out)
        context = self.dropout(attn_out[:, -1, :])
        logits = self.fc_out(context)
        logits = logits.view(-1, 5, TICK_CLASSES)  # (batch, 5, 9)
        return logits

# ... rest of your code unchanged ...
# (upload_model_with_retry, download_model_from_cloudinary, load_model, predict_ticks, retrain_and_upload)
def upload_model_with_retry(max_attempts: int = 3) -> bool:
    """Upload model to Cloudinary with retry mechanism using Cloudinary SDK."""
    logging.info("📦 Zipping model for upload...")
    os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as zipf: 
        zipf.write(MODEL_PATH, arcname="libra_v56.pt")

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
                public_id="libra_v56.pt.zip",
                overwrite=True,
                use_filename=True,
                tags=["libra_model_v56"]
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
            "libra_v56.pt.zip",
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
def predict_ticks(model: LibraModel, ticks: list[float], tick_size: float = 1.0) -> dict:
    """
    Predict next 5 ticks as class deltas (-4..+4), convert to absolute prices, and provide softmax confidence.
    Args:
        - model: LibraModel
        - ticks: list of recent prices
        - tick_size: size of one tick (price unit)
    Returns:
        - dict with 'prices', 'confidence', 'class_probs', 'deltas', 'class_indices'
    """
    ticks_np = np.asarray(ticks, dtype=np.float64)
    x = torch.tensor(ticks_np, dtype=torch.float32).view(1, -1, 1).to(DEVICE)
    
    logits = model(x)  # (1, 5, 9)
    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]  # (5, 9)
    class_indices = np.argmax(probs, axis=-1)           # (5,)
    class_probs = probs[np.arange(5), class_indices]     # (5,)
    deltas = TICK_RANGE[class_indices]                   # (5,)
    next_prices = []
    last = ticks_np[-1]
    for d in deltas:
        last = last + d * tick_size
        next_prices.append(last)
    
    logging.info(f"📈 Predicted tick deltas: {deltas.tolist()}")
    logging.info(f"💡 Predicted prices: {next_prices}")
    logging.info(f"🛡️ Confidence: {class_probs.tolist()}")

    return {
        "prices": next_prices, 
        "confidence": class_probs.tolist(),
        "class_probs": probs.tolist(),
        "deltas": deltas.tolist(),
        "class_indices": class_indices.tolist()
    }

def retrain_and_upload(model: LibraModel, x_data: list[list[float]], y_data: list[list[float]], epochs: int = 50, patience: int = 5, peft_rank: int = 8, tick_size: float = 1.0) -> LibraModel:
    """
    Retrain LibraModelV56 for tick class prediction, CrossEntropyLoss, early stopping, metrics, checkpointing.
    Args:
        - x_data: list of price histories (list of floats)
        - y_data: list of ground truth next prices (list of 5 floats per sample)
        - tick_size: size of one tick (price unit)
    """
    logging.info("🔄 Preparing data for tick class classification...")

    # Convert y_data (list of next 5 prices) into class indices (-4..4 mapped to 0..8)
    x_np = np.array(x_data, dtype=np.float32)
    y_np = np.array(y_data, dtype=np.float32)
    y_classes = []
    for i in range(len(y_np)):
        last = x_np[i, -1]
        # Compute delta for each future tick, then map to class index (0..8)
        deltas = np.round((y_np[i] - last) / tick_size).astype(int)
        deltas = np.clip(deltas, -4, 4)
        class_indices = deltas + 4
        y_classes.append(class_indices)
    y_classes_np = np.stack(y_classes, axis=0)  # shape (batch, 5)

    split_idx = int(0.8 * len(x_np))
    x_train, x_val = x_np[:split_idx], x_np[split_idx:]
    y_train, y_val = y_classes_np[:split_idx], y_classes_np[split_idx:]

    x_train_t = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).unsqueeze(-1)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(x_train_t, y_train_t)
    val_dataset = TensorDataset(x_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)
    val_loader = DataLoader(val_dataset, batch_size=32, pin_memory=True, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)

    if peft_rank > 0:
        model = model.enable_peft(peft_rank)
    model.train().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)  # (batch, 5, 9)

            loss = sum(criterion(logits[:, step, :], yb[:, step]) for step in range(5)) / 5.0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

            preds = torch.argmax(logits, dim=-1)  # (batch, 5)
            train_acc += (preds == yb).float().sum().item()
            total += yb.numel()

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = sum(criterion(logits[:, step, :], yb[:, step]) for step in range(5)) / 5.0
                val_loss += loss.item() * xb.size(0)
                preds = torch.argmax(logits, dim=-1)
                val_acc += (preds == yb).float().sum().item()
                val_total += yb.numel()

        train_loss /= len(train_loader.dataset)
        train_acc /= total if total > 0 else 1
        val_loss /= len(val_loader.dataset)
        val_acc /= val_total if val_total > 0 else 1

        scheduler.step(val_loss)

        logging.info(f"📚 Epoch {epoch+1}/{epochs}:")
        logging.info(f"  Train Loss: {train_loss:.6f} | Acc: {train_acc:.4f}")
        logging.info(f"  Val Loss:   {val_loss:.6f} | Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if (epoch % 5 == 0) or (epoch == 0): 
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"libra_v56_epoch_{epoch+1}_val_loss_{val_loss:.4f}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                logging.info(f"💾 Saved checkpoint: {checkpoint_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"⏳ No improvement for {epochs_no_improve}/{patience} epochs.")
            if epochs_no_improve >= patience:
                logging.info(f"🛑 Early stopping at epoch {epoch+1}")
                break

    final_model_path = os.path.join(CHECKPOINT_DIR, "libra_v56_best_model.pt")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"🏆 Best model (or last trained) saved to: {final_model_path}")

    torch.save(model.state_dict(), MODEL_PATH)
    if upload_model_with_retry():
        logging.info("🚀 Retrained model uploaded to Cloudinary")
    else:
        logging.error("❌ Failed to upload retrained model")

    return model
