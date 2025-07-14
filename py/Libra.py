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
