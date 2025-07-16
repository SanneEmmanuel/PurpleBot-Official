import os
import time
import requests
import zipfile
import torch
import logging
import numpy as np
import torch.nn as nn
from collections import Counter
from torch.nn.utils import weight_norm

# --- Cloudinary dependencies ---
import cloudinary
import cloudinary.uploader
import cloudinary.api

# --- TCN Components (Dependencies for the new model) ---
class Chomp1d(nn.Module):
    """A simple layer to remove the last 'chomp_size' elements from a 1D sequence."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock(nn.Module):
    """
    A single block of a Temporal Convolutional Network.
    It consists of two causal convolutions with weight normalization,
    ReLU activation, and dropout. A residual connection is also used.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """
    Temporal Convolutional Network. This is a generic TCN implementation
    that can be used as a component in larger models.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# --- New Optimized Fusion Model ---

class MultiScaleTCNPatternFusion(nn.Module):
    """
    An optimized fusion model that combines a lightweight TCN with a multi-scale
    pattern embedding module. The pattern matching is heavily vectorized ("SIMD-style")
    by converting patterns into unique integers for ultra-fast comparison.
    """
    def __init__(self, input_size=1, tcn_channels=[32, 64], pattern_scales=[3, 5, 7], num_top_patterns=50, dropout=0.2):
        super().__init__()
        self.pattern_scales = pattern_scales
        self.num_top_patterns = num_top_patterns

        # 1. Lightweight TCN Branch
        self.tcn = TCN(input_size, tcn_channels, kernel_size=7, dropout=dropout)
        tcn_output_size = tcn_channels[-1]

        # 2. Multi-Scale Pattern Frequency Module
        total_pattern_embedding_size = 0
        for scale in pattern_scales:
            # Buffer for integer-encoded patterns for each scale
            self.register_buffer(f'top_patterns_{scale}', torch.zeros(num_top_patterns, dtype=torch.int64))
            # Buffer for base powers used in vectorized integer conversion (e.g., [1, 2, 4, 8...])
            self.register_buffer(f'base_powers_{scale}', 2**torch.arange(scale, dtype=torch.int64))
            total_pattern_embedding_size += num_top_patterns

        # 3. Fusion Layer (MLP Classifier)
        fusion_input_size = tcn_output_size + total_pattern_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_input_size // 2, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def _calculate_pattern_embedding_optimized(self, x, scale):
        """Calculates pattern embedding for a single scale using integer conversion."""
        # Map diffs from {-1, 1} to {0, 1} for binary encoding.
        # Adding 1 and dividing by 2: (-1 -> 0, 1 -> 1)
        x_binary = (x.squeeze(-1) + 1) / 2
        
        # Unfold the input into overlapping windows
        x_unfolded = x_binary.unfold(dimension=1, size=scale, step=1).long()
        
        # Vectorized conversion of each window/pattern into a single integer.
        # This is the core "SIMD-style" optimization.
        base_powers = getattr(self, f'base_powers_{scale}')
        window_integers = x_unfolded @ base_powers
        
        # Retrieve the integer-encoded top patterns for this scale
        top_pattern_integers = getattr(self, f'top_patterns_{scale}')
        
        # Fast, vectorized comparison of integer codes.
        # (batch, num_windows, 1) == (1, 1, num_top_patterns)
        matches = window_integers.unsqueeze(-1) == top_pattern_integers
        
        # Sum matches over windows to get counts of each top pattern.
        embedding = matches.float().sum(dim=1)
        
        # Normalize
        num_windows = x.shape[1] - scale + 1
        if num_windows > 0:
            embedding = embedding / num_windows
            
        return embedding

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # TCN branch
        out_tcn = self.tcn(x.permute(0, 2, 1))[:, :, -1]

        # Pattern Memory branch (multi-scale)
        pattern_embeddings = []
        for scale in self.pattern_scales:
            embedding = self._calculate_pattern_embedding_optimized(x, scale)
            pattern_embeddings.append(embedding)
        
        # Fuse TCN features with all pattern scale embeddings
        fused = torch.cat([out_tcn] + pattern_embeddings, dim=1)
        
        # Final prediction via MLP
        out = self.mlp(fused).squeeze(-1)
        return self.sigmoid(out)

    def update_patterns(self, price_seqs, min_prices_for_seq):
        """Identifies, integer-encodes, and stores the most frequent patterns for each scale."""
        logging.info(f"🧠 Updating multi-scale pattern memory from {len(price_seqs)} sequences...")

        def prices_to_diffs(prices):
            return np.sign(np.diff(np.asarray(prices))).astype(int)

        all_diffs = [prices_to_diffs(seq) for seq in price_seqs if len(seq) >= min_prices_for_seq]
        
        for scale in self.pattern_scales:
            pattern_counts = Counter()
            base_powers = getattr(self, f'base_powers_{scale}').numpy()

            for diff_seq in all_diffs:
                # Patterns are learned from non-zero movements
                diff_seq_nonzero = diff_seq[diff_seq != 0]
                if len(diff_seq_nonzero) < scale:
                    continue
                
                # Map to {0, 1} for binary encoding
                binary_seq = (diff_seq_nonzero + 1) // 2
                
                # Efficiently find and count patterns
                for i in range(len(binary_seq) - scale + 1):
                    pattern_window = binary_seq[i:i+scale]
                    # Convert pattern to integer and count it
                    pattern_int = np.dot(pattern_window, base_powers)
                    pattern_counts[pattern_int] += 1
            
            if not pattern_counts:
                logging.warning(f"⚠️ No patterns of length {scale} found. Memory for this scale remains unchanged.")
                continue

            top_patterns_list = [p[0] for p in pattern_counts.most_common(self.num_top_patterns)]
            num_found = len(top_patterns_list)
            logging.info(f"Found {len(pattern_counts)} unique patterns for scale {scale}. Storing top {num_found}.")

            # Update the corresponding buffer with new integer-encoded patterns
            buffer_name = f'top_patterns_{scale}'
            new_top_patterns = torch.zeros(self.num_top_patterns, dtype=torch.int64)
            if num_found > 0:
                new_top_patterns[:num_found] = torch.tensor(top_patterns_list, dtype=torch.int64)
            
            getattr(self, buffer_name).data = new_top_patterns.to(self.device)

class Libra6:
    """
    Libra7: An optimized predictive model for financial time-series.
    It uses a Multi-Scale TCN-PatternFusion model with "SIMD-style" vectorized
    pattern matching for high-speed, accurate predictions.
    """
    MIN_PRICES_FOR_PREDICTION = 301 # 300 diffs require 301 prices

    def __init__(self, device=None, model_path=None):
        self.model = MultiScaleTCNPatternFusion()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.last_prices = []
        # Cloudinary and path configs
        self.CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "dj4bwntzb")
        self.API_KEY = os.getenv("CLOUDINARY_API_KEY", "354656419316393")
        self.API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "M-Trl9ltKDHyo1dIP2AaLOG-WPM")
        self.MODEL_URL = f"https://res.cloudinary.com/{self.CLOUD_NAME}/raw/upload/v1/libra7.pt.zip"
        self.MODEL_PATH = "/tmp/libra7.pt"
        self.ZIP_PATH = "/tmp/libra7.pt.zip"
        self.CHECKPOINT_DIR = "/tmp/checkpoints"
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        logging.basicConfig(level=logging.INFO)
        logging.info(f"🔌 Using device: {self.device}")
        logging.info(f"🧠 Model: {self.model.__class__.__name__}")

        if model_path:
            self.load_checkpoint(model_path)

    @staticmethod
    def prices_to_diffs(prices):
        """Convert price series to normalized tick differences (-1, 0, +1)."""
        prices = np.asarray(prices)
        if prices.ndim != 1: raise ValueError("Input prices must be a 1D array.")
        diffs = np.sign(np.diff(prices))
        return diffs.astype(int)

    def update(self, prices):
        """Accepts an array of prices and updates internal state."""
        self.last_prices.extend(list(prices))
        if len(self.last_prices) > self.MIN_PRICES_FOR_PREDICTION:
            self.last_prices = self.last_prices[-self.MIN_PRICES_FOR_PREDICTION:]

    def predictWithConfidence(self, num_ticks=1, tick_size=0.1, prices=None, with_confidence=True):
        """Predicts future ticks with confidence scores."""
        source_prices = []
        if prices is not None:
            if len(prices) < self.MIN_PRICES_FOR_PREDICTION:
                raise ValueError(f"Provided prices must have at least {self.MIN_PRICES_FOR_PREDICTION} elements, but got {len(prices)}.")
            source_prices = list(prices)
        else:
            if len(self.last_prices) < self.MIN_PRICES_FOR_PREDICTION:
                raise ValueError(f"Internal state has only {len(self.last_prices)} prices. Need at least {self.MIN_PRICES_FOR_PREDICTION}.")
            source_prices = self.last_prices

        diffs = self.prices_to_diffs(source_prices)
        current_diffs = list(diffs[-(self.MIN_PRICES_FOR_PREDICTION - 1):])
        last_price = source_prices[-1]
        
        preds, confidences, predicted_prices = [], [], []
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(num_ticks):
                inp = np.array(current_diffs).reshape(1, len(current_diffs), 1)
                inp_tensor = torch.tensor(inp, dtype=torch.float32, device=self.device)
                
                prob = self.model(inp_tensor).item()
                next_diff = 1 if prob > 0.5 else -1
                
                preds.append(next_diff)
                if with_confidence: confidences.append(prob if next_diff == 1 else 1 - prob)
                
                current_diffs.append(next_diff)
                current_diffs.pop(0)
                
                last_price += next_diff * tick_size
                predicted_prices.append(round(last_price, 6))

        result = {"diffs": preds, "prices": predicted_prices}
        if with_confidence: result["confidences"] = confidences
        return result

    def continuous_train(self, price_seqs, batch_size=32, epochs=1, lr=1e-4):
        """
        Continual training on new price sequences. This first updates the model's 
        multi-scale pattern memory, then fine-tunes the network weights.
        """
        # Step 1: Update the pattern memory from the new data
        self.model.update_patterns(price_seqs, self.MIN_PRICES_FOR_PREDICTION)

        # Step 2: Prepare data for training the network
        X, Y = [], []
        seq_len = self.MIN_PRICES_FOR_PREDICTION - 1
        for seq in price_seqs:
            if len(seq) < self.MIN_PRICES_FOR_PREDICTION: continue
            diffs = self.prices_to_diffs(seq)
            for t in range(seq_len, len(diffs)):
                y_sample = diffs[t]
                if y_sample == 0: continue
                X.append(diffs[t-seq_len:t])
                Y.append(1 if y_sample > 0 else 0)

        if not X:
            logging.error("No valid training samples created from price sequences.")
            return

        x_tensor = torch.tensor(np.stack(X), dtype=torch.float32, device=self.device).unsqueeze(-1)
        y_tensor = torch.tensor(np.array(Y), dtype=torch.float32, device=self.device)
        
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            
            logging.info(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(dataset):.6f}")
        self.model.eval()

    def save_checkpoint(self, name="libra7.pt"):
        path = os.path.join(self.CHECKPOINT_DIR, name)
        torch.save(self.model.state_dict(), path)
        logging.info(f"💾 Checkpoint saved: {path}")

    def load_checkpoint(self, name="libra7.pt"):
        path = name if os.path.isabs(name) else os.path.join(self.CHECKPOINT_DIR, name)
        if not os.path.exists(path): raise FileNotFoundError(f"Checkpoint not found: {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logging.info(f"✅ Loaded checkpoint: {name}")


if __name__ == "__main__":
    # --- Initialization ---
    model = Libra7()
    
    # --- Generate dummy data for demonstration ---
    np.random.seed(42)
    long_price_sequence = 8300.0 + np.cumsum(np.random.choice([-0.1, 0.1], size=2000))

    # --- Prediction on untrained model ---
    print("\n--- Predicting with untrained model ---")
    prices_for_prediction = long_price_sequence[:301]
    try:
        result = model.predictWithConfidence(num_ticks=3, tick_size=0.1, prices=prices_for_prediction)
        print(f"Next predicted prices: {result['prices']}")
    except ValueError as e:
        print(f"Prediction error: {e}")

    # --- Continual Learning (Training) ---
    print("\n--- Training the model (updates patterns and weights) ---")
    training_sequences = [long_price_sequence]
    model.continuous_train(training_sequences, epochs=2, batch_size=32, lr=1e-4)
    print("Training complete.")

    # --- Save and Load Model ---
    print("\n--- Saving and Loading Model Checkpoint ---")
    model.save_checkpoint("my_libra7_model.pt")
    new_model = Libra7()
    new_model.load_checkpoint("my_libra7_model.pt")
    print("Libra7 model loaded successfully into new instance.")

    # --- Prediction after training ---
    print("\n--- Predicting again with the trained model ---")
    try:
        result = new_model.predictWithConfidence(num_ticks=5, tick_size=0.1, prices=prices_for_prediction)
        print(f"Next diff sequence (post-training): {result['diffs']}")
        print(f"Next predicted prices (post-training): {result['prices']}")
        print(f"Confidence scores (post-training): {[round(c, 2) for c in result['confidences']]}")
    except ValueError as e:
        print(f"Prediction error: {e}")
