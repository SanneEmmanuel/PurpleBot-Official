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

# --- New Fusion Model ---

class TCNPatternFusion(nn.Module):
    """
    A fusion model combining a lightweight TCN for local temporal patterns
    and a pattern frequency memory module to embed recurring subsequences.
    The outputs are fused and passed to an MLP for final classification.
    """
    def __init__(self, input_size=1, tcn_channels=[32, 64], pattern_len=5, num_top_patterns=50, dropout=0.2):
        super().__init__()
        # 1. Lightweight TCN Branch
        self.tcn = TCN(input_size, tcn_channels, kernel_size=7, dropout=dropout)
        tcn_output_size = tcn_channels[-1]

        # 2. Pattern Frequency Memory Module Parameters
        self.pattern_len = pattern_len
        self.num_top_patterns = num_top_patterns
        # Buffer to store the most frequent patterns (non-trainable but part of state_dict)
        self.register_buffer('top_patterns', torch.zeros(num_top_patterns, pattern_len))
        pattern_embedding_size = num_top_patterns

        # 3. Fusion Layer (MLP Classifier)
        fusion_input_size = tcn_output_size + pattern_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_input_size // 2, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def _calculate_pattern_embedding(self, x):
        # x shape: (batch_size, seq_len, 1)
        # self.top_patterns shape: (num_top_patterns, pattern_len)
        seq_len = x.shape[1]
        
        # Unfold the input sequence into overlapping windows of size `pattern_len`
        x_unfolded = x.squeeze(-1).unfold(dimension=1, size=self.pattern_len, step=1)
        # x_unfolded shape: (batch_size, num_windows, pattern_len)
        
        # Expand dims for broadcasting and comparison
        top_patterns_exp = self.top_patterns.unsqueeze(0).unsqueeze(0) # (1, 1, num_top, pattern_len)
        x_unfolded_exp = x_unfolded.unsqueeze(2) # (batch, num_win, 1, pattern_len)

        # Compare each window with each top pattern. `all` checks for perfect match.
        matches = torch.all(x_unfolded_exp == top_patterns_exp, dim=-1)
        
        # Sum matches over the sequence windows to get counts of each top pattern
        embedding = matches.float().sum(dim=1)
        
        # Normalize the embedding by the number of windows to keep magnitudes stable
        num_windows = seq_len - self.pattern_len + 1
        if num_windows > 0:
            embedding = embedding / num_windows

        return embedding

    def forward(self, x):
        # x shape: (batch, seq_len, features), e.g., (32, 300, 1)
        
        # TCN branch expects (batch, features, seq_len)
        x_tcn_in = x.permute(0, 2, 1)
        out_tcn = self.tcn(x_tcn_in)[:, :, -1] # Get feature vector from last time step

        # Pattern Memory branch
        out_pattern = self._calculate_pattern_embedding(x)
        
        # Concatenate features from both branches
        fused = torch.cat((out_tcn, out_pattern), dim=1)
        
        # Final prediction via MLP
        out = self.mlp(fused).squeeze(-1) # Squeeze final dimension
        out = self.sigmoid(out)
        return out
        
    def update_patterns(self, price_seqs, min_prices_for_seq):
        """Identifies and stores the most frequent subsequences from raw price data."""
        logging.info(f"🧠 Updating pattern memory from {len(price_seqs)} sequences...")
        
        def prices_to_diffs(prices):
            prices = np.asarray(prices)
            return np.sign(np.diff(prices)).astype(int)

        all_diffs = [prices_to_diffs(seq) for seq in price_seqs if len(seq) >= min_prices_for_seq]
        
        pattern_counts = Counter()
        for diff_seq in all_diffs:
            # Patterns are learned from non-zero movements
            diff_seq_nonzero = diff_seq[diff_seq != 0]
            if len(diff_seq_nonzero) < self.pattern_len:
                continue
            # Create a generator of patterns and update the counter
            patterns = (tuple(diff_seq_nonzero[i:i+self.pattern_len]) for i in range(len(diff_seq_nonzero) - self.pattern_len + 1))
            pattern_counts.update(patterns)
            
        if not pattern_counts:
            logging.warning("⚠️ No patterns found to update memory. Using existing or zeroed patterns.")
            return

        # Get the N most common patterns
        top_patterns_list = [p[0] for p in pattern_counts.most_common(self.num_top_patterns)]
        num_found = len(top_patterns_list)
        logging.info(f"Found {len(pattern_counts)} unique patterns. Storing top {num_found} in memory.")
        
        # Create a new tensor for the top patterns and update the buffer
        new_top_patterns = torch.zeros_like(self.top_patterns)
        if num_found > 0:
            new_top_patterns[:num_found] = torch.tensor(top_patterns_list, dtype=torch.float32)
        
        # Use .data to update buffer in place without breaking graph
        self.top_patterns.data = new_top_patterns.to(self.top_patterns.device)

class Libra6:
    """
    Libra6.4 is a predictive model for financial time-series data.
    It uses a TCN-PatternFusion model to predict tick movements, combining
    local feature extraction with frequency-based pattern analysis.
    """
    MIN_PRICES_FOR_PREDICTION = 301 # 300 diffs require 301 prices

    def __init__(self, device=None, model_path=None):
        self.model = TCNPatternFusion()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.last_prices = []
        # Cloudinary and path configs
        self.CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "dj4bwntzb")
        self.API_KEY = os.getenv("CLOUDINARY_API_KEY", "354656419316393")
        self.API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "M-Trl9ltKDHyo1dIP2AaLOG-WPM")
        self.MODEL_URL = f"https://res.cloudinary.com/{self.CLOUD_NAME}/raw/upload/v1/libra6_4.pt.zip"
        self.MODEL_PATH = "/tmp/libra6_4.pt"
        self.ZIP_PATH = "/tmp/libra6_4.pt.zip"
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
        if prices.ndim != 1:
            raise ValueError("Input prices must be a 1D array.")
        diffs = np.sign(np.diff(prices))
        return diffs.astype(int)

    @staticmethod
    def diffs_to_price(last_price, diffs, tick_size=0.1):
        """Convert diffs back to price given last price."""
        return float(last_price + np.sum(diffs) * tick_size)

    def update(self, prices):
        """Accepts an array of prices and updates internal state."""
        prices = list(prices)
        self.last_prices.extend(prices)
        if len(self.last_prices) > self.MIN_PRICES_FOR_PREDICTION:
            self.last_prices = self.last_prices[-self.MIN_PRICES_FOR_PREDICTION:]

    def predict(self, num_ticks=1, tick_size=0.1, prices=None):
        """Predict next num_ticks ticks, returning diffs and absolute prices."""
        return self.predictWithConfidence(num_ticks, tick_size, prices, with_confidence=False)

    def predictWithConfidence(self, num_ticks=1, tick_size=0.1, prices=None, with_confidence=True):
        """Predicts future ticks with confidence scores."""
        if prices is not None:
            if len(prices) < self.MIN_PRICES_FOR_PREDICTION:
                raise ValueError(
                    f"Provided prices must have at least {self.MIN_PRICES_FOR_PREDICTION} elements, "
                    f"but got {len(prices)}."
                )
            source_prices = list(prices)
        else:
            if len(self.last_prices) < self.MIN_PRICES_FOR_PREDICTION:
                raise ValueError(
                    f"Internal state has only {len(self.last_prices)} prices. "
                    f"Need at least {self.MIN_PRICES_FOR_PREDICTION}. Use 'update' method first."
                )
            source_prices = self.last_prices

        diffs = self.prices_to_diffs(source_prices)
        current_diffs = list(diffs[-(self.MIN_PRICES_FOR_PREDICTION - 1):])
        last_price = source_prices[-1]
        
        preds, confidences, predicted_prices = [], [], []
        
        self.model.eval()
        for _ in range(num_ticks):
            inp = np.array(current_diffs).reshape(1, len(current_diffs), 1).astype(np.float32)
            inp_tensor = torch.tensor(inp, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                prob = self.model(inp_tensor).item()
                next_diff = 1 if prob > 0.5 else -1
            
            preds.append(next_diff)
            if with_confidence:
                confidences.append(prob if next_diff == 1 else 1 - prob)
            
            current_diffs.append(next_diff)
            current_diffs.pop(0)
            
            last_price += next_diff * tick_size
            predicted_prices.append(round(last_price, 6))

        result = {"diffs": preds, "prices": predicted_prices}
        if with_confidence:
            result["confidences"] = confidences
            
        return result

    def continuous_train(self, price_seqs, batch_size=32, epochs=1, lr=1e-4):
        """
        Continual training on new price sequences. This first updates the model's 
        pattern memory based on the new data, then fine-tunes the network weights.
        """
        # Step 1: Update the pattern memory from the new data
        self.model.update_patterns(price_seqs, self.MIN_PRICES_FOR_PREDICTION)

        # Step 2: Prepare data for training the network
        X, Y = [], []
        seq_len = self.MIN_PRICES_FOR_PREDICTION - 1

        for seq in price_seqs:
            if len(seq) < self.MIN_PRICES_FOR_PREDICTION:
                logging.warning(f"Skipping sequence of length {len(seq)}, requires {self.MIN_PRICES_FOR_PREDICTION}.")
                continue
            
            diffs = self.prices_to_diffs(seq)
            
            for t in range(seq_len, len(diffs)):
                x_sample = diffs[t-seq_len:t]
                y_sample = diffs[t]
                if y_sample == 0: continue # Skip neutral ticks
                X.append(x_sample)
                Y.append(1 if y_sample > 0 else 0) # Binary target: 1 for up, 0 for down

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
            
            avg_loss = total_loss / len(dataset)
            logging.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
            
        self.model.eval()

    # --- Cloudinary integration functions with Retry ---
    def upload_model_to_cloudinary(self, max_attempts=3):
        logging.info("📦 Zipping model for upload...")
        torch.save(self.model.state_dict(), self.MODEL_PATH)
        with zipfile.ZipFile(self.ZIP_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(self.MODEL_PATH, os.path.basename(self.MODEL_PATH))

        cloudinary.config(cloud_name=self.CLOUD_NAME, api_key=self.API_KEY, api_secret=self.API_SECRET, secure=True)

        attempts = 0
        while attempts < max_attempts:
            try:
                result = cloudinary.uploader.upload(
                    self.ZIP_PATH, resource_type='raw', public_id=os.path.basename(self.ZIP_PATH),
                    overwrite=True, use_filename=True, tags=["libra6_model"]
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

    def download_model_from_cloudinary(self, max_attempts=3):
        logging.info(f"📥 Downloading model from {self.MODEL_URL}...")
        cloudinary.config(cloud_name=self.CLOUD_NAME, api_key=self.API_KEY, api_secret=self.API_SECRET, secure=True)

        attempts = 0
        while attempts < max_attempts:
            try:
                url, _ = cloudinary.utils.cloudinary_url(os.path.basename(self.ZIP_PATH), resource_type='raw', type='upload', sign_url=True, expires_at=int(time.time()) + 600)
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()

                os.makedirs(os.path.dirname(self.ZIP_PATH), exist_ok=True)
                with open(self.ZIP_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
                logging.info(f"✅ ZIP saved to {self.ZIP_PATH}")

                with zipfile.ZipFile(self.ZIP_PATH, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(self.MODEL_PATH))
                logging.info("✅ Model extracted.")

                self.model.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.device))
                self.model.eval()
                logging.info("✅ Model weights loaded into Libra6.4.")
                return # Success
            except (requests.exceptions.RequestException, zipfile.BadZipFile, IOError, Exception) as e:
                attempts += 1
                if attempts < max_attempts:
                    wait_time = 2 ** attempts
                    logging.warning(f"⚠️ Download/load failed (attempt {attempts}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"❌ Download failed after {max_attempts} attempts: {e}")
                    raise

    def save_checkpoint(self, name="libra6_4.pt"):
        path = os.path.join(self.CHECKPOINT_DIR, name)
        torch.save(self.model.state_dict(), path)
        logging.info(f"💾 Checkpoint saved: {path}")

    def load_checkpoint(self, name="libra6_4.pt"):
        path = name if os.path.isabs(name) else os.path.join(self.CHECKPOINT_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logging.info(f"✅ Loaded checkpoint: {name}")

# Example usage:
if __name__ == "__main__":
    # --- Initialization ---
    model = Libra6_4()
    
    # --- Generate dummy data for demonstration ---
    np.random.seed(42)
    initial_price = 8300.0
    long_price_sequence = initial_price + np.cumsum(np.random.choice([-0.1, 0.1], size=1000))

    # --- Prediction using a direct price array (on untrained model) ---
    print("\n--- Predicting with a direct price array (untrained model) ---")
    prices_for_prediction = long_price_sequence[:301]
    try:
        result = model.predictWithConfidence(num_ticks=3, tick_size=0.1, prices=prices_for_prediction)
        print("Input prices shape:", prices_for_prediction.shape)
        print("Next diff sequence:", result["diffs"])
        print("Next predicted prices:", result["prices"])
        print("Confidence scores:", [round(c, 2) for c in result["confidences"]])
    except ValueError as e:
        print(f"Prediction error: {e}")

    # --- Continual Learning (Training) ---
    print("\n--- Training the model (updates patterns and weights) ---")
    training_sequences = [long_price_sequence]
    model.continuous_train(training_sequences, epochs=3, batch_size=16, lr=1e-4)
    print("Top patterns found:", model.model.top_patterns.data[0:5]) # Display a few learned patterns

    # --- Save and Load Model ---
    print("\n--- Saving and Loading Model Checkpoint ---")
    model.save_checkpoint("my_libra6_4_model.pt")
    
    new_model = Libra6_4()
    new_model.load_checkpoint("my_libra6_4_model.pt")
    print("Libra6.4 model loaded successfully into new instance.")

    # --- Prediction after training ---
    print("\n--- Predicting again with the trained model ---")
    try:
        # Using the same initial prices, but now with a trained model
        result = new_model.predictWithConfidence(num_ticks=3, tick_size=0.1, prices=prices_for_prediction)
        print("Next diff sequence (post-training):", result["diffs"])
        print("Next predicted prices (post-training):", result["prices"])
        print("Confidence scores (post-training):", [round(c, 2) for c in result["confidences"]])
    except ValueError as e:
        print(f"Prediction error: {e}")
