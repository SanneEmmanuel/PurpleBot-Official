import os
import time
import requests
import zipfile
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# --- Cloudinary dependencies ---
import cloudinary
import cloudinary.uploader
import cloudinary.api

# --- TCN Components ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

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
    Temporal Convolutional Network.
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

# --- Fusion Model ---

class TCNGRUFusion(nn.Module):
    """
    A fusion model combining a TCN and a GRU with an attention mechanism.
    The input is processed by both TCN and GRU branches, and their
    outputs are fused to make the final prediction.
    """
    def __init__(self, input_size=1, tcn_channels=[32, 64, 128], gru_hidden_size=64, gru_num_layers=2, attn_heads=4, dropout=0.2):
        super().__init__()
        # TCN Branch
        self.tcn = TCN(input_size, tcn_channels, kernel_size=7, dropout=dropout)
        tcn_output_size = tcn_channels[-1]

        # GRU Branch
        self.gru = nn.GRU(input_size, gru_hidden_size, gru_num_layers, batch_first=True, dropout=dropout)
        
        # Fusion Layer
        fusion_input_size = tcn_output_size + gru_hidden_size
        self.attn = nn.MultiheadAttention(embed_dim=fusion_input_size, num_heads=attn_heads, batch_first=True)
        
        # Output Layer
        self.fc = nn.Linear(fusion_input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len, features) -> e.g., (32, 300, 1)
        
        # TCN branch expects (batch, features, seq_len)
        x_tcn = x.permute(0, 2, 1)
        out_tcn = self.tcn(x_tcn)
        # TCN output is (batch, channels, seq_len), take the last time step
        out_tcn = out_tcn[:, :, -1]

        # GRU branch
        out_gru, _ = self.gru(x)
        # GRU output is (batch, seq_len, hidden_size), take the last time step
        out_gru = out_gru[:, -1, :]
        
        # Concatenate features from both branches
        fused = torch.cat((out_tcn, out_gru), dim=1)
        
        # Apply attention to the fused features
        # Attention expects (batch, seq_len, embed_dim), so we unsqueeze
        fused_attn_input = fused.unsqueeze(1)
        attn_out, _ = self.attn(fused_attn_input, fused_attn_input, fused_attn_input)
        attn_out = attn_out.squeeze(1)

        # Final prediction
        out = self.fc(attn_out)
        out = self.sigmoid(out)
        return out

class Libra6:
    """
    Libra6 is a predictive model for financial time-series data.
    It uses a TCN-GRU fusion model to predict tick movements.
    """
    MIN_PRICES_FOR_PREDICTION = 301 # 300 diffs require 301 prices

    def __init__(self, device=None, model_path=None):
        self.model = TCNGRUFusion()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.last_prices = []
        # Cloudinary and path configs
        self.CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "dj4bwntzb")
        self.API_KEY = os.getenv("CLOUDINARY_API_KEY", "354656419316393")
        self.API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "M-Trl9ltKDHyo1dIP2AaLOG-WPM")
        self.MODEL_URL = f"https://res.cloudinary.com/{self.CLOUD_NAME}/raw/upload/v1/libra6.pt.zip"
        self.MODEL_PATH = "/tmp/libra6.pt"
        self.ZIP_PATH = "/tmp/libra6.pt.zip"
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
        """
        Accepts an array of prices and updates internal state.
        Only the latest `MIN_PRICES_FOR_PREDICTION` prices are kept.
        """
        prices = list(prices)
        self.last_prices.extend(prices)
        if len(self.last_prices) > self.MIN_PRICES_FOR_PREDICTION:
            self.last_prices = self.last_prices[-self.MIN_PRICES_FOR_PREDICTION:]

    def predict(self, num_ticks=1, tick_size=0.1, prices=None):
        """
        Predict next num_ticks ticks, returning both diffs and absolute prices.
        """
        return self.predictWithConfidence(num_ticks, tick_size, prices, with_confidence=False)

    def predictWithConfidence(self, num_ticks=1, tick_size=0.1, prices=None, with_confidence=True):
        """
        Predicts future ticks with confidence scores.
        """
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
                    f"Need at least {self.MIN_PRICES_FOR_PREDICTION}. Use the 'update' method first."
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
        Continual training on new price sequences.
        """
        X, Y = [], []
        seq_len = self.MIN_PRICES_FOR_PREDICTION - 1

        for seq in price_seqs:
            if len(seq) < self.MIN_PRICES_FOR_PREDICTION:
                logging.warning(f"Skipping a sequence with length {len(seq)}, requires at least {self.MIN_PRICES_FOR_PREDICTION}.")
                continue
            
            diffs = self.prices_to_diffs(seq)
            
            for t in range(seq_len, len(diffs)):
                x = diffs[t-seq_len:t]
                y = diffs[t]
                if y == 0: continue
                X.append(x)
                Y.append(1 if y > 0 else 0)

        if not X:
            logging.error("No valid training samples could be created from the provided price sequences.")
            return

        X = np.stack(X).astype(np.float32)
        Y = np.array(Y).astype(np.float32)
        
        x_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(-1)
        y_tensor = torch.tensor(Y, dtype=torch.float32, device=self.device)
        
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self.model(xb).squeeze()
                loss = criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            
            avg_loss = total_loss / len(dataset)
            logging.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
            
        self.model.eval()

    # --- Cloudinary integration functions ---
    def upload_model_to_cloudinary(self, max_attempts=3):
        logging.info("📦 Zipping model for upload...")
        torch.save(self.model.state_dict(), self.MODEL_PATH)
        with zipfile.ZipFile(self.ZIP_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(self.MODEL_PATH, arcname="libra6.pt")

        cloudinary.config(
            cloud_name=self.CLOUD_NAME,
            api_key=self.API_KEY,
            api_secret=self.API_SECRET,
            secure=True
        )

        attempts = 0
        while attempts < max_attempts:
            try:
                result = cloudinary.uploader.upload(
                    self.ZIP_PATH,
                    resource_type='raw',
                    public_id="libra6.pt.zip",
                    overwrite=True,
                    use_filename=True,
                    tags=["libra6_model"]
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

    def download_model_from_cloudinary(self):
        logging.info("📥 Downloading model ZIP from Cloudinary...")
        cloudinary.config(
            cloud_name=self.CLOUD_NAME,
            api_key=self.API_KEY,
            api_secret=self.API_SECRET,
            secure=True
        )

        try:
            url, _ = cloudinary.utils.cloudinary_url(
                "libra6.pt.zip",
                resource_type='raw',
                type='upload',
                sign_url=True,
                expires_at=int(time.time()) + 600
            )

            response = requests.get(url, stream=True)
            response.raise_for_status()

            os.makedirs(os.path.dirname(self.ZIP_PATH), exist_ok=True)
            with open(self.ZIP_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info(f"✅ ZIP saved to {self.ZIP_PATH}")

            with zipfile.ZipFile(self.ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(self.MODEL_PATH))
            logging.info("✅ Model extracted.")

            self.model.load_state_dict(torch.load(self.MODEL_PATH, map_location=self.device))
            self.model.eval()
            logging.info("✅ Model weights loaded into Libra6.")
        except Exception as e:
            logging.error(f"❌ Download or extraction failed: {e}")
            raise

    def save_checkpoint(self, name="libra6.pt"):
        path = os.path.join(self.CHECKPOINT_DIR, name)
        torch.save(self.model.state_dict(), path)
        logging.info(f"💾 Checkpoint saved: {path}")

    def load_checkpoint(self, name="libra6.pt"):
        path = name if os.path.isabs(name) else os.path.join(self.CHECKPOINT_DIR, name)
        if not os.path.exists(path):
            logging.error(f"Checkpoint file not found at {path}")
            raise FileNotFoundError(f"Checkpoint file not found at {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logging.info(f"✅ Loaded checkpoint: {name}")

# Example usage:
if __name__ == "__main__":
    # --- Initialization ---
    model = Libra6()
    
    # --- Generate dummy data for demonstration ---
    np.random.seed(42)
    initial_price = 8300.0
    long_price_sequence = initial_price + np.cumsum(np.random.choice([-0.1, 0.1], size=500))

    # --- Prediction using a direct price array ---
    print("\n--- Predicting with a direct price array ---")
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
    print("\n--- Training the model ---")
    training_sequences = [long_price_sequence]
    model.continuous_train(training_sequences, epochs=3, batch_size=16, lr=1e-4)

    # --- Save and Load Model ---
    print("\n--- Saving and Loading Model Checkpoint ---")
    model.save_checkpoint("my_libra6_fusion_model.pt")
    
    new_model = Libra6()
    new_model.load_checkpoint("my_libra6_fusion_model.pt")
    print("Fusion model loaded successfully into new instance.")

    # --- Prediction after training ---
    print("\n--- Predicting again with the trained model ---")
    try:
        result = new_model.predictWithConfidence(num_ticks=3, tick_size=0.1, prices=prices_for_prediction)
        print("Next diff sequence (post-training):", result["diffs"])
        print("Next predicted prices (post-training):", result["prices"])
        print("Confidence scores (post-training):", [round(c, 2) for c in result["confidences"]])
    except ValueError as e:
        print(f"Prediction error: {e}")
