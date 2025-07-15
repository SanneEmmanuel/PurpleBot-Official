import os
import time
import requests
import zipfile
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# --- Cloudinary dependencies ---
import cloudinary
import cloudinary.uploader
import cloudinary.api

class MultiPatternRNN(nn.Module):
    """
    A multi-pattern learning RNN, using GRU + Attention for pattern memory.
    Captures both local and dynamic (long-term) dependencies.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=3, attn_heads=4):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.15)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=attn_heads, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, h = self.gru(x)
        # Apply attention: Query=last, Key/Value=all
        query = out[:, -1:, :]  # (batch, 1, hidden)
        attn_out, _ = self.attn(query, out, out)  # (batch, 1, hidden)
        attn_out = attn_out.squeeze(1)
        out = self.fc(attn_out)
        out = self.sigmoid(out)
        return out  # Probability of +1 tick

class Libra6:
    def __init__(self, device=None, model_path=None):
        self.model = MultiPatternRNN()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.last_prices = []
        self.last_diffs = []
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
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

    @staticmethod
    def prices_to_diffs(prices):
        """Convert price series to normalized tick differences (-1, +1)."""
        prices = np.asarray(prices)
        diffs = np.sign(np.diff(prices))
        # Ensure output is int and only -1, +1
        diffs[diffs >= 1] = 1
        diffs[diffs <= -1] = -1
        diffs[diffs == 0] = 0  # Flat ticks can be 0 if needed, but we focus on -1,+1 for this design
        return diffs.astype(int)

    @staticmethod
    def diffs_to_price(last_price, diffs, tick_size=0.1):
        """Convert diffs back to price given last price."""
        return float(last_price + np.sum(diffs) * tick_size)

    def update(self, prices):
        """
        Accepts an array of prices and updates internal state.
        Only the latest 301 prices are kept (since 300 diffs = 301 prices).
        """
        prices = list(prices)
        self.last_prices.extend(prices)
        if len(self.last_prices) > 301:
            self.last_prices = self.last_prices[-301:]
        if len(self.last_prices) >= 2:
            self.last_diffs = self.prices_to_diffs(self.last_prices)
        else:
            self.last_diffs = []

    def predictWithConfidence(self, num_ticks=1, tick_size=0.1):
        if len(self.last_prices) < 2 or len(self.last_diffs) < 300:
            raise ValueError("Need at least 301 prices for prediction!")
        current_diffs = list(self.last_diffs[-300:])
        last_price = self.last_prices[-1]
        preds = []
        confidences = []
        prices = []
        for _ in range(num_ticks):
            inp = np.array(current_diffs).reshape(1, 300, 1).astype(np.float32)
            inp = torch.tensor(inp, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                prob = self.model(inp).item()
                next_diff = 1 if prob > 0.5 else -1
            preds.append(next_diff)
            confidences.append(prob if next_diff == 1 else 1 - prob)
            current_diffs.append(next_diff)
            if len(current_diffs) > 300:
                current_diffs = current_diffs[-300:]
            last_price = last_price + next_diff * tick_size
            prices.append(round(last_price, 6))
        return {
            "diffs": preds,
            "prices": prices,
            "confidences": confidences  # List of confidence values per prediction
        }

    def predict(self, num_ticks=1, tick_size=0.1):
        """
        Predict next num_ticks ticks, returning both diffs and absolute prices.
        Returns: {"diffs": [...], "prices": [...]}
        """
        if len(self.last_prices) < 2 or len(self.last_diffs) < 300:
            raise ValueError("Need at least 301 prices for prediction!")
        current_diffs = list(self.last_diffs[-300:])  # Always use last 300 diffs
        last_price = self.last_prices[-1]
        preds = []
        prices = []
        for _ in range(num_ticks):
            inp = np.array(current_diffs).reshape(1, 300, 1).astype(np.float32)
            inp = torch.tensor(inp, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                prob = self.model(inp).item()
                next_diff = 1 if prob > 0.5 else -1
            preds.append(next_diff)
            current_diffs.append(next_diff)
            if len(current_diffs) > 300:
                current_diffs = current_diffs[-300:]
            last_price = last_price + next_diff * tick_size
            prices.append(round(last_price, 6))
        return {"diffs": preds, "prices": prices}

    def continuous_train(self, price_seqs, batch_size=32, epochs=1, tick_size=0.1, lr=1e-3):
        """
        Continual training: price_seqs is a list of price arrays, each at least 301 long.
        Each (seq) is converted into (diffs, target) samples.
        """
        X = []
        Y = []
        for seq in price_seqs:
            seq = np.asarray(seq)
            if len(seq) < 301:
                continue
            diffs = self.prices_to_diffs(seq)
            # For each time t, use last 300 diffs to predict next diff
            for t in range(300, len(diffs)):
                x = diffs[t-300:t]
                y = diffs[t]
                if y == 0:
                    continue  # skip flat
                X.append(x)
                Y.append(1 if y > 0 else 0)
        if not X:
            raise ValueError("No valid patterns for training.")
        X = np.stack(X).astype(np.float32)
        Y = np.array(Y).astype(np.float32)
        x_tensor = torch.tensor(X.reshape(-1, 300, 1), dtype=torch.float32, device=self.device)
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

    # --- Cloudinary integration functions (same as before) ---
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
        torch.save(self.model.state_dict(), os.path.join(self.CHECKPOINT_DIR, name))
        logging.info(f"💾 Checkpoint saved: {name}")

    def load_checkpoint(self, name="libra6.pt"):
        path = os.path.join(self.CHECKPOINT_DIR, name)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logging.info(f"✅ Loaded checkpoint: {name}")

# Example usage:
if __name__ == "__main__":
    model = Libra6()
    # Example: updating with prices
    prices = [8321.1, 8321.2, 8321.3, 8321.2, 8321.1, 8321.2, 8321.1, 8321.0]
    model.update(prices)
    # Predict the next 3 price points (using last 301 prices)
    try:
        result = model.predict(num_ticks=3, tick_size=0.1)
        print("Next diff sequence:", result["diffs"])
        print("Next predicted prices:", result["prices"])
    except Exception as e:
        print(f"Prediction error: {e}")
    # For continual learning (training on new sequences)
    # model.continuous_train([prices], epochs=3)
    # model.upload_model_to_cloudinary()
    # model.download_model_from_cloudinary()
