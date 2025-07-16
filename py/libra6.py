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

# --- TCN Components ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock(nn.Module):
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
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# --- Optimized Fusion Model ---
class MultiScaleTCNPatternFusion(nn.Module):
    """
    Combines a TCN with a multi-scale pattern embedding module. Pattern matching
    is accelerated by converting patterns into unique integers for fast comparison.
    """
    def __init__(self, input_size=1, tcn_channels=[32, 64], pattern_scales=[3, 5, 7], num_top_patterns=50, dropout=0.2):
        super().__init__()
        self.pattern_scales = pattern_scales
        self.num_top_patterns = num_top_patterns
        self.tcn = TCN(input_size, tcn_channels, kernel_size=7, dropout=dropout)
        tcn_output_size = tcn_channels[-1]

        total_pattern_embedding_size = 0
        for scale in pattern_scales:
            self.register_buffer(f'top_patterns_{scale}', torch.zeros(num_top_patterns, dtype=torch.int64))
            self.register_buffer(f'base_powers_{scale}', 2**torch.arange(scale, dtype=torch.int64))
            total_pattern_embedding_size += num_top_patterns

        fusion_input_size = tcn_output_size + total_pattern_embedding_size
        self.mlp = nn.Sequential(nn.Linear(fusion_input_size, fusion_input_size // 2), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(fusion_input_size // 2, 1))
        self.sigmoid = nn.Sigmoid()

    def _calculate_pattern_embedding_optimized(self, x, scale):
        x_binary = (x.squeeze(-1) + 1) / 2
        x_unfolded = x_binary.unfold(dimension=1, size=scale, step=1).long()
        base_powers = getattr(self, f'base_powers_{scale}')
        window_integers = x_unfolded @ base_powers
        top_pattern_integers = getattr(self, f'top_patterns_{scale}')
        matches = window_integers.unsqueeze(-1) == top_pattern_integers
        embedding = matches.float().sum(dim=1)
        num_windows = x.shape[1] - scale + 1
        if num_windows > 0:
            embedding = embedding / num_windows
        return embedding

    def forward(self, x):
        out_tcn = self.tcn(x.permute(0, 2, 1))[:, :, -1]
        pattern_embeddings = [self._calculate_pattern_embedding_optimized(x, scale) for scale in self.pattern_scales]
        fused = torch.cat([out_tcn] + pattern_embeddings, dim=1)
        out = self.mlp(fused).squeeze(-1)
        return self.sigmoid(out)

    def update_patterns(self, price_seqs, min_prices_for_seq):
        logging.info(f"🧠 Updating multi-scale pattern memory from {len(price_seqs)} sequences...")
        all_diffs = [np.sign(np.diff(np.asarray(seq))).astype(int) for seq in price_seqs if len(seq) >= min_prices_for_seq]

        for scale in self.pattern_scales:
            pattern_counts = Counter()
            base_powers = getattr(self, f'base_powers_{scale}').cpu().numpy()
            for diff_seq in all_diffs:
                diff_seq_nonzero = diff_seq[diff_seq != 0]
                if len(diff_seq_nonzero) < scale: continue
                binary_seq = (diff_seq_nonzero + 1) // 2
                for i in range(len(binary_seq) - scale + 1):
                    pattern_int = np.dot(binary_seq[i:i+scale], base_powers)
                    pattern_counts[pattern_int] += 1
            
            if not pattern_counts: continue
            top_patterns_list = [p[0] for p in pattern_counts.most_common(self.num_top_patterns)]
            buffer_name = f'top_patterns_{scale}'
            new_top_patterns = torch.zeros(self.num_top_patterns, dtype=torch.int64)
            new_top_patterns[:len(top_patterns_list)] = torch.tensor(top_patterns_list, dtype=torch.int64)
            getattr(self, buffer_name).data = new_top_patterns.to(self.device)

class Libra6:
    MIN_PRICES_FOR_PREDICTION = 301

    def __init__(self, device=None, model_path=None, download_on_init=False, upload_on_fail=False):
        self.model = MultiScaleTCNPatternFusion()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.last_prices = []
        self.model_loaded_successfully = False

        self.CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "dj4bwntzb")
        self.API_KEY = os.getenv("CLOUDINARY_API_KEY", "354656419316393")
        self.API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "M-Trl9ltKDHyo1dIP2AaLOG-WPM")
        self.MODEL_PATH = "/tmp/libra7.pt"
        self.ZIP_PATH = "/tmp/libra7.pt.zip"
        self.CHECKPOINT_DIR = "/tmp/checkpoints"
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        logging.basicConfig(level=logging.INFO)
        logging.info(f"🔌 Using device: {self.device} | Model: {self.model.__class__.__name__}")

        if model_path:
            try:
                self.load_checkpoint(model_path)
            except FileNotFoundError as e:
                logging.error(f"Local checkpoint load failed: {e}")
        elif download_on_init:
            self.download_model_from_cloudinary()

        if download_on_init and upload_on_fail and not self.model_loaded_successfully:
            logging.warning("Failed to load model from Cloudinary. Uploading new, untrained model as a baseline.")
            self.upload_model_to_cloudinary()

    def download_model_from_cloudinary(self, max_attempts=3):
        cloudinary.config(cloud_name=self.CLOUD_NAME, api_key=self.API_KEY, api_secret=self.API_SECRET)
        for attempt in range(max_attempts):
            try:
                url, _ = cloudinary.utils.cloudinary_url(os.path.basename(self.ZIP_PATH), resource_type='raw', sign_url=True)
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                with open(self.ZIP_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
                with zipfile.ZipFile(self.ZIP_PATH, 'r') as zf: zf.extractall(os.path.dirname(self.MODEL_PATH))
                self.load_checkpoint(self.MODEL_PATH)
                logging.info("✅ Successfully downloaded and loaded model from Cloudinary.")
                return True
            except Exception as e:
                logging.warning(f"⚠️ Cloudinary download attempt {attempt + 1}/{max_attempts} failed: {e}")
                if attempt < max_attempts - 1: time.sleep(2 ** (attempt + 1))
        logging.error("❌ All attempts to download model from Cloudinary failed.")
        self.model_loaded_successfully = False
        return False


    def upload_model_to_cloudinary(self, max_attempts=3):
        self.save_checkpoint(self.MODEL_PATH)
        with zipfile.ZipFile(self.ZIP_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(self.MODEL_PATH, os.path.basename(self.MODEL_PATH))
        cloudinary.config(cloud_name=self.CLOUD_NAME, api_key=self.API_KEY, api_secret=self.API_SECRET)
        for attempt in range(max_attempts):
            try:
                cloudinary.uploader.upload(self.ZIP_PATH, resource_type='raw', public_id=os.path.basename(self.ZIP_PATH), overwrite=True)
                logging.info("✅ Model successfully uploaded to Cloudinary.")
                return True
            except Exception as e:
                logging.warning(f"⚠️ Cloudinary upload attempt {attempt + 1}/{max_attempts} failed: {e}")
                if attempt < max_attempts - 1: time.sleep(2 ** (attempt + 1))
        logging.error("❌ All attempts to upload model to Cloudinary failed.")
        return False
        
    @staticmethod
    def prices_to_diffs(prices):
        return np.sign(np.diff(np.asarray(prices))).astype(int)

    def update(self, prices):
        self.last_prices.extend(list(prices))
        if len(self.last_prices) > self.MIN_PRICES_FOR_PREDICTION:
            self.last_prices = self.last_prices[-self.MIN_PRICES_FOR_PREDICTION:]

    def predictWithConfidence(self, num_ticks=1, tick_size=0.1, prices=None):
        if prices is not None:
            if len(prices) < self.MIN_PRICES_FOR_PREDICTION: raise ValueError(f"Need {self.MIN_PRICES_FOR_PREDICTION} prices, got {len(prices)}")
            source_prices = list(prices)
        else:
            if len(self.last_prices) < self.MIN_PRICES_FOR_PREDICTION: raise ValueError(f"Need {self.MIN_PRICES_FOR_PREDICTION} prices, have {len(self.last_prices)}")
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
                confidences.append(prob if next_diff == 1 else 1 - prob)
                current_diffs.append(next_diff)
                current_diffs.pop(0)
                last_price += next_diff * tick_size
                predicted_prices.append(round(last_price, 6))
        return {"diffs": preds, "prices": predicted_prices, "confidences": confidences}

    def continuous_train(self, price_seqs, batch_size=32, epochs=1, lr=1e-4):
        self.model.update_patterns(price_seqs, self.MIN_PRICES_FOR_PREDICTION)
        X, Y = [], []
        seq_len = self.MIN_PRICES_FOR_PREDICTION - 1
        for seq in price_seqs:
            if len(seq) < self.MIN_PRICES_FOR_PREDICTION: continue
            diffs = self.prices_to_diffs(seq)
            for t in range(seq_len, len(diffs)):
                if diffs[t] == 0: continue
                X.append(diffs[t-seq_len:t])
                Y.append(1 if diffs[t] > 0 else 0)

        if not X:
            logging.error("No valid training samples created.")
            return {"final_loss": -1, "trained_samples": 0}

        dataset = torch.utils.data.TensorDataset(torch.tensor(np.stack(X), dtype=torch.float32).unsqueeze(-1),
                                                 torch.tensor(np.array(Y), dtype=torch.float32))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        self.model.train()
        final_epoch_loss = 0.0
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            final_epoch_loss = total_loss / len(dataset)
            logging.info(f"Epoch {epoch+1}/{epochs} | Loss: {final_epoch_loss:.6f}")
        self.model.eval()
        return {"final_loss": final_epoch_loss, "trained_samples": len(dataset)}

    def save_checkpoint(self, name="libra7.pt"):
        path = name if os.path.isabs(name) else os.path.join(self.CHECKPOINT_DIR, name)
        torch.save(self.model.state_dict(), path)
        logging.info(f"💾 Checkpoint saved: {path}")

    def load_checkpoint(self, name="libra7.pt"):
        path = name if os.path.isabs(name) else os.path.join(self.CHECKPOINT_DIR, name)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        self.model_loaded_successfully = True
        logging.info(f"✅ Loaded checkpoint: {name}")


if __name__ == "__main__":
    print("\n--- Initializing a new model instance ---")
    # Set download_on_init to True to attempt to fetch the latest model from the cloud.
    # If it fails, upload_on_fail=True will save the new untrained model as a starting point.
    model = Libra6(download_on_init=True, upload_on_fail=True)

    print("\n--- Generating dummy data and training the model ---")
    np.random.seed(42)
    price_data = 8300.0 + np.cumsum(np.random.choice([-0.1, 0.1], size=2000))
    training_summary = model.continuous_train([price_data], epochs=2)
    print(f"✅ Training complete. Summary: {training_summary}")

    print("\n--- Saving the newly trained model locally and to the cloud ---")
    model.save_checkpoint("my_libra7_model.pt")
    model.upload_model_to_cloudinary()

    print("\n--- Predicting with the trained model ---")
    try:
        prediction_prices = price_data[:model.MIN_PRICES_FOR_PREDICTION]
        result = model.predictWithConfidence(num_ticks=5, prices=prediction_prices)
        print(f"Predicted prices: {result['prices']} with confidences {[round(c, 2) for c in result['confidences']]}")
    except ValueError as e:
        print(f"Prediction error: {e}")
