import os
import time
import requests
import zipfile
import torch
import logging
import numpy as np
import torch.nn as nn
from torch.nn.utils import weight_norm
from sklearn.preprocessing import MinMaxScaler

import cloudinary
import cloudinary.uploader
import cloudinary.api

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TCNBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
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
        with torch.no_grad():
            self.conv1.weight.normal_(0, 0.01)
            self.conv2.weight.normal_(0, 0.01)
            if self.downsample is not None:
                self.downsample.weight.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1)*dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MindModel(nn.Module):
    def __init__(self, input_size=2, output_size=2, num_channels=[32, 64, 128], kernel_size=7, dropout=0.2):
        super().__init__()
        self.tcn = TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.normal_(0, 0.01)

    def forward(self, x):
        # x: (batch, seq_len, features)
        # TCN expects: (batch, features, seq_len)
        y1 = self.tcn(x.transpose(1, 2))
        return self.linear(y1[:, :, -1])


class Mind:
    def __init__(self, sequence_length=20, device=None, download_on_init=False, upload_on_fail=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = sequence_length
        self.model = MindModel(input_size=2, output_size=2)
        self.model.to(self.device)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.model_loaded_successfully = False

        self.CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "dj4bwntzb")
        self.API_KEY = os.getenv("CLOUDINARY_API_KEY", "354656419316393")
        self.API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "M-Trl9ltKDHyo1dIP2AaLOG-WPM")
        self.MODEL_PUBLIC_ID = "mind_hl_model_v1.zip"
        self.MODEL_LOCAL_PATH = "/tmp/mind.pt"
        self.ZIP_LOCAL_PATH = "/tmp/mind.zip"

        logging.info(f"🧠 Mind initialized on device: {self.device}")

        cloudinary.config(cloud_name=self.CLOUD_NAME, api_key=self.API_KEY, api_secret=self.API_SECRET, secure=True)

        if download_on_init:
            self.awaken()
        if download_on_init and upload_on_fail and not self.model_loaded_successfully:
            logging.warning("Failed to awaken Mind from Cloudinary. Uploading new, untrained Mind as a baseline.")
            self.sleep()

    def _prepare_sequences(self, data):
        sequences, labels = [], []
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i+self.sequence_length]
            label = data[i+self.sequence_length]
            sequences.append(seq)
            labels.append(label)
        return np.array(sequences), np.array(labels)

    def learn(self, data, epochs=50, lr=0.001, batch_size=32):
        """Trains the Mind on new data. Returns training losses and final stats."""
        if not isinstance(data, np.ndarray) or len(data) < self.sequence_length + 1:
            raise ValueError("Training data must be a numpy array with sufficient length.")

        logging.info(f"Mind is learning from {len(data)} data points for {epochs} epochs...")
        self.model.train()

        data_scaled = self.scaler.fit_transform(data)
        X, y = self._prepare_sequences(data_scaled)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        epoch_losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                y_pred = self.model(xb)
                loss = criterion(y_pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_avg_loss = epoch_loss / len(loader)
            epoch_losses.append(epoch_avg_loss)
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                logging.info(f"  Epoch [{epoch + 1}/{epochs}], Loss: {epoch_avg_loss:.6f}")
        logging.info("✅ Learning complete.")

        return {
            "epoch_losses": epoch_losses,
            "final_loss": epoch_losses[-1] if epoch_losses else None,
            "epochs": epochs
        }

    def predict(self, last_20_ohlc):
        """Predicts the next High and Low given the last 20 periods."""
        logging.info("Mind is predicting...")
        self.model.eval()
        if not isinstance(last_20_ohlc, (np.ndarray, list)):
            raise ValueError("Input must be a numpy array or list.")

        arr = np.array(last_20_ohlc)
        if arr.shape != (self.sequence_length, 2):
            raise ValueError(f"Input data must have shape ({self.sequence_length}, 2)")

        try:
            data_scaled = self.scaler.transform(arr)
        except Exception as e:
            raise RuntimeError("Scaler not fitted. You must train or load a model before predicting.") from e

        sequence = torch.tensor(data_scaled, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            prediction_scaled = self.model(sequence)
        prediction = self.scaler.inverse_transform(prediction_scaled.cpu().numpy())
        return {"Predicted High": float(prediction[0, 0]), "Predicted Low": float(prediction[0, 1])}

    def sleep(self, max_attempts=3):
        """Saves the Mind's state (memory) to Cloudinary."""
        logging.info("Mind is going to sleep. Saving memory to the cloud...")
        state = {
            'model_state_dict': self.model.state_dict(),
            'scaler_min': getattr(self.scaler, "min_", None),
            'scaler_scale': getattr(self.scaler, "scale_", None),
            'scaler_n_samples': getattr(self.scaler, "n_samples_seen_", None)
        }
        print(state)
        torch.save(model.state_dict(), self.MODEL_LOCAL_PATH)

        with zipfile.ZipFile(self.ZIP_LOCAL_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(self.MODEL_LOCAL_PATH, os.path.basename(self.MODEL_LOCAL_PATH))

        for attempt in range(max_attempts):
            try:
                cloudinary.uploader.upload(
                    self.ZIP_LOCAL_PATH,
                    resource_type='raw',
                    public_id=self.MODEL_PUBLIC_ID,
                    overwrite=True
                )
                logging.info("✅ Mind's memory successfully stored in the cloud.")
                return True
            except Exception as e:
                logging.warning(f"Cloudinary upload attempt {attempt + 1}/{max_attempts} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2 ** (attempt + 1))
        logging.error("❌ All attempts to upload Mind's memory failed.")
        return False

    def awaken(self, max_attempts=3):
        """Loads the Mind's state (memory) from Cloudinary."""
        logging.info("Mind is awakening. Loading memory from the cloud...")
        for attempt in range(max_attempts):
            try:
                url, _ = cloudinary.utils.cloudinary_url(self.MODEL_PUBLIC_ID, resource_type='raw', sign_url=True)
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                with open(self.ZIP_LOCAL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                with zipfile.ZipFile(self.ZIP_LOCAL_PATH, 'r') as zf:
                    zf.extractall(os.path.dirname(self.MODEL_LOCAL_PATH))

                state = torch.load(self.MODEL_LOCAL_PATH, map_location=self.device, weights_only=False)
                self.model.load_state_dict(state['model_state_dict'])
                if 'scaler_min' in state and 'scaler_scale' in state:
                    self.scaler.min_ = state['scaler_min']
                    self.scaler.scale_ = state['scaler_scale']
                    self.scaler.n_samples_seen_ = state['scaler_n_samples']

                else:
                    raise RuntimeError("Scaler parameters missing from checkpoint.")
                self.model.eval()
                self.model_loaded_successfully = True
                logging.info("✅ Mind has awakened successfully.")
                return True
            except Exception as e:
                logging.warning(f"Cloudinary download attempt {attempt + 1}/{max_attempts} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2 ** (attempt + 1))
        logging.error("❌ All attempts to awaken the Mind failed.")
        self.model_loaded_successfully = False
        return False


# --- Example Usage ---
if __name__ == '__main__':
    def generate_synthetic_data(num_points=500):
        time_arr = np.arange(0, num_points, 1)
        series = np.sin(time_arr / 20) * 75 + time_arr * 0.1 + np.random.randn(num_points) * 15
        lows = series - np.random.rand(num_points) * 5
        highs = series + np.random.rand(num_points) * 5
        return np.stack([highs, lows], axis=1)

    mind = Mind(sequence_length=20, download_on_init=True, upload_on_fail=True)

    if not mind.model_loaded_successfully:
        logging.info("No pre-trained Mind found. Training a new one.")
        historical_data = generate_synthetic_data(1000)
        train_results = mind.learn(historical_data, epochs=50)
        mind.sleep()
        logging.info(f"Training finished. Final loss: {train_results['final_loss']:.6f}")
    else:
        logging.info("Pre-trained Mind awakened. Ready for predictions.")

    prediction_data_source = generate_synthetic_data(100)
    data_for_prediction = prediction_data_source[-mind.sequence_length:]
    prediction = mind.predict(data_for_prediction)

    print("\n" + "="*40)
    print("🔮 MIND PREDICTION 🔮")
    print(f"Last Known High: {data_for_prediction[-1, 0]:.2f}")
    print(f"Last Known Low:  {data_for_prediction[-1, 1]:.2f}")
    print("----------------------------------------")
    print(f"Predicted Next High: {prediction['Predicted High']:.2f}")
    print(f"Predicted Next Low:  {prediction['Predicted Low']:.2f}")
    print("="*40 + "\n")
