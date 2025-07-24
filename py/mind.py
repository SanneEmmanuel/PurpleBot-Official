import os
import time
import requests
import zipfile
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils import weight_norm
from loguru import logger
import normflows as nf
from torch.utils.data import TensorDataset, DataLoader

import cloudinary
import cloudinary.uploader
import cloudinary.api
import cloudinary.utils

# =========================
# Cloudinary Configuration
# =========================
cloudinary.config(
    cloud_name="dj4bwntzb",
    api_key="354656419316393",
    api_secret="M-Trl9ltKDHyo1dIP2AaLOG-WPM",
    secure=True
)

# =========================
# Model Components
# =========================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x

class TCNBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            padding = (kernel_size - 1) * dilation
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, 
                                  stride=1, dilation=dilation, padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MindModel(nn.Module):
    def __init__(self, input_size=2, output_size=2, num_channels=[32, 64, 128], kernel_size=7, dropout=0.2):
        super().__init__()
        self.tcn = TCN(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        nn.init.normal_(self.linear.weight, 0, 0.01)

    def forward(self, x):
        # x: (batch, seq, feat)
        x = x.transpose(1, 2)  # (batch, feat, seq)
        tcn_out = self.tcn(x)
        # Use the last time step
        last_step = tcn_out[:, :, -1]
        return self.linear(last_step)

class FlowWrapper:
    def __init__(self, dim=2, num_layers=5, hidden_units=32, device=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dim = dim

        base = nf.distributions.base.DiagGaussian(dim)
        flows = []
        for i in range(num_layers):
            arn = nf.nets.MLP([dim, hidden_units, hidden_units, dim * 3],
                              init_zeros=True, output_fn="sigmoid")
            flows.append(nf.flows.MaskedAffineAutoregressive(
                features=dim,
                hidden_features=hidden_units,
                context_features=None,
                num_blocks=2,
                use_residual_blocks=False,
                random_mask=False,
                activation=torch.nn.ReLU,
                dropout_probability=0.0,
                use_batch_norm=False
            ))
            flows.append(nf.flows.Permute(dim, mode='swap'))
        self.model = nf.NormalizingFlow(base, flows).to(self.device)

    def fit(self, data, epochs=100, batch_size=512, lr=0.01):
        """Train flow model using maximum likelihood"""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                loss = -self.model.log_prob(x).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            if (epoch + 1) % 10 == 0:
                logger.info(f"Flow Epoch [{epoch+1}/{epochs}], NLL: {avg_loss:.4f}")

    def transform(self, data):
        """Transform data to latent space"""
        self.model.eval()
        with torch.no_grad():
            data_tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
            latent, _ = self.model.inverse(data_tensor)
        return latent.cpu().numpy()

    def inverse_transform(self, latent):
        """Transform latent samples back to data space"""
        self.model.eval()
        with torch.no_grad():
            latent_tensor = torch.tensor(latent, dtype=torch.float32, device=self.device)
            samples, _ = self.model.forward(latent_tensor)
        return samples.cpu().numpy()

# =========================
# Mind Class
# =========================

class Mind:
    def __init__(self, sequence_length=20, device=None, download_on_init=False, upload_on_fail=False):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.sequence_length = sequence_length
        self.model = MindModel().to(self.device)
        self.flow = FlowWrapper(device=self.device)
        self.model_loaded_successfully = False

        self.MODEL_PUBLIC_ID = "mind_hl_model_v1.zip"
        self.MODEL_LOCAL_PATH = "/tmp/mind.pt"
        self.ZIP_LOCAL_PATH = "/tmp/mind.zip"

        logger.info(f"🧠 Mind initialized on device: {self.device}")

        if download_on_init:
            self.awaken()
        if download_on_init and upload_on_fail and not self.model_loaded_successfully:
            logger.warning("Failed to awaken Mind. Uploading new untrained model.")
            self.sleep()

    def _prepare_sequences(self, data: np.ndarray):
        n = len(data)
        if n <= self.sequence_length:
            raise ValueError("Data length must be greater than sequence_length.")
        seq_size = n - self.sequence_length
        indices = np.arange(seq_size)[:, None] + np.arange(self.sequence_length)
        sequences = data[indices].reshape(seq_size, self.sequence_length, -1)
        labels = data[self.sequence_length:]
        return sequences, labels

    def learn(self, data: np.ndarray, epochs=50, lr=0.001, batch_size=32, flow_epochs=100):
        if not isinstance(data, np.ndarray) or len(data) < self.sequence_length + 1:
            raise ValueError("Training data must be a numpy array with sufficient length.")
        logger.info(f"Mind learning from {len(data)} points for {epochs} epochs...")

        # Train normalizing flow
        logger.info("Training normalizing flow...")
        self.flow.fit(data, epochs=flow_epochs, batch_size=min(512, len(data)))
        data_transformed = self.flow.transform(data)

        # Prepare sequences and train TCN
        self.model.train()
        X, y = self._prepare_sequences(data_transformed)
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32, device=self.device),
            torch.tensor(y, dtype=torch.float32, device=self.device)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        epoch_losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            epoch_losses.append(avg_loss)
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                logger.info(f"TCN Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        logger.success("Learning complete.")
        return {"epoch_losses": epoch_losses, "final_loss": epoch_losses[-1]}

    def predict(self, last_20_ohlc):
        logger.info("Making prediction...")
        self.model.eval()
        arr = np.array(last_20_ohlc, dtype=np.float32)
        if arr.shape != (self.sequence_length, 2):
            raise ValueError(f"Input must be shape ({self.sequence_length}, 2)")
        try:
            transformed = self.flow.transform(arr)
            with torch.no_grad():
                pred = self.model(torch.tensor(transformed[None], dtype=torch.float32, device=self.device))
                prediction = self.flow.inverse_transform(pred.cpu().numpy())
        except Exception as e:
            raise RuntimeError("Flow not trained. Train or load model first.") from e
        return {"Predicted High": float(prediction[0, 0]), "Predicted Low": float(prediction[0, 1])}

    def sleep(self, max_attempts=3):
        logger.info("Saving model to cloud...")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'flow_state_dict': self.flow.model.state_dict()
        }, self.MODEL_LOCAL_PATH)

        try:
            cloudinary.api.delete_resources([self.MODEL_PUBLIC_ID.replace(".zip", "")], resource_type='raw')
            logger.info("🗑️ Old model deleted.")
        except Exception as e:
            logger.warning(f"⚠️ Delete failed: {e}")

        with zipfile.ZipFile(self.ZIP_LOCAL_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(self.MODEL_LOCAL_PATH, os.path.basename(self.MODEL_LOCAL_PATH))

        for attempt in range(max_attempts):
            try:
                cloudinary.uploader.upload(
                    self.ZIP_LOCAL_PATH,
                    resource_type='raw',
                    public_id=self.MODEL_PUBLIC_ID,
                    overwrite=True
                )
                logger.success("Model saved to cloud.")
                return True
            except Exception as e:
                logger.warning(f"Upload attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
        logger.error("All upload attempts failed.")
        return False

    def awaken(self, max_attempts=3):
        logger.info("Loading model from cloud...")
        for attempt in range(max_attempts):
            try:
                url, _ = cloudinary.utils.cloudinary_url(
                    self.MODEL_PUBLIC_ID, resource_type='raw', sign_url=True)
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                with open(self.ZIP_LOCAL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                with zipfile.ZipFile(self.ZIP_LOCAL_PATH, 'r') as zf:
                    zf.extractall(os.path.dirname(self.MODEL_LOCAL_PATH))
                state = torch.load(self.MODEL_LOCAL_PATH, map_location=self.device)
                self.model.load_state_dict(state['model_state_dict'])
                self.flow = FlowWrapper(device=self.device)
                self.flow.model.load_state_dict(state['flow_state_dict'])
                self.flow.model.to(self.device)
                self.model.eval()
                self.model_loaded_successfully = True
                logger.success("Model loaded successfully.")
                return True
            except Exception as e:
                logger.warning(f"Download attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
        logger.error("All download attempts failed.")
        self.model_loaded_successfully = False
        return False

# =========================
# Example Usage / Testing
# =========================

def generate_synthetic_data(num_points=500):
    """
    Generates synthetic OHLC-like data for testing.
    Returns numpy array of shape (num_points, 2) where [:,0]=high, [:,1]=low
    """
    time_arr = np.arange(0, num_points, 1)
    series = np.sin(time_arr / 20) * 75 + time_arr * 0.1 + np.random.randn(num_points) * 15
    lows = series - np.random.rand(num_points) * 5
    highs = series + np.random.rand(num_points) * 5
    return np.stack([highs, lows], axis=1)

if __name__ == '__main__':
    mind = Mind(sequence_length=20, download_on_init=True, upload_on_fail=True)
    if not mind.model_loaded_successfully:
        logger.info("Training new model...")
        train_results = mind.learn(
            generate_synthetic_data(1000),
            epochs=50,
            flow_epochs=100
        )
        mind.sleep()
        logger.info(f"Training finished. Final loss: {train_results['final_loss']:.6f}")
    else:
        logger.info("Pre-trained model loaded.")

    test_data = generate_synthetic_data(100)[-20:]
    prediction = mind.predict(test_data)

    print("\n" + "="*40)
    print("🔮 MIND PREDICTION 🔮")
    print(f"Last Known High: {test_data[-1, 0]:.2f}")
    print(f"Last Known Low:  {test_data[-1, 1]:.2f}")
    print("----------------------------------------")
    print(f"Predicted Next High: {prediction['Predicted High']:.2f}")
    print(f"Predicted Next Low:  {prediction['Predicted Low']:.2f}")
    print("="*40 + "\n")
