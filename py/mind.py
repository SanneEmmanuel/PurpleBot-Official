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
# Your Cloudinary credentials are kept as they were.
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
    """A simple layer to remove padding from the end of a 1D sequence."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x

class TCNBlock(nn.Module):
    """
    A single block of a Temporal Convolutional Network (TCN),
    containing two convolutional layers with residual connections.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        # First convolutional layer
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolutional layer
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Downsample layer for the residual connection if input and output dimensions don't match
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """Initializes weights with a normal distribution."""
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass, rewritten for clarity and to resolve the original bug.
        """
        # First convolutional block
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second convolutional block
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """Temporal Convolutional Network model."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class MindModel(nn.Module):
    """The core predictive model, combining TCN with a final linear layer."""
    def __init__(self, input_size=2, output_size=2, num_channels=[32, 64, 128], kernel_size=7, dropout=0.2):
        super().__init__()
        self.tcn = TCN(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        nn.init.normal_(self.linear.weight, 0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        tcn_out = self.tcn(x)
        # Use the output of the last time step for prediction
        last_step = tcn_out[:, :, -1]
        return self.linear(last_step)

class FlowWrapper:
    """A wrapper for the Normalizing Flow model for data transformation."""
    def __init__(self, dim=2, num_layers=5, hidden_units=32, device=None):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dim = dim

        base = nf.distributions.base.DiagGaussian(dim)
        flows = []
        for _ in range(num_layers):
            flows.append(nf.flows.MaskedAffineAutoregressive(
                features=dim,
                hidden_features=hidden_units,
                num_blocks=2
            ))
            flows.append(nf.flows.Permute(dim, mode='swap'))
        self.model = nf.NormalizingFlow(base, flows).to(self.device)

    def fit(self, data: np.ndarray, epochs=100, batch_size=512, lr=0.01):
        """Trains the flow model using maximum likelihood estimation."""
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

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transforms data from the original space to the latent space."""
        self.model.eval()
        with torch.no_grad():
            data_tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
            latent, _ = self.model.inverse(data_tensor)
        return latent.cpu().numpy()

    def inverse_transform(self, latent: np.ndarray) -> np.ndarray:
        """Transforms data from the latent space back to the original space."""
        self.model.eval()
        with torch.no_grad():
            latent_tensor = torch.tensor(latent, dtype=torch.float32, device=self.device)
            samples, _ = self.model.forward(latent_tensor)
        return samples.cpu().numpy()

# =========================
# Mind Class
# =========================

class Mind:
    """The main class orchestrating the model, data, and cloud storage."""
    def __init__(self, sequence_length=20, device=None, download_on_init=False, upload_on_fail=False):
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.sequence_length = sequence_length
        self.model = MindModel().to(self.device)
        self.flow = FlowWrapper(device=self.device)
        self.model_loaded_successfully = False

        self.MODEL_PUBLIC_ID = "mind_hl_model_v1" # .zip is added by logic
        self.MODEL_LOCAL_PATH = "/tmp/mind.pt"
        self.ZIP_LOCAL_PATH = f"/tmp/{self.MODEL_PUBLIC_ID}.zip"

        logger.info(f"🧠 Mind initialized on device: {self.device}")

        if download_on_init:
            self.awaken()
        if download_on_init and upload_on_fail and not self.model_loaded_successfully:
            logger.warning("Failed to awaken Mind. A new untrained model will be used and uploaded upon training.")
            self.sleep() # Save the fresh, untrained model

    def _prepare_sequences(self, data: np.ndarray):
        """Converts a time series into sequences and corresponding labels."""
        n = len(data)
        if n <= self.sequence_length:
            raise ValueError("Data length must be greater than sequence_length.")
        
        X, y = [], []
        for i in range(n - self.sequence_length):
            X.append(data[i : i + self.sequence_length])
            y.append(data[i + self.sequence_length])
            
        return np.array(X), np.array(y)


    def learn(self, data: np.ndarray, epochs=50, lr=0.001, batch_size=32, flow_epochs=100):
        """Trains the Mind model on the provided data."""
        if not isinstance(data, np.ndarray) or len(data) < self.sequence_length + 1:
            raise ValueError(f"Training data must be a numpy array with at least {self.sequence_length + 1} points.")
        logger.info(f"Mind learning from {len(data)} points for {epochs} epochs...")

        logger.info("Training normalizing flow...")
        self.flow.fit(data, epochs=flow_epochs, batch_size=min(512, len(data)))
        data_transformed = self.flow.transform(data)

        logger.info("Training TCN model on transformed data...")
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

    def predict(self, last_n_ohlc: np.ndarray):
        """Predicts the next High and Low values."""
        logger.info("Making prediction...")
        self.model.eval()
        arr = np.array(last_n_ohlc, dtype=np.float32)
        if arr.shape != (self.sequence_length, 2):
            raise ValueError(f"Input must be a numpy array of shape ({self.sequence_length}, 2)")

        try:
            # Transform input, predict in latent space, then inverse transform the prediction
            transformed_input = self.flow.transform(arr)
            with torch.no_grad():
                latent_pred = self.model(torch.tensor(transformed_input[None], dtype=torch.float32, device=self.device))
                prediction = self.flow.inverse_transform(latent_pred.cpu().numpy())
        except Exception as e:
            logger.error(f"Prediction failed. Ensure the flow model is trained. Error: {e}")
            raise RuntimeError("Flow not trained or prediction failed.") from e
            
        return {"Predicted High": float(prediction[0, 0]), "Predicted Low": float(prediction[0, 1])}

    def sleep(self, max_attempts=3):
        """Saves the current model state to a local file and uploads it to Cloudinary."""
        logger.info("Saving model state to cloud...")
        # Save both model and flow state dictionaries
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'flow_state_dict': self.flow.model.state_dict()
        }, self.MODEL_LOCAL_PATH)

        # Zip the model file for upload
        with zipfile.ZipFile(self.ZIP_LOCAL_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(self.MODEL_LOCAL_PATH, os.path.basename(self.MODEL_LOCAL_PATH))
        
        logger.info(f"Zipped model to {self.ZIP_LOCAL_PATH}")

        for attempt in range(max_attempts):
            try:
                logger.info(f"Upload attempt {attempt+1}/{max_attempts}...")
                cloudinary.uploader.upload(
                    self.ZIP_LOCAL_PATH,
                    resource_type='raw',
                    public_id=self.MODEL_PUBLIC_ID,
                    overwrite=True
                )
                logger.success("✅ Model successfully saved to cloud.")
                # Clean up local files
                if os.path.exists(self.MODEL_LOCAL_PATH): os.remove(self.MODEL_LOCAL_PATH)
                if os.path.exists(self.ZIP_LOCAL_PATH): os.remove(self.ZIP_LOCAL_PATH)
                return True
            except Exception as e:
                logger.warning(f"⚠️ Upload attempt {attempt+1} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)
        logger.error("❌ All model upload attempts failed.")
        return False

    def awaken(self, max_attempts=3):
        """Downloads the model from Cloudinary and loads the state."""
        logger.info("Attempting to load model from cloud...")
        for attempt in range(max_attempts):
            try:
                # Get a signed URL for the private resource
                url = cloudinary.utils.cloudinary_url(
                    f"{self.MODEL_PUBLIC_ID}.zip", resource_type='raw')[0]
                
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()

                # Save the downloaded zip file
                with open(self.ZIP_LOCAL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Unzip and load the model
                with zipfile.ZipFile(self.ZIP_LOCAL_PATH, 'r') as zf:
                    zf.extractall(os.path.dirname(self.MODEL_LOCAL_PATH))
                
                # *** FIX: Set weights_only=False to allow loading of non-tensor data ***
                state = torch.load(self.MODEL_LOCAL_PATH, map_location=self.device, weights_only=False)
                
                self.model.load_state_dict(state['model_state_dict'])
                self.flow.model.load_state_dict(state['flow_state_dict'])
                self.model.eval()
                self.flow.model.eval()
                
                self.model_loaded_successfully = True
                logger.success("✅ Model loaded successfully from cloud.")
                # Clean up local files
                if os.path.exists(self.MODEL_LOCAL_PATH): os.remove(self.MODEL_LOCAL_PATH)
                if os.path.exists(self.ZIP_LOCAL_PATH): os.remove(self.ZIP_LOCAL_PATH)
                return True
            except Exception as e:
                logger.warning(f"⚠️ Download attempt {attempt+1} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)
        
        logger.error("❌ All model download attempts failed.")
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
    # A mix of sine wave, linear trend, and noise to create a complex series
    series = np.sin(time_arr / 20) * 75 + time_arr * 0.1 + np.random.randn(num_points) * 15
    lows = series - np.random.rand(num_points) * 5 - 2
    highs = series + np.random.rand(num_points) * 5 + 2
    return np.stack([highs, lows], axis=1).astype(np.float32)

if __name__ == '__main__':
    # Initialize Mind, attempting to download a pre-trained model.
    # If download fails, it will proceed with an untrained model.
    mind = Mind(sequence_length=20, download_on_init=True)

    if not mind.model_loaded_successfully:
        logger.info("Training a new model as no pre-trained model was loaded.")
        # Generate synthetic data for training
        training_data = generate_synthetic_data(1000)
        
        # Train the model
        train_results = mind.learn(
            training_data,
            epochs=50,       # Fewer epochs for a quick example
            flow_epochs=100, # Fewer epochs for a quick example
            lr=0.001,
            batch_size=64
        )
        
        # Save the newly trained model to the cloud
        mind.sleep()
        logger.info(f"Training finished. Final TCN loss: {train_results['final_loss']:.6f}")
    else:
        logger.info("✅ Pre-trained model loaded and ready for prediction.")

    # Prepare some test data (the last 20 points from a new synthetic set)
    test_data_full = generate_synthetic_data(100)
    test_sequence = test_data_full[-20:] # Input must have sequence_length

    # Make a prediction
    prediction = mind.predict(test_sequence)

    print("\n" + "="*40)
    print("🔮 MIND PREDICTION 🔮")
    print(f"Last Known High: {test_sequence[-1, 0]:.2f}")
    print(f"Last Known Low:  {test_sequence[-1, 1]:.2f}")
    print("----------------------------------------")
    print(f"Predicted Next High: {prediction['Predicted High']:.2f}")
    print(f"Predicted Next Low:  {prediction['Predicted Low']:.2f}")
    print("="*40 + "\n")
