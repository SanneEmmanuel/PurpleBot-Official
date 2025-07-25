import time
import os
import requests
import zipfile
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils import weight_norm
from loguru import logger
from copy import deepcopy

import cloudinary
import cloudinary.uploader
import cloudinary.api

# --- PyTorch Compatibility Fix ---
# This addresses the "Unsupported global: numpy._core.multiarray._reconstruct" error
# by explicitly trusting the numpy global during deserialization.
from torch.serialization import add_safe_globals
import numpy.core.multiarray

add_safe_globals([numpy.core.multiarray._reconstruct, numpy.core.multiarray.scalar])


# Initialize Cloudinary
cloudinary.config(
    cloud_name="dj4bwntzb",
    api_key="354656419316393",
    api_secret="M-Trl9ltKDHyo1dIP2AaLOG-WPM",
    secure=True
)

class Chomp1d(nn.Module):
    """A 1D chomping module to remove the rightmost padding from a tensor."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """Applies the chomping operation."""
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock(nn.Module):
    """A single block of a Temporal Convolutional Network (TCN)."""
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
        """Initializes the weights of the convolutional layers."""
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)

    def forward(self, x):
        """Forward pass through the TCN block."""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """Temporal Convolutional Network (TCN) model."""
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
        """Forward pass through the TCN."""
        return self.network(x)

class MindModel(nn.Module):
    """The main Mind model, combining a TCN with a linear layer."""
    def __init__(self, input_size=2, output_size=2, num_channels=[32, 64, 128], kernel_size=7, dropout=0.2):
        super().__init__()
        self.tcn = TCN(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        nn.init.normal_(self.linear.weight, 0, 0.01)

    def forward(self, x):
        """Forward pass through the Mind model."""
        tcn_out = self.tcn(x.transpose(1, 2))
        return self.linear(tcn_out[:, :, -1])

class Mind:
    """A class to manage the Mind model, including training, prediction, and cloud storage."""
    def __init__(self, sequence_length=20, device=None, download_on_init=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = sequence_length
        self.model = MindModel().to(self.device)
        self.model_loaded_successfully = False
        self.norm_params = {}

        self.MODEL_PUBLIC_ID = "mind_hl_model_v2"
        self.FALLBACK_URL = "https://res.cloudinary.com/dj4bwntzb/raw/upload/mind_hl_model_v2.zip"
        self.MODEL_WEIGHTS_LOCAL_PATH = "/tmp/mind_v2.pt"
        self.NORM_PARAMS_LOCAL_PATH = "/tmp/norm_param.pt"
        self.ZIP_LOCAL_PATH = "/tmp/mind_v2.zip"

        logger.info(f"🧠 Mind initialized on device: {self.device}")

        if download_on_init:
            self.awaken()
        if not self.model_loaded_successfully:
             logger.warning("Could not load a pre-trained model. Please train a new one.")


    def _normalize(self, data, is_training=False):
        """Custom normalization function. If training, it calculates and stores new params."""
        if is_training or not self.norm_params:
            self.norm_params['min'] = np.min(data, axis=0)
            self.norm_params['max'] = np.max(data, axis=0)
        
        min_val = self.norm_params.get('min', 0)
        max_val = self.norm_params.get('max', 1)
        
        return (data - min_val) / (max_val - min_val + 1e-8)

    def _denormalize(self, data):
        """Custom denormalization function."""
        if not self.norm_params:
            raise RuntimeError("Normalization parameters not found. Train or load a model first.")
        
        min_val = self.norm_params['min']
        max_val = self.norm_params['max']
        
        return data * (max_val - min_val + 1e-8) + min_val

    def _prepare_sequences(self, data):
        """Prepares sequences for training."""
        n_samples = len(data) - self.sequence_length
        if n_samples <= 0:
            raise ValueError("Data is not long enough to create at least one sequence.")
            
        sequences = np.array([data[i:i+self.sequence_length] for i in range(n_samples)])
        labels = data[self.sequence_length:]
        
        return sequences, labels

    def learn(self, data, epochs=50, lr=0.001, batch_size=32, validation_split=0.15):
        """
        Trains the Mind model, retaining only the best model based on validation loss.
        """
        if not isinstance(data, np.ndarray) or len(data) < self.sequence_length + 1:
            raise ValueError("Training data must be a numpy array with sufficient length.")

        logger.info(f"Mind learning from {len(data)} data points for {epochs} epochs...")
        
        # Normalize data and store parameters for denormalization later
        data_scaled = self._normalize(data, is_training=True)
        X, y = self._prepare_sequences(data_scaled)

        # Split data into training and validation sets
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("Not enough data to create a validation set. Reduce validation_split or provide more data.")

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32, device=self.device),
            torch.tensor(y_train, dtype=torch.float32, device=self.device)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32, device=self.device),
            torch.tensor(y_val, dtype=torch.float32, device=self.device)
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # --- Training Phase ---
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)

            # --- Validation Phase ---
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = self.model(xb)
                    loss = criterion(pred, yb)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            # --- Best Model Check ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = deepcopy(self.model.state_dict())
                logger.info(f"✨ New best model found at epoch {epoch+1} with Val Loss: {best_val_loss:.6f}")

        # Restore the best model found during training
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            logger.success(f"Learning complete. Best model with validation loss {best_val_loss:.6f} is loaded.")
        else:
            logger.warning("Training finished, but a best model was not saved. The model state from the last epoch will be used.")
            
        return {"best_validation_loss": best_val_loss}

    def predict(self, last_20_ohlc):
        """Makes a prediction based on the last 20 data points."""
        logger.info("Making prediction...")
        self.model.eval()
        
        if not self.norm_params:
            raise RuntimeError("Model is not ready. Train or load a model with normalization params first.")
        
        if not isinstance(last_20_ohlc, (np.ndarray, list)):
            raise ValueError("Input must be a numpy array or a list.")
            
        arr = np.array(last_20_ohlc, dtype=np.float32)
        if arr.shape != (self.sequence_length, 2):
            raise ValueError(f"Input must have the shape ({self.sequence_length}, 2).")
            
        try:
            # Use existing params for normalization, do not recalculate
            scaled = self._normalize(arr, is_training=False)
            with torch.no_grad():
                tensor_input = torch.tensor(scaled[None], dtype=torch.float32, device=self.device)
                pred = self.model(tensor_input)
                prediction = self._denormalize(pred.cpu().numpy())[0]
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError("Ensure the model is trained or loaded before prediction.") from e
            
        return {"Predicted High": prediction[0], "Predicted Low": prediction[1]}

    def sleep(self, max_attempts=3):
        """Saves the model weights and normalization parameters to separate files, zips them, and uploads to the cloud."""
        if not self.norm_params:
            logger.error("Cannot save model: Normalization parameters are missing. Please train the model first.")
            return False

        logger.info("Saving model and normalization parameters...")
        torch.save(self.model.state_dict(), self.MODEL_WEIGHTS_LOCAL_PATH)
        torch.save(self.norm_params, self.NORM_PARAMS_LOCAL_PATH)

        logger.info(f"Zipping artifacts to {self.ZIP_LOCAL_PATH}...")
        try:
            with zipfile.ZipFile(self.ZIP_LOCAL_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(self.MODEL_WEIGHTS_LOCAL_PATH, os.path.basename(self.MODEL_WEIGHTS_LOCAL_PATH))
                zf.write(self.NORM_PARAMS_LOCAL_PATH, os.path.basename(self.NORM_PARAMS_LOCAL_PATH))
        except Exception as e:
            logger.error(f"Failed to create zip file: {e}")
            return False

        logger.info("Uploading model to cloud...")
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

    def _download_and_extract(self, url, source_description):
        """Helper function to download from a URL and extract model files."""
        try:
            logger.info(f"Attempting to download from {source_description}...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(self.ZIP_LOCAL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info("Extracting model files...")
            with zipfile.ZipFile(self.ZIP_LOCAL_PATH, 'r') as zf:
                zf.extractall(os.path.dirname(self.MODEL_WEIGHTS_LOCAL_PATH))

            logger.info(f"Loading weights from {self.MODEL_WEIGHTS_LOCAL_PATH}")
            self.model.load_state_dict(torch.load(self.MODEL_WEIGHTS_LOCAL_PATH, map_location=self.device))
            
            logger.info(f"Loading normalization params from {self.NORM_PARAMS_LOCAL_PATH}")
            self.norm_params = torch.load(self.NORM_PARAMS_LOCAL_PATH, weights_only=False)
            
            self.model.eval()
            self.model_loaded_successfully = True
            logger.success(f"Model loaded successfully from {source_description}.")
            return True
        except Exception as e:
            logger.warning(f"Download from {source_description} failed: {e}")
            return False

    def awaken(self, max_attempts=3):
        """Loads the model from the cloud, with a fallback to a public URL."""
        logger.info("Waking Mind... Attempting to load model from the cloud.")
        
        # Primary method: try to get a signed URL and download
        signed_url_success = False
        for attempt in range(max_attempts):
            try:
                url = cloudinary.utils.cloudinary_url(
                    f"{self.MODEL_PUBLIC_ID}.zip", resource_type='raw', sign_url=True)[0]
                if self._download_and_extract(url, "signed Cloudinary URL"):
                    signed_url_success = True
                    break
            except Exception as e:
                logger.warning(f"Could not generate signed URL on attempt {attempt+1}/{max_attempts}: {e}")
            
            time.sleep(2 ** attempt)

        if signed_url_success:
            return True

        # Fallback method: if the signed URL fails after all attempts, try the public URL
        logger.warning("Primary download method failed. Trying fallback public URL...")
        if self._download_and_extract(self.FALLBACK_URL, "fallback public URL"):
            return True
        
        logger.error("All download attempts failed, including the fallback URL.")
        self.model_loaded_successfully = False
        return False


if __name__ == '__main__':
    def generate_synthetic_data(num_points=500):
        """Generates synthetic data for training and testing."""
        time_arr = np.arange(0, num_points, 1)
        series = (np.sin(time_arr / 20) * 75 + 
                  np.sin(time_arr / 10) * 25 + 
                  time_arr * 0.1 + 
                  np.random.randn(num_points) * 15)
        lows = series - np.abs(np.random.randn(num_points) * 5 + 2)
        highs = series + np.abs(np.random.randn(num_points) * 5 + 2)
        return np.stack([highs, lows], axis=1).astype(np.float32)

    # Initialize Mind. It will try to download a pre-trained model.
    mind = Mind(sequence_length=20, download_on_init=True)

    # If no model was loaded, train a new one.
    if not mind.model_loaded_successfully:
        logger.info("Training a new model as a pre-trained one was not found or failed to load...")
        training_data = generate_synthetic_data(2000)
        train_results = mind.learn(training_data, epochs=100, lr=0.001)
        
        # Save the newly trained model to the cloud.
        mind.sleep()
        logger.info(f"Training finished. Best validation loss: {train_results['best_validation_loss']:.6f}")
    else:
        logger.info("A pre-trained model has been successfully loaded.")

    # Generate test data and make a prediction.
    try:
        # Use the last 20 points from new synthetic data for prediction
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
    except RuntimeError as e:
        logger.error(f"Could not make a prediction. Error: {e}")
