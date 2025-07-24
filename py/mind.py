import time
import os
import requests
import zipfile
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils import weight_norm
from sklearn.preprocessing import MinMaxScaler
from loguru import logger

import cloudinary
import cloudinary.uploader
import cloudinary.api

# Initialize Cloudinary
cloudinary.config(
    cloud_name="dj4bwntzb",
    api_key="354656419316393",
    api_secret="M-Trl9ltKDHyo1dIP2AaLOG-WPM",
    secure=True
)
import numpy.core.multiarray
from torch.serialization import add_safe_globals
add_safe_globals([np.core.multiarray._reconstruct])

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
        return self.linear(self.tcn(x.transpose(1, 2))[:, :, -1])

class Mind:
    def __init__(self, sequence_length=20, device=None, download_on_init=False, upload_on_fail=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = sequence_length
        self.model = MindModel().to(self.device)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
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

    def _prepare_sequences(self, data):
        n = len(data)
        seq_size = n - self.sequence_length
        indices = np.arange(seq_size)[:, None] + np.arange(self.sequence_length)
        sequences = data[indices].reshape(seq_size, self.sequence_length, -1)
        labels = data[self.sequence_length:]
        return sequences, labels

    def learn(self, data, epochs=50, lr=0.001, batch_size=32):
        if not isinstance(data, np.ndarray) or len(data) < self.sequence_length + 1:
            raise ValueError("Training data must be a numpy array with sufficient length.")

        logger.info(f"Mind learning from {len(data)} points for {epochs} epochs...")
        self.model.train()
        
        data_scaled = self.scaler.fit_transform(data)
        X, y = self._prepare_sequences(data_scaled)
        
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32, device=self.device),
            torch.tensor(y, dtype=torch.float32, device=self.device)
        )
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        epoch_losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(loader)
            epoch_losses.append(avg_loss)
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
                
        logger.success("Learning complete.")
        return {"epoch_losses": epoch_losses, "final_loss": epoch_losses[-1]}

    def predict(self, last_20_ohlc):
        logger.info("Making prediction...")
        self.model.eval()
        
        if not isinstance(last_20_ohlc, (np.ndarray, list)):
            raise ValueError("Input must be numpy array or list")
            
        arr = np.array(last_20_ohlc, dtype=np.float32)
        if arr.shape != (self.sequence_length, 2):
            raise ValueError(f"Input must be shape ({self.sequence_length}, 2)")
            
        try:
            scaled = self.scaler.transform(arr)
            with torch.no_grad():
                pred = self.model(torch.tensor(scaled[None], dtype=torch.float32, device=self.device))
                prediction = self.scaler.inverse_transform(pred.cpu().numpy())[0]
        except Exception as e:
            raise RuntimeError("Scaler not fitted. Train or load model first.") from e
            
        return {"Predicted High": prediction[0], "Predicted Low": prediction[1]}

    def sleep(self, max_attempts=3):
        logger.info("Saving model to cloud...")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_params': {
                'data_min': self.scaler.data_min_,
                'data_max': self.scaler.data_max_,
                'data_range': self.scaler.data_range_,
                'feature_range': self.scaler.feature_range,
                'scale_': self.scaler.scale_,
                'min_': self.scaler.min_
            }

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
                url = cloudinary.utils.cloudinary_url(
                    self.MODEL_PUBLIC_ID, resource_type='raw', sign_url=True)[0]
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                with open(self.ZIP_LOCAL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                with zipfile.ZipFile(self.ZIP_LOCAL_PATH, 'r') as zf:
                    zf.extractall(os.path.dirname(self.MODEL_LOCAL_PATH))
                    
                state = torch.load(self.MODEL_LOCAL_PATH, map_location=self.device)
                self.model.load_state_dict(state['model_state_dict'])
                
                # Reconstruct scaler
                params = state['scaler_params']
                self.scaler = MinMaxScaler(feature_range=params['feature_range'])
                self.scaler.data_min_ = params['data_min']
                self.scaler.data_max_ = params['data_max']
                self.scaler.data_range_ = params['data_range']
                self.scaler.scale_ = params['scale_']
                self.scaler.min_ = params['min_']


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

# Example usage remains the same as original
if __name__ == '__main__':
    def generate_synthetic_data(num_points=500):
        time_arr = np.arange(0, num_points, 1)
        series = np.sin(time_arr / 20) * 75 + time_arr * 0.1 + np.random.randn(num_points) * 15
        lows = series - np.random.rand(num_points) * 5
        highs = series + np.random.rand(num_points) * 5
        return np.stack([highs, lows], axis=1)

    mind = Mind(sequence_length=20, download_on_init=True, upload_on_fail=True)

    if not mind.model_loaded_successfully:
        logger.info("Training new model...")
        train_results = mind.learn(generate_synthetic_data(1000), epochs=50)
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
