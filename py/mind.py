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

# --- Cloudinary dependencies ---
import cloudinary
import cloudinary.uploader
import cloudinary.api

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Temporal Convolutional Network (TCN) Components ---
# This is a powerful architecture for sequence data, often outperforming LSTMs.

class Chomp1d(nn.Module):
    """Removes the padding from the end of a sequence."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock(nn.Module):
    """A single block of a Temporal Convolutional Network."""
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
    """The full Temporal Convolutional Network."""
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

# --- The Mind Model ---
class MindModel(nn.Module):
    """
    The core neural network of the Mind. It uses a TCN to process the sequence
    and a linear layer to predict the next High and Low.
    """
    def __init__(self, input_size=2, output_size=2, num_channels=[32, 64, 128], kernel_size=7, dropout=0.2):
        super(MindModel, self).__init__()
        self.tcn = TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        # The final layer predicts two values: High and Low
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # Input x shape: (batch_size, seq_length, num_features)
        # TCN expects: (batch_size, num_features, seq_length)
        y1 = self.tcn(x.transpose(1, 2))
        # We take the output from the last time step
        return self.linear(y1[:, :, -1])

# --- The Mind Controller ---
class Mind:
    """
    The main controller class. It manages the model, data, training,
    predictions, and communication with Cloudinary.
    """
    def __init__(self, sequence_length=20, device=None, download_on_init=False, upload_on_fail=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = sequence_length
        self.model = MindModel(input_size=2, output_size=2)
        self.model.to(self.device)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.model_loaded_successfully = False

        # --- Cloudinary Configuration (from Libra.py) ---
        self.CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "dj4bwntzb")
        self.API_KEY = os.getenv("CLOUDINARY_API_KEY", "354656419316393")
        self.API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "M-Trl9ltKDHyo1dIP2AaLOG-WPM")
        self.MODEL_PUBLIC_ID = "mind_hl_model_v1.zip" # A unique name for this model's file
        self.MODEL_LOCAL_PATH = "/tmp/mind.pt"
        self.ZIP_LOCAL_PATH = "/tmp/mind.zip"

        logging.info(f"🧠 Mind initialized on device: {self.device}")

        # Configure Cloudinary API
        cloudinary.config(cloud_name=self.CLOUD_NAME, api_key=self.API_KEY, api_secret=self.API_SECRET, secure=True)

        if download_on_init:
            self.awaken() # Awaken the Mind from the cloud

        if download_on_init and upload_on_fail and not self.model_loaded_successfully:
            logging.warning("Failed to awaken Mind from Cloudinary. Uploading new, untrained Mind as a baseline.")
            self.sleep() # Put the new Mind to sleep in the cloud

    def _prepare_sequences(self, data):
        """Creates sequences and labels from the input data."""
        sequences, labels = [], []
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i+self.sequence_length]
            label = data[i+self.sequence_length]
            sequences.append(seq)
            labels.append(label)
        return np.array(sequences), np.array(labels)

    def learn(self, data, epochs=50, lr=0.001, batch_size=32):
        """Trains the Mind on new data."""
        logging.info(f"Mind is learning from {len(data)} data points for {epochs} epochs...")
        self.model.train()

        # Fit the scaler ONLY on the training data to avoid data leakage
        data_scaled = self.scaler.fit_transform(data)
        X, y = self._prepare_sequences(data_scaled)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss() # Mean Squared Error for regression

        for epoch in range(epochs):
            epoch_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                y_pred = self.model(xb)
                loss = criterion(y_pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logging.info(f"  Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(loader):.6f}")
        logging.info("✅ Learning complete.")

    def predict(self, last_20_ohlc):
        """Predicts the next High and Low given the last 20 periods."""
        logging.info("Mind is predicting...")
        self.model.eval()
        if len(last_20_ohlc) != self.sequence_length:
            raise ValueError(f"Input data must have a length of {self.sequence_length}")

        # Use the already fitted scaler to transform the new data
        data_scaled = self.scaler.transform(last_20_ohlc)
        sequence = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction_scaled = self.model(sequence)

        # Inverse transform to get the actual predicted values
        prediction = self.scaler.inverse_transform(prediction_scaled.cpu().numpy())
        return {"Predicted High": prediction[0, 0], "Predicted Low": prediction[0, 1]}

    def sleep(self, max_attempts=3):
        """Saves the Mind's state (memory) to Cloudinary."""
        logging.info("Mind is going to sleep. Saving memory to the cloud...")
        # Save the model and scaler state
        state = {
            'model_state_dict': self.model.state_dict(),
            'scaler_min': self.scaler.min_,
            'scaler_scale': self.scaler.scale_
        }
        torch.save(state, self.MODEL_LOCAL_PATH)

        # Zip the file for more efficient storage
        with zipfile.ZipFile(self.ZIP_LOCAL_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(self.MODEL_LOCAL_PATH, os.path.basename(self.MODEL_LOCAL_PATH))

        # Upload to Cloudinary
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
                if attempt < max_attempts - 1: time.sleep(2 ** (attempt + 1))
        logging.error("❌ All attempts to upload Mind's memory failed.")
        return False

    def awaken(self, max_attempts=3):
        """Loads the Mind's state (memory) from Cloudinary."""
        logging.info("Mind is awakening. Loading memory from the cloud...")
        for attempt in range(max_attempts):
            try:
                # Get a signed URL to download the private raw file
                url, _ = cloudinary.utils.cloudinary_url(self.MODEL_PUBLIC_ID, resource_type='raw', sign_url=True)
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()

                with open(self.ZIP_LOCAL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                with zipfile.ZipFile(self.ZIP_LOCAL_PATH, 'r') as zf:
                    zf.extractall(os.path.dirname(self.MODEL_LOCAL_PATH))

                # Load the state
                state = torch.load(self.MODEL_LOCAL_PATH, map_location=self.device, weights_only=False)
                self.model.load_state_dict(state['model_state_dict'])
                self.scaler.min_ = state['scaler_min']
                self.scaler.scale_ = state['scaler_scale']
                self.model.eval()
                self.model_loaded_successfully = True
                logging.info("✅ Mind has awakened successfully.")
                return True
            except Exception as e:
                logging.warning(f"Cloudinary download attempt {attempt + 1}/{max_attempts} failed: {e}")
                if attempt < max_attempts - 1: time.sleep(2 ** (attempt + 1))
        logging.error("❌ All attempts to awaken the Mind failed.")
        self.model_loaded_successfully = False
        return False

# --- Example Usage ---
if __name__ == '__main__':
    # --- 1. Generate Synthetic Data ---
    def generate_synthetic_data(num_points=500):
        time = np.arange(0, num_points, 1)
        series = np.sin(time / 20) * 75 + time * 0.1 + np.random.randn(num_points) * 15
        lows = series - np.random.rand(num_points) * 5
        highs = series + np.random.rand(num_points) * 5
        return np.stack([highs, lows], axis=1)

    # --- 2. Initialize the Mind ---
    # Set download_on_init=True to load a pre-trained model from your Cloudinary
    # Set upload_on_fail=True to save a new model if download fails
    mind = Mind(sequence_length=20, download_on_init=True, upload_on_fail=True)

    # --- 3. Train the Mind (if it didn't load a pre-trained one) ---
    if not mind.model_loaded_successfully:
        logging.info("No pre-trained Mind found. Training a new one.")
        historical_data = generate_synthetic_data(1000)
        mind.learn(historical_data, epochs=50)
        # After training, put the new Mind to sleep so it can be awakened later
        mind.sleep()
    else:
        logging.info("Pre-trained Mind awakened. Ready for predictions.")

    # --- 4. Make a Prediction ---
    # Create some new data for prediction
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
