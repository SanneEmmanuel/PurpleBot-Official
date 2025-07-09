import os, requests, zipfile, torch, logging
import torch.nn as nn
from io import BytesIO
from torch.utils.data import DataLoader, TensorDataset

# ========== 🔐 Cloudinary Config ==========
CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "dj4bwntzb")
API_KEY = os.getenv("CLOUDINARY_API_KEY", "354656419316393")
API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "M-Trl9ltKDHyo1dIP2AaLOG-WPM")
MODEL_URL = f"https://res.cloudinary.com/{CLOUD_NAME}/raw/upload/v1/model.pt.zip"
MODEL_PATH = "/tmp/model.pt"
ZIP_PATH = "/tmp/model.pt.zip"

# ========== ⚙️ Device Setup ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
logging.info(f"🔌 Using device: {DEVICE}")

# ========== 🧠 Model Definition ==========
class LibraModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, lstm_layers=2, dropout=0.2, attn_heads=2):
        super().__init__()
        logging.info("🧠 Initializing LibraModel...")

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=True
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attn_heads,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 5)

        logging.info(f"✅ LSTM: input_size={input_size}, hidden_size={hidden_size}, layers={lstm_layers}")
        logging.info(f"✅ MultiHeadAttention: embed_dim={hidden_size}, heads={attn_heads}")
        logging.info("✅ Dropout and FC initialized.")

    def forward(self, x):
        logging.debug(f"📥 Input shape: {x.shape}")
        lstm_out, _ = self.lstm(x)                          # Shape: (B, T, H)
        logging.debug(f"🔁 LSTM output shape: {lstm_out.shape}")

        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        logging.debug(f"🔍 Attention output shape: {attn_out.shape}")

        out = self.dropout(attn_out[:, -1, :])              # Take last time step, apply dropout
        output = self.fc(out)
        logging.debug(f"🎯 Final output shape: {output.shape}")

        return output

# ========== ⬇️ Model Download ==========
def download_model_from_cloudinary():
    logging.info("📦 Downloading model from Cloudinary...")
    r = requests.get(MODEL_URL, auth=(API_KEY, API_SECRET))
    if r.status_code != 200:
        raise RuntimeError(f"❌ Failed to download model: {r.status_code}")
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        z.extractall("/tmp")
    logging.info("✅ Model extracted to /tmp.")

# ========== 🚀 Load Model ==========
def load_model():
    if not os.path.exists(MODEL_PATH):
        download_model_from_cloudinary()

    model = LibraModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    logging.info("✅ Model loaded.")
    return model

# ========== 🔮 Predict Function ==========
def predict_ticks(model, ticks):
    x = torch.tensor(ticks, dtype=torch.float32).view(1, 300, 1).to(DEVICE)
    logging.debug(f"📡 Input to model: {x.shape}")
    with torch.no_grad():
        output = model(x)
    result = output.squeeze().cpu().tolist()
    logging.info(f"📈 Predicted next 5 ticks: {result}")
    return result

# ========== 🔁 Retrain + Upload ==========
def retrain_and_upload(model, x_data, y_data, epochs=10):
    logging.info("🔄 Retraining model...")
    model.train()
    x_tensor = torch.tensor(x_data, dtype=torch.float32).view(-1, 300, 1)
    y_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 5)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model.to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logging.info(f"📚 Epoch {epoch+1}: Loss = {total_loss:.4f}")

    model.eval()
    torch.save(model.state_dict(), MODEL_PATH)
    logging.info("💾 Model saved locally.")

    # Zip model.pt
    with zipfile.ZipFile(ZIP_PATH, 'w') as zipf:
        zipf.write(MODEL_PATH, arcname="model.pt")
    logging.info("📦 Model zipped.")

    # Upload to Cloudinary
    with open(ZIP_PATH, "rb") as f:
        response = requests.post(
            f"https://api.cloudinary.com/v1_1/{CLOUD_NAME}/raw/upload",
            auth=(API_KEY, API_SECRET),
            files={"file": f},
            data={"public_id": "model.pt", "overwrite": True}
        )
    if response.status_code == 200:
        logging.info("🚀 Uploaded new model to Cloudinary.")
    else:
        logging.error(f"❌ Upload failed: {response.text}")
