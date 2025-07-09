# Libra.py
import os, requests, zipfile, torch
import torch.nn as nn
from io import BytesIO
from torch.utils.data import DataLoader, TensorDataset

# 🔐 Cloudinary config
CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "dj4bwntzb")
API_KEY = os.getenv("CLOUDINARY_API_KEY", "354656419316393")
API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "M-Trl9ltKDHyo1dIP2AaLOG-WPM")
MODEL_URL = f"https://res.cloudinary.com/{CLOUD_NAME}/raw/upload/v1/model.pt.zip"
MODEL_PATH = "/tmp/model.pt"
ZIP_PATH = "/tmp/model.pt.zip"

# 📦 Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔌 Using device: {DEVICE}")

# 🧠 Model definition
class LibraModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 5)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 🔽 Download model.pt.zip from Cloudinary and extract
def download_model_from_cloudinary():
    print("📦 Downloading model from Cloudinary...")
    r = requests.get(MODEL_URL, auth=(API_KEY, API_SECRET))
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download model: {r.status_code}")
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        z.extractall("/tmp")
    print("✅ Model extracted to /tmp.")

# 🚀 Load model with GPU support
def load_model():
    if not os.path.exists(MODEL_PATH):
        download_model_from_cloudinary()

    model = LibraModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ Model loaded.")
    return model

# 🔮 Predict next 5 ticks from 300
def predict_ticks(model, ticks):
    x = torch.tensor(ticks, dtype=torch.float32).view(1, 300, 1).to(DEVICE)
    with torch.no_grad():
        return model(x).squeeze().cpu().tolist()

# 🔁 Retrain model on new batch of (X, Y) and upload
def retrain_and_upload(model, x_data, y_data, epochs=10):
    print("🔄 Retraining...")
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
        print(f"📚 Epoch {epoch+1}: Loss = {total_loss:.4f}")

    model.eval()
    torch.save(model.state_dict(), MODEL_PATH)
    print("💾 Model saved locally.")

    # Zip model.pt
    with zipfile.ZipFile(ZIP_PATH, 'w') as zipf:
        zipf.write(MODEL_PATH, arcname="model.pt")
    print("📦 Model zipped.")

    # Upload to Cloudinary
    with open(ZIP_PATH, "rb") as f:
        response = requests.post(
            f"https://api.cloudinary.com/v1_1/{CLOUD_NAME}/raw/upload",
            auth=(API_KEY, API_SECRET),
            files={"file": f},
            data={"public_id": "model.pt", "overwrite": True}
        )
    if response.status_code == 200:
        print("🚀 Uploaded new model to Cloudinary.")
    else:
        print(f"❌ Upload failed: {response.text}")
