from fastapi import FastAPI
from pydantic import BaseModel
import torch
import joblib
import numpy as np

app = FastAPI()

# -------- INPUT SCHEMA (Fix for FastAPI body parsing) --------
class InputData(BaseModel):
    features: list[float]

# -------- LOAD MODEL --------
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(16, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

model = MLP()
model.load_state_dict(
    torch.load("mlp_ton_iot.pt", map_location=torch.device("cpu"))
)
model.eval()

# -------- LOAD SCALER --------
scaler = joblib.load("scaler.pkl")

# -------- ROUTES --------
@app.get("/")
def home():
    return {"message": "IoT IDS Cloud Running"}

@app.post("/predict")
def predict(data: InputData):

    # Convert list to numpy array
    features = np.array(data.features).reshape(1, -1)

    # Validate feature length
    if features.shape[1] != 16:
        return {"error": "Expected 16 input features."}

    # Scale input
    x_scaled = scaler.transform(features)

    # Convert to tensor
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    # Model inference
    with torch.no_grad():
        output = model(x_tensor)

    pred = torch.argmax(output, dim=1).item()

    return {"prediction": int(pred)}
