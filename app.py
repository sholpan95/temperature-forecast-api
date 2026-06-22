from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np

app = FastAPI()

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
       self.lstm = nn.LSTM(input_size=1, hidden_size=100, batch_first=True)
self.fc = nn.Linear(100,1)

    def forward(self,x):
        out,_ = self.lstm(x)
        out = self.fc(out[:,-1,:])
        return out

model = LSTM()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

class InputData(BaseModel):
    temperatures: list[float]

@app.get("/")
def home():
    return {"message":"Temperature Forecasting API"}

@app.post("/predict")
def predict(data: InputData):
    values = np.array(data.temperatures).reshape(1,30,1)
    tensor = torch.FloatTensor(values)

    with torch.no_grad():
        prediction = model(tensor)

    return {"predicted_temperature": float(prediction.item())}
