# app/main.py
from fastapi import FastAPI
from app.routes import predict
from app.models import DataRequest

app = FastAPI(title="Time Series Prediction API")

app.include_router(predict.router)

@app.get("/")
def read_root():
    return {"message": "Time Series Prediction AI API is running."}

@app.post("/predict")  # 🚨 반드시 @app.post() 여야 함
def predict_endpoint(data_request: DataRequest):
    return {"message": "Prediction received", "data": data_request}

