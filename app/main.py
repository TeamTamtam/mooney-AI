# app/main.py
from fastapi import FastAPI
from app.routes import predict

app = FastAPI(title="Time Series Prediction API")

app.include_router(predict.router)

@app.get("/")
def read_root():
    return {"message": "Time Series Prediction AI API is running."}

