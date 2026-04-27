from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import sklearn

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(title="Week 7 Practice API")


class PredictRequest(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float


@app.get("/")
def root():
    return {"status": "ML API running", "week": 7, "day": 2}

@app.get("/health")
def health():
    if model is None or scaler is None:
        return {"status": "not ready"}
    return {"status": "ok"}


@app.get("/model/info")
def model_info():
    return {
        "model_version": "v1.0",
        "model_type": type(model).__name__,
        "features": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
        "training_accuracy": 0.91,
        "dataset": "stock_data"
    }

@app.post("/predict")
def predict(req: PredictRequest):
    features = np.reshape([req.feature_1,req.feature_2,req.feature_3,req.feature_4,req.feature_5],(1,5))
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)
    probability = model.predict_proba(scaled)
    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0][1]),
        "version": "v1.0"
    }