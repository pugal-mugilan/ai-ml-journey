import joblib
import numpy as np
import requests

# Load the same model and scaler directly
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# 5 test samples — make up some numbers, or grab from your Week 6 data
test_samples = [
    [20.1, 300, 50, 0.9, 22],
    [15.5, 150, 30, 0.5, 10],
    [50.0, 500, 80, 1.2, 45],
    [5.0, 50, 10, 0.1, 3],
    [35.0, 400, 65, 0.8, 30],
]

for i, sample in enumerate(test_samples):
    # Direct prediction
    scaled = scaler.transform(np.array(sample).reshape(1, 5))
    direct_pred = int(model.predict(scaled)[0])
    direct_prob = float(model.predict_proba(scaled)[0][1])

    # API prediction
    payload = {f"feature_{j+1}": sample[j] for j in range(5)}
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    api_result = response.json()

    # Compare
    match = direct_pred == api_result["prediction"] and abs(direct_prob - api_result["probability"]) < 1e-10
    print(f"Sample {i+1}: direct={direct_pred} ({direct_prob:.6f}) | API={api_result['prediction']} ({api_result['probability']:.6f}) | {'MATCH' if match else 'MISMATCH'}")