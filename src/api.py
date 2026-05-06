from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load the best model
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Fraud Detection API")

class Transaction(BaseModel):
    features: list[float]  # 30 features from the dataset

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: Transaction):
    features = np.array(transaction.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    return {
        "prediction": "fraud" if prediction == 1 else "legitimate",
        "fraud_probability": round(float(probability), 4)
    }