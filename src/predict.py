"""
predict.py
FastAPI app that serves the trained model.
POST /predict  →  { "features": [cylinders, displacement, horsepower,
                                  weight, acceleration, model_year,
                                  origin, car_name_encoded] }
GET  /health   →  { "status": "ok" }
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

MODELS_DIR = "models"

app = FastAPI(title="Auto MPG Predictor", version="1.0.0")

# Load artifacts once at startup
model  = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))


class PredictRequest(BaseModel):
    # Order: cylinders, displacement, horsepower, weight,
    #        acceleration, model_year, origin, car_name (encoded int)
    features: List[float]


class PredictResponse(BaseModel):
    predicted_mpg: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.features) != 8:
        raise HTTPException(
            status_code=422,
            detail="Exactly 8 features required: cylinders, displacement, "
                   "horsepower, weight, acceleration, model_year, origin, car_name_encoded"
        )
    X = np.array(req.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    mpg = model.predict(X_scaled)[0]
    return {"predicted_mpg": round(float(mpg), 2)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
