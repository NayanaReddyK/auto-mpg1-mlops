"""
evaluate.py
Loads the saved model and test split, prints a full evaluation report,
and writes metrics/metrics.json (can be run standalone to re-evaluate).
"""

import os
import json
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

PROCESSED  = os.path.join("data", "processed")
MODELS_DIR = "models"
METRICS    = "metrics"


def main():
    model  = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
    X_test = np.load(os.path.join(PROCESSED, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED, "y_test.npy"))

    y_pred = model.predict(X_test)

    metrics = {
        "r2":  round(float(r2_score(y_test, y_pred)), 4),
        "mse": round(float(mean_squared_error(y_test, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
    }

    print("=" * 40)
    print("         EVALUATION REPORT")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"  {k.upper():5s}: {v}")
    print("=" * 40)

    os.makedirs(METRICS, exist_ok=True)
    with open(os.path.join(METRICS, "metrics.json"), "w") as f:
        json.dump({"test": metrics}, f, indent=2)

    print("Metrics saved to metrics/metrics.json")


if __name__ == "__main__":
    main()
