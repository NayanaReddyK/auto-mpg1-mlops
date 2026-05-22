"""
train.py
Reads processed splits, trains the model selected in params.yaml,
saves the model to models/model.pkl, and logs metrics to metrics/train_metrics.json.
"""

import os
import json
import yaml
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

PROCESSED  = os.path.join("data", "processed")
MODELS_DIR = "models"
METRICS    = "metrics"
PARAMS_FILE = "params.yaml"


def load_params():
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def build_model(params):
    mtype = params["model"]["type"]
    if mtype == "linear_regression":
        return LinearRegression()
    elif mtype == "svr_rbf":
        return SVR(kernel=params["svr"]["kernel"], epsilon=params["svr"]["epsilon"])
    elif mtype == "decision_tree":
        return DecisionTreeRegressor(max_depth=params["decision_tree"]["max_depth"])
    else:
        raise ValueError(f"Unknown model type: {mtype}")


def compute_metrics(y_true, y_pred):
    return {
        "r2":  round(float(r2_score(y_true, y_pred)), 4),
        "mse": round(float(mean_squared_error(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
    }


def main():
    params = load_params()

    X_train = np.load(os.path.join(PROCESSED, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED, "y_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED, "X_test.npy"))
    y_test  = np.load(os.path.join(PROCESSED, "y_test.npy"))

    model = build_model(params)
    print(f"Training: {params['model']['type']} ...")
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    train_metrics = compute_metrics(y_train, y_pred_train)
    test_metrics  = compute_metrics(y_test,  y_pred_test)

    print(f"  Train → R2={train_metrics['r2']}  MSE={train_metrics['mse']}  MAE={train_metrics['mae']}")
    print(f"  Test  → R2={test_metrics['r2']}   MSE={test_metrics['mse']}   MAE={test_metrics['mae']}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS, exist_ok=True)

    joblib.dump(model, os.path.join(MODELS_DIR, "model.pkl"))

    with open(os.path.join(METRICS, "metrics.json"), "w") as f:
        json.dump({"train": train_metrics, "test": test_metrics}, f, indent=2)

    print("Model saved to models/model.pkl")
    print("Metrics saved to metrics/metrics.json")


if __name__ == "__main__":
    main()
