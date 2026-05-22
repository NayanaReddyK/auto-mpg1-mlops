import os
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

RAW_DATA   = os.path.join("data", "auto-mpg.data")
PROCESSED  = os.path.join("data", "processed")
MODELS_DIR = "models"
PARAMS_FILE = "params.yaml"

COLUMN_NAMES = [
    "mpg", "cylinders", "displacement", "horsepower",
    "weight", "acceleration", "model_year", "origin", "car_name",
]

def load_params():
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)

def load_and_clean(path):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            # first 8 are numeric fields, rest is car_name
            row = parts[:8] + [' '.join(parts[8:])]
            rows.append(row)
    df = pd.DataFrame(rows, columns=COLUMN_NAMES)
    df = df[df["horsepower"] != "?"].copy()
    for col in ["mpg","cylinders","displacement","horsepower",
                "weight","acceleration","model_year","origin"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    return df

def engineer_features(df):
    X = df.drop("mpg", axis=1).copy()
    y = df["mpg"]
    le = LabelEncoder()
    X["car_name"] = le.fit_transform(X["car_name"])
    return X, y, le

def main():
    params = load_params()
    test_size    = params["data"]["test_size"]
    random_state = params["data"]["random_state"]

    print("Loading raw data ...")
    df = load_and_clean(RAW_DATA)
    print(f"  {len(df)} rows after cleaning")

    X, y, le = engineer_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    os.makedirs(PROCESSED, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    np.save(os.path.join(PROCESSED, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(PROCESSED, "X_test.npy"),  X_test_scaled)
    np.save(os.path.join(PROCESSED, "y_train.npy"), y_train.values)
    np.save(os.path.join(PROCESSED, "y_test.npy"),  y_test.values)

    pd.Series(X.columns.tolist()).to_csv(
        os.path.join(PROCESSED, "feature_names.csv"), index=False, header=False
    )

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(le,     os.path.join(MODELS_DIR, "label_encoder.pkl"))

    print("Preprocessing done. Artifacts saved.")

if __name__ == "__main__":
    main()