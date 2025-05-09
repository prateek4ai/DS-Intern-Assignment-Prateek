# scripts/preprocess.py

import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def load_and_clean(path="data/data.csv"):
    # 1. Load
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # 2. Coerce all other columns to numeric
    num_cols = df.columns.drop("timestamp", errors="ignore")
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    return df

def engineer_features(df):
    # Time‐based features
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["month"]       = df["timestamp"].dt.month

    # Zone aggregates
    temp_cols  = [f"zone{i}_temperature" for i in range(1,10)]
    humid_cols = [f"zone{i}_humidity"    for i in range(1,10)]

    df["zone_temp_mean"]  = df[temp_cols].mean(axis=1)
    df["zone_temp_std"]   = df[temp_cols].std(axis=1)
    df["zone_humid_mean"] = df[humid_cols].mean(axis=1)
    df["zone_humid_std"]  = df[humid_cols].std(axis=1)

    # Interaction term
    df["temp_humid_interaction"] = df["zone_temp_mean"] * df["zone_humid_mean"]
    return df

def build_preprocessor(df):
    """
    Returns a ColumnTransformer that:
      - extracts and scales the engineered time & aggregate features
      - imputes & scales the remaining numeric sensor features
    """
    # Features to exclude from numeric imputation
    to_drop = ["timestamp", "equipment_energy_consumption"]

    # All numeric columns (including engineered)
    all_numeric = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [c for c in all_numeric if c not in to_drop]

    # Decide which of those are "engineered" vs raw
    engineered = ["hour", "day_of_week", "is_weekend", "month",
                  "zone_temp_mean", "zone_temp_std",
                  "zone_humid_mean", "zone_humid_std",
                  "temp_humid_interaction"]
    raw_numeric = [c for c in feature_cols if c not in engineered]

    # Pipeline for engineered features (no missing or minimal; just scale)
    engineered_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    # Pipeline for raw sensor features
    raw_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("eng", engineered_pipeline, engineered),
        ("raw", raw_pipeline, raw_numeric),
    ], remainder="drop")

    return preprocessor

if __name__ == "__main__":
    # Example usage
    df = load_and_clean()
    df = engineer_features(df)
    preprocessor = build_preprocessor(df)

    # Split out features & target
    X = df.drop(["timestamp", "equipment_energy_consumption"], axis=1)
    y = df["equipment_energy_consumption"]

    # Fit & transform
    X_prepared = preprocessor.fit_transform(X)
    print("Preprocessing complete. Transformed shape:", X_prepared.shape)
