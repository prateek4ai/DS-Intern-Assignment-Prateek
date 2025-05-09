#!/usr/bin/env python3
"""
scripts/predict.py

Load a trained model and predict equipment energy consumption on new data.
"""

import os
import sys
import pandas as pd
import joblib

# Ensure we can import from scripts/
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from preprocess import load_and_clean, engineer_features

def main():
    if len(sys.argv) != 3:
        print("Usage: python predict.py <input_csv> <output_csv>")
        sys.exit(1)

    input_csv, output_csv = sys.argv[1], sys.argv[2]

    # 1. Load & clean
    df = load_and_clean(input_csv)
    df = engineer_features(df)

    # 2. Keep timestamps for output
    ids = df["timestamp"]

    # 3. Prepare feature matrix
    X = df.drop(
        ["timestamp", "equipment_energy_consumption"],
        axis=1,
        errors="ignore"
    )

    # 4. Load model
    model_path = os.path.join("models", "RandomForest_tuned.pkl")
    model = joblib.load(model_path)

    # 5. Predict
    preds = model.predict(X)

    # 6. Make sure output directory exists
    out_dir = os.path.dirname(output_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 7. Save results
    out = pd.DataFrame({
        "timestamp": ids,
        "predicted_energy_consumption": preds
    })
    out.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    main()
