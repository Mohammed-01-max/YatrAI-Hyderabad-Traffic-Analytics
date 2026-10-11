from pathlib import Path
import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "hyderabad"
    / "hyderabad_forecasting_model.joblib"
)

PREDICTIONS_FILE = (
    BASE_DIR
    / "outputs"
    / "hyderabad"
    / "model"
    / "test_predictions.csv"
)


print("=" * 70)
print("HYDERABAD RIDGE MODEL — FINAL SANITY CHECK")
print("=" * 70)


print("\nModel:")
print(MODEL_FILE)

model = joblib.load(MODEL_FILE)


print("\nModel type:")
print(type(model))


print("\nModel parameters:")

if hasattr(model, "alpha"):
    print("alpha:", model.alpha)

if hasattr(model, "fit_intercept"):
    print("fit_intercept:", model.fit_intercept)

if hasattr(model, "coef_"):
    print("number of coefficients:", len(model.coef_))

if hasattr(model, "intercept_"):
    print("intercept:", model.intercept_)


print("\nPrediction file:")
print(PREDICTIONS_FILE)


if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Prediction file not found:\n{PREDICTIONS_FILE}"
    )


pred = pd.read_csv(PREDICTIONS_FILE)


print("\nPrediction columns:")
print(pred.columns.tolist())


print("\nPrediction preview:")
print(pred.head().to_string(index=False))


print("\nPrediction shape:")
print(pred.shape)


print("\n" + "=" * 70)
print("BASIC SANITY CHECK COMPLETED")
print("=" * 70)