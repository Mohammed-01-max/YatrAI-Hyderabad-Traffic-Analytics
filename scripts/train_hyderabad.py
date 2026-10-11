from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "hyderabad"
    / "forecasting_features.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "hyderabad"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "hyderabad"
    / "model"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "co2_tonnes"

FEATURES = [
    "dow",
    "month",
    "dayofyear",
    "dayofmonth",
    "weekofyear",
    "is_weekend",
    "sin_doy",
    "cos_doy",
    "sin_dow",
    "cos_dow",
    "lag1",
    "lag2",
    "lag3",
    "lag7",
    "lag14",
    "roll3_mean",
    "roll7_mean",
    "roll7_std",
    "roll14_mean",
    "trend7",
]

# Same chronological split methodology
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.10


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    mape = np.mean(
        np.abs(
            (y_true - y_pred) / y_true
        )
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
    }


# ============================================================
# SEASONAL NAIVE
# ============================================================

def seasonal_naive_prediction(df, season_length):
    """
    Rolling one-day-ahead seasonal naive forecast.

    Prediction for date t:
        actual value from t-season_length

    This is valid because t-season_length is already observed
    when forecasting date t.
    """

    return df[TARGET].shift(season_length)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("HYDERABAD TRAFFIC FORECASTING EXPERIMENT")
print("=" * 70)

print("\nLoading data:")
print(DATA_FILE)

df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print(f"\nTotal usable observations: {len(df)}")
print(
    f"Date range: "
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)

# ============================================================
# VALIDATE
# ============================================================

required_columns = ["date", TARGET] + FEATURES

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df[TARGET].isna().any():
    raise ValueError(
        "Target contains missing values."
    )

print("\nDataset validation: PASSED")


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(df)

train_end = int(n * TRAIN_RATIO)

validation_end = int(
    n * (TRAIN_RATIO + VALIDATION_RATIO)
)

train_df = df.iloc[:train_end].copy()

validation_df = df.iloc[
    train_end:validation_end
].copy()

test_df = df.iloc[
    validation_end:
].copy()

print("\n" + "=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print(
    f"TRAIN      : {len(train_df)} rows | "
    f"{train_df['date'].min().date()} → "
    f"{train_df['date'].max().date()}"
)

print(
    f"VALIDATION : {len(validation_df)} rows | "
    f"{validation_df['date'].min().date()} → "
    f"{validation_df['date'].max().date()}"
)

print(
    f"TEST       : {len(test_df)} rows | "
    f"{test_df['date'].min().date()} → "
    f"{test_df['date'].max().date()}"
)


# ============================================================
# FEATURE MATRICES
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_val = validation_df[FEATURES]
y_val = validation_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {
    "Ridge": Ridge(
        alpha=10.0
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.03,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    ),

    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=2,
        random_state=42,
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    ),

    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    ),
}


# ============================================================
# VALIDATION EXPERIMENT
# ============================================================

validation_results = []

validation_predictions = {}

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)


# ------------------------------------------------------------
# Seasonal Naive models
# ------------------------------------------------------------

for season_length in [7, 14, 28]:

    # Prediction is generated using the original complete
    # chronological series, then restricted to validation dates.
    seasonal_prediction = seasonal_naive_prediction(
        df,
        season_length
    )

    predictions = seasonal_prediction.iloc[
        validation_df.index
    ].values

    valid_mask = ~np.isnan(predictions)

    metrics = calculate_metrics(
        y_val.values[valid_mask],
        predictions[valid_mask]
    )

    model_name = (
        f"SeasonalNaive_{season_length}D"
    )

    validation_results.append({
        "Model": model_name,
        **metrics
    })

    validation_predictions[model_name] = predictions

    print(
        f"\n{model_name}"
        f"\n  MAE  : {metrics['MAE']:.4f}"
        f"\n  RMSE : {metrics['RMSE']:.4f}"
        f"\n  R²   : {metrics['R2']:.4f}"
        f"\n  MAPE : {metrics['MAPE']:.2f}%"
    )


# ------------------------------------------------------------
# ML models
# ------------------------------------------------------------

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_val
    )

    metrics = calculate_metrics(
        y_val,
        predictions
    )

    validation_results.append({
        "Model": name,
        **metrics
    })

    validation_predictions[name] = predictions

    print(
        f"  MAE  : {metrics['MAE']:.4f}"
        f"\n  RMSE : {metrics['RMSE']:.4f}"
        f"\n  R²   : {metrics['R2']:.4f}"
        f"\n  MAPE : {metrics['MAPE']:.2f}%"
    )


# ============================================================
# VALIDATION COMPARISON
# ============================================================

validation_results_df = pd.DataFrame(
    validation_results
)

validation_results_df = validation_results_df.sort_values(
    "RMSE"
).reset_index(drop=True)

validation_results_df.to_csv(
    OUTPUT_DIR / "validation_model_comparison.csv",
    index=False
)

print("\n" + "=" * 70)
print("VALIDATION MODEL RANKING")
print("=" * 70)

print(
    validation_results_df.to_string(
        index=False
    )
)


# ============================================================
# MODEL SELECTION
# ============================================================

best_model_name = validation_results_df.iloc[0]["Model"]

print("\n" + "=" * 70)
print("MODEL SELECTION")
print("=" * 70)

print(
    f"Selected model based ONLY on validation RMSE:"
    f"\n{best_model_name}"
)


# ============================================================
# REFIT BEST MODEL ON TRAIN + VALIDATION
# ============================================================

train_val_df = pd.concat(
    [
        train_df,
        validation_df
    ],
    ignore_index=True
)

X_train_val = train_val_df[FEATURES]
y_train_val = train_val_df[TARGET]


# ------------------------------------------------------------
# Seasonal naive
# ------------------------------------------------------------

if best_model_name.startswith(
    "SeasonalNaive_"
):

    season_length = int(
        best_model_name.split("_")[1].replace("D", "")
    )

    final_predictions = (
        df[TARGET]
        .shift(season_length)
        .iloc[test_df.index]
        .values
    )

    final_model = {
        "type": "seasonal_naive",
        "season_length": season_length,
    }

    test_mask = ~np.isnan(
        final_predictions
    )

    test_metrics = calculate_metrics(
        y_test.values[test_mask],
        final_predictions[test_mask]
    )

else:

    final_model = models[
        best_model_name
    ]

    print(
        f"\nRefitting {best_model_name} "
        "on train + validation..."
    )

    final_model.fit(
        X_train_val,
        y_train_val
    )

    final_predictions = final_model.predict(
        X_test
    )

    test_metrics = calculate_metrics(
        y_test,
        final_predictions
    )


# ============================================================
# FINAL TEST RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL UNTOUCHED TEST RESULTS")
print("=" * 70)

print(
    f"Model: {best_model_name}"
)

print(
    f"MAE  : {test_metrics['MAE']:.4f}"
)

print(
    f"RMSE : {test_metrics['RMSE']:.4f}"
)

print(
    f"R²   : {test_metrics['R2']:.4f}"
)

print(
    f"MAPE : {test_metrics['MAPE']:.2f}%"
)


# ============================================================
# SAVE FINAL METRICS
# ============================================================

final_metrics_df = pd.DataFrame([
    {
        "Model": best_model_name,
        **test_metrics
    }
])

final_metrics_df.to_csv(
    OUTPUT_DIR / "final_test_metrics.csv",
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({
    "date": test_df["date"].values,
    "actual_co2_tonnes": y_test.values,
    "predicted_co2_tonnes": final_predictions,
})

predictions_df["absolute_error"] = (
    np.abs(
        predictions_df["actual_co2_tonnes"]
        - predictions_df["predicted_co2_tonnes"]
    )
)

predictions_df.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

if isinstance(final_model, dict):

    joblib.dump(
        final_model,
        MODEL_DIR / "hyderabad_forecasting_model.joblib"
    )

else:

    joblib.dump(
        final_model,
        MODEL_DIR / "hyderabad_forecasting_model.joblib"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

if not isinstance(final_model, dict):

    if hasattr(
        final_model,
        "feature_importances_"
    ):

        importance_df = pd.DataFrame({
            "feature": FEATURES,
            "importance":
                final_model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            "importance",
            ascending=False
        )

        importance_df.to_csv(
            OUTPUT_DIR / "feature_importance.csv",
            index=False
        )


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "dataset": "CHETNA-Road Hyderabad 2021",
    "target": "daily road-traffic CO2 emissions (tonnes/day)",
    "forecast_type": "rolling one-day-ahead",
    "total_observations": len(df),
    "train_rows": len(train_df),
    "validation_rows": len(validation_df),
    "test_rows": len(test_df),
    "selected_model": best_model_name,
    "test_MAE": test_metrics["MAE"],
    "test_RMSE": test_metrics["RMSE"],
    "test_R2": test_metrics["R2"],
    "test_MAPE": test_metrics["MAPE"],
}

pd.DataFrame(
    [summary]
).to_csv(
    OUTPUT_DIR / "experiment_summary.csv",
    index=False
)

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

print(
    f"\nSelected model: {best_model_name}"
)

print(
    f"Test RMSE: {test_metrics['RMSE']:.4f}"
)

print(
    f"Test MAPE: {test_metrics['MAPE']:.2f}%"
)

print(
    f"\nResults saved to:\n{OUTPUT_DIR}"
)

print(
    f"\nModel saved to:\n"
    f"{MODEL_DIR / 'hyderabad_forecasting_model.joblib'}"
)