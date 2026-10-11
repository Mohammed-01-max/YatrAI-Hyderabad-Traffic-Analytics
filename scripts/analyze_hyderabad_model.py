from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "hyderabad"
    / "model"
)

PREDICTIONS_FILE = (
    MODEL_OUTPUT_DIR
    / "test_predictions.csv"
)


# ============================================================
# LOAD TEST PREDICTIONS
# ============================================================

print("=" * 70)
print("HYDERABAD RIDGE MODEL — RESIDUAL ANALYSIS")
print("=" * 70)

print("\nLoading:")
print(PREDICTIONS_FILE)

if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Test predictions not found:\n{PREDICTIONS_FILE}\n\n"
        "Run train_hyderabad.py first."
    )

df = pd.read_csv(PREDICTIONS_FILE)

df["date"] = pd.to_datetime(df["date"])

required_columns = [
    "date",
    "actual_co2_tonnes",
    "predicted_co2_tonnes",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# RESIDUAL CALCULATIONS
# ============================================================

df["residual"] = (
    df["actual_co2_tonnes"]
    - df["predicted_co2_tonnes"]
)

df["absolute_error"] = (
    np.abs(df["residual"])
)

df["percentage_error"] = (
    np.abs(df["residual"])
    / df["actual_co2_tonnes"]
) * 100


# ============================================================
# OVERALL ERROR STATISTICS
# ============================================================

mae = df["absolute_error"].mean()

rmse = np.sqrt(
    np.mean(
        df["residual"] ** 2
    )
)

mape = df["percentage_error"].mean()

mean_residual = df["residual"].mean()

median_residual = df["residual"].median()

std_residual = df["residual"].std()

max_absolute_error = (
    df["absolute_error"].max()
)

min_absolute_error = (
    df["absolute_error"].min()
)

error_date = df.loc[
    df["absolute_error"].idxmax(),
    "date"
]

actual_at_max_error = df.loc[
    df["absolute_error"].idxmax(),
    "actual_co2_tonnes"
]

predicted_at_max_error = df.loc[
    df["absolute_error"].idxmax(),
    "predicted_co2_tonnes"
]


# ============================================================
# PRINT OVERALL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("OVERALL TEST ERROR ANALYSIS")
print("=" * 70)

print(
    f"\nTest observations       : {len(df)}"
)

print(
    f"MAE                     : {mae:.4f} tonnes/day"
)

print(
    f"RMSE                    : {rmse:.4f} tonnes/day"
)

print(
    f"MAPE                    : {mape:.4f}%"
)

print(
    f"Mean residual           : {mean_residual:.4f}"
)

print(
    f"Median residual         : {median_residual:.4f}"
)

print(
    f"Residual std deviation  : {std_residual:.4f}"
)

print(
    f"Minimum absolute error  : {min_absolute_error:.4f}"
)

print(
    f"Maximum absolute error  : {max_absolute_error:.4f}"
)

print(
    f"Maximum-error date      : {error_date.date()}"
)

print(
    f"Actual on that date     : "
    f"{actual_at_max_error:.4f}"
)

print(
    f"Predicted on that date  : "
    f"{predicted_at_max_error:.4f}"
)


# ============================================================
# ERROR DIRECTION
# ============================================================

overpredictions = (
    df["residual"] < 0
).sum()

underpredictions = (
    df["residual"] > 0
).sum()

exact_predictions = (
    df["residual"] == 0
).sum()

print("\n" + "=" * 70)
print("ERROR DIRECTION")
print("=" * 70)

print(
    f"\nUnderpredictions : {underpredictions}"
)

print(
    f"Overpredictions  : {overpredictions}"
)

print(
    f"Exact predictions: {exact_predictions}"
)


# ============================================================
# MONTHLY ERROR ANALYSIS
# ============================================================

df["month"] = df["date"].dt.month

monthly = (
    df.groupby("month")
    .agg(
        observations=("residual", "count"),
        MAE=("absolute_error", "mean"),
        RMSE=(
            "residual",
            lambda x: np.sqrt(
                np.mean(x ** 2)
            ),
        ),
        MAPE=(
            "percentage_error",
            "mean"
        ),
        mean_residual=(
            "residual",
            "mean"
        ),
    )
    .reset_index()
)

monthly["month_name"] = pd.to_datetime(
    monthly["month"],
    format="%m"
).dt.strftime("%B")

monthly = monthly[
    [
        "month",
        "month_name",
        "observations",
        "MAE",
        "RMSE",
        "MAPE",
        "mean_residual",
    ]
]

print("\n" + "=" * 70)
print("MONTHLY TEST ERROR")
print("=" * 70)

print(
    monthly.to_string(
        index=False
    )
)


# ============================================================
# BEST / WORST TEST DAYS
# ============================================================

worst_days = (
    df.sort_values(
        "absolute_error",
        ascending=False
    )
    .head(10)
    [
        [
            "date",
            "actual_co2_tonnes",
            "predicted_co2_tonnes",
            "residual",
            "absolute_error",
            "percentage_error",
        ]
    ]
)

best_days = (
    df.sort_values(
        "absolute_error",
        ascending=True
    )
    .head(10)
    [
        [
            "date",
            "actual_co2_tonnes",
            "predicted_co2_tonnes",
            "residual",
            "absolute_error",
            "percentage_error",
        ]
    ]
)


# ============================================================
# SAVE DATA
# ============================================================

df.to_csv(
    MODEL_OUTPUT_DIR
    / "residual_analysis.csv",
    index=False
)

monthly.to_csv(
    MODEL_OUTPUT_DIR
    / "monthly_test_errors.csv",
    index=False
)

worst_days.to_csv(
    MODEL_OUTPUT_DIR
    / "worst_prediction_days.csv",
    index=False
)

best_days.to_csv(
    MODEL_OUTPUT_DIR
    / "best_prediction_days.csv",
    index=False
)

error_statistics = pd.DataFrame([
    {
        "test_observations": len(df),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "mean_residual": mean_residual,
        "median_residual": median_residual,
        "residual_std": std_residual,
        "min_absolute_error": min_absolute_error,
        "max_absolute_error": max_absolute_error,
        "max_error_date": error_date.date(),
        "underpredictions": underpredictions,
        "overpredictions": overpredictions,
    }
])

error_statistics.to_csv(
    MODEL_OUTPUT_DIR
    / "error_statistics.csv",
    index=False
)


# ============================================================
# PLOT 1 — ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["date"],
    df["actual_co2_tonnes"],
    label="Actual",
    linewidth=2
)

plt.plot(
    df["date"],
    df["predicted_co2_tonnes"],
    label="Predicted",
    linewidth=2
)

plt.xlabel("Date")
plt.ylabel(
    "Daily Road-Traffic CO₂ Emissions (tonnes/day)"
)

plt.title(
    "Hyderabad — Actual vs Predicted Daily Road-Traffic CO₂"
)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    MODEL_OUTPUT_DIR
    / "actual_vs_predicted.png",
    dpi=200
)

plt.close()


# ============================================================
# PLOT 2 — RESIDUALS OVER TIME
# ============================================================

plt.figure(figsize=(12, 6))

plt.axhline(
    0,
    linewidth=1
)

plt.plot(
    df["date"],
    df["residual"],
    linewidth=1.8
)

plt.xlabel("Date")
plt.ylabel(
    "Residual (Actual − Predicted)"
)

plt.title(
    "Hyderabad Ridge Model — Residuals Over Time"
)

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    MODEL_OUTPUT_DIR
    / "residuals_over_time.png",
    dpi=200
)

plt.close()


# ============================================================
# PLOT 3 — RESIDUAL DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["residual"],
    bins=20,
    edgecolor="black"
)

plt.axvline(
    0,
    linewidth=1
)

plt.xlabel(
    "Residual (Actual − Predicted)"
)

plt.ylabel("Frequency")

plt.title(
    "Distribution of Ridge Model Residuals"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    MODEL_OUTPUT_DIR
    / "residual_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# PLOT 4 — MONTHLY MAE
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    monthly["month_name"],
    monthly["MAE"]
)

plt.xlabel("Month")
plt.ylabel(
    "MAE (tonnes/day)"
)

plt.title(
    "Monthly Test MAE — Ridge Model"
)

plt.xticks(
    rotation=45
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    MODEL_OUTPUT_DIR
    / "monthly_test_mae.png",
    dpi=200
)

plt.close()


# ============================================================
# PLOT 5 — ABSOLUTE ERROR OVER TIME
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["date"],
    df["absolute_error"],
    linewidth=1.8
)

plt.xlabel("Date")
plt.ylabel(
    "Absolute Error (tonnes/day)"
)

plt.title(
    "Hyderabad Ridge Model — Absolute Prediction Error"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    MODEL_OUTPUT_DIR
    / "absolute_error_over_time.png",
    dpi=200
)

plt.close()


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("RESIDUAL ANALYSIS COMPLETED")
print("=" * 70)

print(
    "\nFiles saved to:"
)

print(
    MODEL_OUTPUT_DIR
)

print("\nGenerated:")

files = [
    "residual_analysis.csv",
    "monthly_test_errors.csv",
    "worst_prediction_days.csv",
    "best_prediction_days.csv",
    "error_statistics.csv",
    "actual_vs_predicted.png",
    "residuals_over_time.png",
    "residual_distribution.png",
    "monthly_test_mae.png",
    "absolute_error_over_time.png",
]

for file in files:
    print(f"  ✓ {file}")