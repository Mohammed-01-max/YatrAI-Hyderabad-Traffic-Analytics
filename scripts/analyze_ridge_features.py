from pathlib import Path

import joblib
import numpy as np
import pandas as pd


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

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "hyderabad"
    / "hyderabad_forecasting_model.joblib"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "hyderabad"
    / "model"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FEATURES
# ============================================================

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


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("RIDGE FEATURE / COEFFICIENT ANALYSIS")
print("=" * 70)

print("\nModel:")
print(MODEL_FILE)

print("\nData:")
print(DATA_FILE)


model = joblib.load(
    MODEL_FILE
)

df = pd.read_csv(
    DATA_FILE
)

df["date"] = pd.to_datetime(
    df["date"]
)


# ============================================================
# VERIFY MODEL
# ============================================================

if not hasattr(
    model,
    "coef_"
):
    raise TypeError(
        "The saved model does not appear to be a linear Ridge model."
    )


if len(model.coef_) != len(FEATURES):
    raise ValueError(
        "Number of model coefficients does not match FEATURES."
    )


# ============================================================
# RAW COEFFICIENTS
# ============================================================

coef_df = pd.DataFrame({
    "feature": FEATURES,
    "coefficient": model.coef_,
})

coef_df["absolute_coefficient"] = (
    np.abs(
        coef_df["coefficient"]
    )
)

coef_df["direction"] = np.where(
    coef_df["coefficient"] > 0,
    "Positive",
    "Negative"
)

coef_df = coef_df.sort_values(
    "absolute_coefficient",
    ascending=False
).reset_index(drop=True)


# ============================================================
# STANDARDIZED COEFFICIENT IMPORTANCE
# ============================================================

X = df[FEATURES].copy()

feature_std = X.std(
    ddof=0
)

coef_df["feature_std"] = (
    coef_df["feature"]
    .map(feature_std)
)

coef_df["standardized_effect"] = (
    coef_df["coefficient"]
    * coef_df["feature_std"]
)

coef_df["absolute_standardized_effect"] = (
    np.abs(
        coef_df["standardized_effect"]
    )
)

coef_df["standardized_direction"] = np.where(
    coef_df["standardized_effect"] > 0,
    "Positive",
    "Negative"
)

coef_df = coef_df.sort_values(
    "absolute_standardized_effect",
    ascending=False
).reset_index(drop=True)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("STANDARDIZED FEATURE INFLUENCE")
print("=" * 70)

display_columns = [
    "feature",
    "coefficient",
    "feature_std",
    "standardized_effect",
    "standardized_direction",
]

print(
    coef_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

coef_df.to_csv(
    OUTPUT_DIR
    / "ridge_feature_coefficients.csv",
    index=False
)


# ============================================================
# TOP FEATURES
# ============================================================

top_features = coef_df.head(10).copy()

top_features.to_csv(
    OUTPUT_DIR
    / "ridge_top_10_features.csv",
    index=False
)


# ============================================================
# HUMAN-READABLE SUMMARY
# ============================================================

summary_rows = []

for _, row in top_features.iterrows():

    summary_rows.append({
        "feature": row["feature"],
        "effect_direction":
            row["standardized_direction"],
        "standardized_effect":
            row["standardized_effect"],
        "absolute_standardized_effect":
            row["absolute_standardized_effect"],
    })

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUTPUT_DIR
    / "ridge_feature_summary.csv",
    index=False
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("RIDGE FEATURE ANALYSIS COMPLETED")
print("=" * 70)

print("\nTop 10 features by standardized influence:")

print(
    summary_df.to_string(
        index=False
    )
)

print("\nSaved:")

print(
    OUTPUT_DIR
    / "ridge_feature_coefficients.csv"
)

print(
    OUTPUT_DIR
    / "ridge_top_10_features.csv"
)

print(
    OUTPUT_DIR
    / "ridge_feature_summary.csv"
)