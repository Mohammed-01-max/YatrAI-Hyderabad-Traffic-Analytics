from pathlib import Path
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "hyderabad"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "hyderabad"
)

SPATIAL_DIR = (
    BASE_DIR
    / "outputs"
    / "hyderabad"
    / "spatial"
)


DAILY_FILE = DATA_DIR / "daily_city_co2.csv"

FEATURE_FILE = DATA_DIR / "forecasting_features.csv"

MODEL_FILE = MODEL_DIR / "hyderabad_forecasting_model.joblib"

HOTSPOT_FILE = (
    SPATIAL_DIR
    / "hyderabad_validated_p95_hotspots.csv"
)


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


def load_daily_data():
    if not DAILY_FILE.exists():
        raise FileNotFoundError(
            f"Hyderabad daily dataset not found: {DAILY_FILE}"
        )

    df = pd.read_csv(DAILY_FILE)

    df["date"] = pd.to_datetime(df["date"])

    return df


def load_forecasting_data():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Forecasting feature dataset not found: {FEATURE_FILE}"
        )

    df = pd.read_csv(FEATURE_FILE)

    df["date"] = pd.to_datetime(df["date"])

    return df


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Hyderabad Ridge model not found: {MODEL_FILE}"
        )

    return joblib.load(MODEL_FILE)


def load_hotspots():
    if not HOTSPOT_FILE.exists():
        raise FileNotFoundError(
            f"Validated hotspot file not found: {HOTSPOT_FILE}"
        )

    df = pd.read_csv(HOTSPOT_FILE)

    return df


def get_summary():
    df = load_daily_data()

    return {
        "city": "Hyderabad",
        "state": "Telangana",
        "dataset_year": 2021,
        "observations": int(len(df)),
        "start_date": df["date"].min().strftime("%Y-%m-%d"),
        "end_date": df["date"].max().strftime("%Y-%m-%d"),
        "mean_daily_co2_tonnes": float(
            df["co2_tonnes"].mean()
        ),
        "minimum_daily_co2_tonnes": float(
            df["co2_tonnes"].min()
        ),
        "maximum_daily_co2_tonnes": float(
            df["co2_tonnes"].max()
        ),
    }


def get_latest_observation():
    df = load_daily_data()

    row = df.sort_values("date").iloc[-1]

    return {
        "date": row["date"].strftime("%Y-%m-%d"),
        "co2_tonnes": float(row["co2_tonnes"]),
    }


def get_model_info():
    model = load_model()

    return {
        "model": "Ridge Regression",
        "algorithm": type(model).__name__,
        "alpha": float(model.alpha),
        "features": len(FEATURES),
        "training_observations": 280,
        "forecasting_target": (
            "Daily road-traffic CO2 emissions "
            "(tonnes/day)"
        ),
        "dataset_year": 2021,
        "forecast_type": (
            "Rolling one-day-ahead historical forecasting"
        ),
    }


def get_hotspots(limit=20):
    df = load_hotspots()

    df = df.sort_values(
        "mean_co2_tonnes",
        ascending=False
    ).head(limit)

    records = []

    for _, row in df.iterrows():

        records.append({
            "rank": int(row["hotspot_rank"]),
            "latitude": float(row["lat"]),
            "longitude": float(row["lon"]),
            "mean_co2_tonnes": float(
                row["mean_co2_tonnes"]
            ),
            "total_co2_tonnes": float(
                row["total_co2_tonnes"]
            ),
            "max_daily_co2_tonnes": float(
                row["max_daily_co2_tonnes"]
            ),
            "active_days": int(
                row["active_days"]
            ),
            "coverage_percent": float(
                row["coverage_percent"]
            ),
        })

    return {
        "threshold": "95th percentile",
        "minimum_temporal_coverage": "90%",
        "count": len(records),
        "hotspots": records,
    }