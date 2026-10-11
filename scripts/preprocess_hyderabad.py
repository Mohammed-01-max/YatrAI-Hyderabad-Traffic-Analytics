from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hyderabad"
    / "chetna_hyderabad_co2_2021.nc"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hyderabad"
)

CITY_OUTPUT = PROCESSED_DIR / "daily_city_co2.csv"
GRID_OUTPUT = PROCESSED_DIR / "grid_daily_co2.csv"
FEATURE_OUTPUT = PROCESSED_DIR / "forecasting_features.csv"


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_CITY = "Hyderabad"
EXPECTED_STATE = "Telangana"
EXPECTED_YEAR = 2021


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(ds):
    print("=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)

    city = ds.attrs.get("city")
    state = ds.attrs.get("state")
    year = ds.attrs.get("year")

    print(f"City : {city}")
    print(f"State: {state}")
    print(f"Year : {year}")

    if city != EXPECTED_CITY:
        raise ValueError(
            f"Wrong city detected. Expected {EXPECTED_CITY}, got {city}"
        )

    if state != EXPECTED_STATE:
        raise ValueError(
            f"Wrong state detected. Expected {EXPECTED_STATE}, got {state}"
        )

    if int(year) != EXPECTED_YEAR:
        raise ValueError(
            f"Wrong year detected. Expected {EXPECTED_YEAR}, got {year}"
        )

    required_dimensions = {"time", "lat", "lon"}

    if not required_dimensions.issubset(ds.dims):
        raise ValueError(
            f"Missing required dimensions. Found: {list(ds.dims)}"
        )

    if "co2" not in ds.data_vars:
        raise ValueError(
            "CO2 variable not found. Expected variable: co2"
        )

    print("\nValidation successful.")


# ============================================================
# CITY-LEVEL DAILY DATA
# ============================================================

def create_city_daily_data(ds):
    print("\n" + "=" * 80)
    print("CREATING CITY-LEVEL DAILY DATA")
    print("=" * 80)

    co2 = ds["co2"]

    # Sum across all spatial grid cells for each day.
    city_daily = co2.sum(
        dim=("lat", "lon"),
        skipna=True
    )

    df = city_daily.to_dataframe(
        name="co2_tonnes"
    ).reset_index()

    df["date"] = pd.to_datetime(df["time"])

    df = df.drop(columns=["time"])

    df = df[
        [
            "date",
            "co2_tonnes"
        ]
    ]

    # Ensure chronological order.
    df = df.sort_values("date").reset_index(drop=True)

    # Basic checks.
    if len(df) != 365:
        raise ValueError(
            f"Expected 365 daily observations, found {len(df)}"
        )

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found.")

    if df["co2_tonnes"].isna().any():
        print("WARNING: Missing city-level CO2 values detected.")

    if (df["co2_tonnes"] < 0).any():
        print("WARNING: Negative CO2 values detected.")

    print(f"Rows: {len(df)}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    print("\nCO2 statistics:")
    print(df["co2_tonnes"].describe())

    return df


# ============================================================
# GRID-LEVEL DAILY DATA
# ============================================================

def create_grid_daily_data(ds):
    print("\n" + "=" * 80)
    print("CREATING GRID-LEVEL DATA")
    print("=" * 80)

    co2 = ds["co2"]

    # Convert the 3D array into long-format tabular data.
    df = co2.to_dataframe(
        name="co2_tonnes"
    ).reset_index()

    df["date"] = pd.to_datetime(df["time"])

    df = df.drop(columns=["time"])

    df = df[
        [
            "date",
            "lat",
            "lon",
            "co2_tonnes"
        ]
    ]

    df = df.sort_values(
        ["date", "lat", "lon"]
    ).reset_index(drop=True)

    print(f"Rows: {len(df):,}")
    print(f"Grid cells per day: {df.groupby('date').size().iloc[0]:,}")
    print(f"Unique latitudes: {df['lat'].nunique()}")
    print(f"Unique longitudes: {df['lon'].nunique()}")

    print("\nMissing values:")
    print(df.isna().sum())

    return df


# ============================================================
# FORECASTING FEATURES
# ============================================================

def create_forecasting_features(city_df):
    print("\n" + "=" * 80)
    print("CREATING FORECASTING FEATURES")
    print("=" * 80)

    df = city_df.copy()

    df = df.sort_values("date").reset_index(drop=True)

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    df["dow"] = df["date"].dt.dayofweek

    df["month"] = df["date"].dt.month

    df["dayofyear"] = df["date"].dt.dayofyear

    df["dayofmonth"] = df["date"].dt.day

    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

    df["is_weekend"] = (
        df["dow"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Cyclical seasonal features
    # --------------------------------------------------------

    df["sin_doy"] = np.sin(
        2 * np.pi * df["dayofyear"] / 365.0
    )

    df["cos_doy"] = np.cos(
        2 * np.pi * df["dayofyear"] / 365.0
    )

    df["sin_dow"] = np.sin(
        2 * np.pi * df["dow"] / 7.0
    )

    df["cos_dow"] = np.cos(
        2 * np.pi * df["dow"] / 7.0
    )

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    df["lag1"] = df["co2_tonnes"].shift(1)

    df["lag2"] = df["co2_tonnes"].shift(2)

    df["lag3"] = df["co2_tonnes"].shift(3)

    df["lag7"] = df["co2_tonnes"].shift(7)

    df["lag14"] = df["co2_tonnes"].shift(14)

    # --------------------------------------------------------
    # Rolling statistics
    #
    # shift(1) is VERY important.
    #
    # It prevents today's target from leaking into today's
    # rolling features.
    # --------------------------------------------------------

    previous_values = df["co2_tonnes"].shift(1)

    df["roll3_mean"] = (
        previous_values
        .rolling(window=3)
        .mean()
    )

    df["roll7_mean"] = (
        previous_values
        .rolling(window=7)
        .mean()
    )

    df["roll7_std"] = (
        previous_values
        .rolling(window=7)
        .std()
    )

    df["roll14_mean"] = (
        previous_values
        .rolling(window=14)
        .mean()
    )

    # --------------------------------------------------------
    # Short-term trend
    # --------------------------------------------------------

    df["trend7"] = (
        df["lag1"] - df["lag7"]
    )

    # --------------------------------------------------------
    # Remove rows where lag/rolling features are unavailable.
    #
    # First 14 days cannot have all required lag features.
    # --------------------------------------------------------

    feature_columns = [
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

    before = len(df)

    df = df.dropna(
        subset=feature_columns + ["co2_tonnes"]
    ).reset_index(drop=True)

    after = len(df)

    print(f"Rows before feature cleanup: {before}")
    print(f"Rows after feature cleanup : {after}")
    print(f"Rows removed               : {before - after}")

    print("\nFeature columns:")
    for column in feature_columns:
        print(f"  - {column}")

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("HYDERABAD CHETNA-ROAD PREPROCESSING PIPELINE")
    print("=" * 80)

    print(f"\nRaw file:")
    print(RAW_FILE)

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_FILE}"
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nOpening NetCDF...")

    with xr.open_dataset(RAW_FILE) as ds:

        validate_dataset(ds)

        city_df = create_city_daily_data(ds)

        grid_df = create_grid_daily_data(ds)

    # Create forecasting features outside the xarray context.
    feature_df = create_forecasting_features(city_df)

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    city_df.to_csv(
        CITY_OUTPUT,
        index=False
    )

    grid_df.to_csv(
        GRID_OUTPUT,
        index=False
    )

    feature_df.to_csv(
        FEATURE_OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FILES CREATED")
    print("=" * 80)

    print(f"\n1. City daily data:")
    print(CITY_OUTPUT)

    print(f"\n2. Grid daily data:")
    print(GRID_OUTPUT)

    print(f"\n3. Forecasting features:")
    print(FEATURE_OUTPUT)

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()