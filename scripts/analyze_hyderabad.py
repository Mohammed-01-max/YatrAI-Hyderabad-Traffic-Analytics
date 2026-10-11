from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hyderabad"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "hyderabad"
    / "eda"
)

CITY_FILE = PROCESSED_DIR / "daily_city_co2.csv"
GRID_FILE = PROCESSED_DIR / "grid_daily_co2.csv"


# ============================================================
# SETUP
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CITY-LEVEL ANALYSIS
# ============================================================

def analyze_city_data():

    print("=" * 80)
    print("CITY-LEVEL DATA QUALITY ANALYSIS")
    print("=" * 80)

    df = pd.read_csv(CITY_FILE)

    df["date"] = pd.to_datetime(df["date"])

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate dates:")
    print(df["date"].duplicated().sum())

    print("\nDate range:")
    print(df["date"].min())
    print("to")
    print(df["date"].max())

    print("\nCO2 statistics:")
    print(df["co2_tonnes"].describe())

    # --------------------------------------------------------
    # Highest / lowest days
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("TOP 10 HIGHEST-EMISSION DAYS")
    print("-" * 80)

    highest = (
        df.nlargest(10, "co2_tonnes")
        [["date", "co2_tonnes"]]
    )

    print(highest.to_string(index=False))

    print("\n" + "-" * 80)
    print("TOP 10 LOWEST-EMISSION DAYS")
    print("-" * 80)

    lowest = (
        df.nsmallest(10, "co2_tonnes")
        [["date", "co2_tonnes"]]
    )

    print(lowest.to_string(index=False))

    # --------------------------------------------------------
    # Weekly statistics
    # --------------------------------------------------------

    df["day_name"] = df["date"].dt.day_name()

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    weekday_stats = (
        df.groupby("day_name")["co2_tonnes"]
        .agg(["count", "mean", "std", "min", "max"])
        .reindex(weekday_order)
    )

    print("\n" + "-" * 80)
    print("DAY-OF-WEEK STATISTICS")
    print("-" * 80)

    print(weekday_stats)

    weekday_stats.to_csv(
        OUTPUT_DIR / "weekday_statistics.csv"
    )

    # --------------------------------------------------------
    # Monthly statistics
    # --------------------------------------------------------

    df["month"] = df["date"].dt.month

    monthly_stats = (
        df.groupby("month")["co2_tonnes"]
        .agg(["count", "mean", "std", "min", "max"])
    )

    print("\n" + "-" * 80)
    print("MONTHLY STATISTICS")
    print("-" * 80)

    print(monthly_stats)

    monthly_stats.to_csv(
        OUTPUT_DIR / "monthly_statistics.csv"
    )

    # --------------------------------------------------------
    # Plot 1: Daily CO2
    # --------------------------------------------------------

    plt.figure(figsize=(14, 6))

    plt.plot(
        df["date"],
        df["co2_tonnes"],
        linewidth=1
    )

    plt.title(
        "Hyderabad Daily Road-Traffic CO₂ Emissions — 2021"
    )

    plt.xlabel("Date")
    plt.ylabel("CO₂ emissions (tonnes/day)")

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "daily_co2_timeseries.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 2: 7-day rolling average
    # --------------------------------------------------------

    df["rolling_7"] = (
        df["co2_tonnes"]
        .rolling(7)
        .mean()
    )

    plt.figure(figsize=(14, 6))

    plt.plot(
        df["date"],
        df["co2_tonnes"],
        alpha=0.35,
        label="Daily"
    )

    plt.plot(
        df["date"],
        df["rolling_7"],
        linewidth=2,
        label="7-day rolling mean"
    )

    plt.title(
        "Hyderabad CO₂ Emissions with 7-Day Rolling Mean"
    )

    plt.xlabel("Date")
    plt.ylabel("CO₂ emissions (tonnes/day)")

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "daily_co2_rolling7.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 3: Day-of-week distribution
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    df.boxplot(
        column="co2_tonnes",
        by="day_name",
        grid=False
    )

    plt.title(
        "Hyderabad CO₂ Distribution by Day of Week"
    )

    plt.suptitle("")

    plt.xlabel("Day of week")
    plt.ylabel("CO₂ emissions (tonnes/day)")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "co2_by_weekday.png",
        dpi=200
    )

    plt.close()

    return df


# ============================================================
# GRID-LEVEL ANALYSIS
# ============================================================

def analyze_grid_data():

    print("\n" + "=" * 80)
    print("GRID-LEVEL DATA QUALITY ANALYSIS")
    print("=" * 80)

    df = pd.read_csv(GRID_FILE)

    df["date"] = pd.to_datetime(df["date"])

    print("\nShape:")
    print(df.shape)

    total_values = len(df)

    missing_values = df["co2_tonnes"].isna().sum()

    non_missing_values = df["co2_tonnes"].notna().sum()

    missing_pct = (
        missing_values / total_values * 100
    )

    print("\nTotal grid-day records:")
    print(f"{total_values:,}")

    print("\nNon-missing CO2 records:")
    print(f"{non_missing_values:,}")

    print("\nMissing CO2 records:")
    print(f"{missing_values:,}")

    print("\nMissing percentage:")
    print(f"{missing_pct:.2f}%")

    # --------------------------------------------------------
    # Active grid cells per day
    # --------------------------------------------------------

    active_cells = (
        df.groupby("date")["co2_tonnes"]
        .apply(lambda x: x.notna().sum())
        .rename("active_cells")
    )

    active_stats = active_cells.describe()

    print("\n" + "-" * 80)
    print("ACTIVE GRID CELLS PER DAY")
    print("-" * 80)

    print(active_stats)

    print("\nDays with zero active cells:")
    print((active_cells == 0).sum())

    # --------------------------------------------------------
    # Missing cells per day
    # --------------------------------------------------------

    daily_missing = (
        df.groupby("date")["co2_tonnes"]
        .apply(lambda x: x.isna().sum())
        .rename("missing_cells")
    )

    daily_coverage = pd.concat(
        [
            active_cells,
            daily_missing
        ],
        axis=1
    )

    daily_coverage["total_cells"] = (
        daily_coverage["active_cells"]
        + daily_coverage["missing_cells"]
    )

    daily_coverage["coverage_pct"] = (
        daily_coverage["active_cells"]
        / daily_coverage["total_cells"]
        * 100
    )

    print("\nCoverage statistics:")
    print(
        daily_coverage["coverage_pct"].describe()
    )

    daily_coverage.to_csv(
        OUTPUT_DIR / "daily_grid_coverage.csv"
    )

    # --------------------------------------------------------
    # Grid-cell activity
    # --------------------------------------------------------

    cell_activity = (
        df.groupby(["lat", "lon"])["co2_tonnes"]
        .agg(
            active_days=lambda x: x.notna().sum(),
            mean_co2="mean",
            total_co2="sum",
            max_co2="max"
        )
        .reset_index()
    )

    cell_activity["active_day_pct"] = (
        cell_activity["active_days"]
        / 365
        * 100
    )

    print("\n" + "-" * 80)
    print("GRID-CELL ACTIVITY")
    print("-" * 80)

    print(
        cell_activity[
            [
                "active_days",
                "active_day_pct",
                "mean_co2",
                "total_co2"
            ]
        ].describe()
    )

    cell_activity.to_csv(
        OUTPUT_DIR / "grid_cell_activity.csv",
        index=False
    )

    # --------------------------------------------------------
    # Spatial hotspot ranking
    # --------------------------------------------------------

    hotspots = (
        cell_activity
        .sort_values(
            "total_co2",
            ascending=False
        )
        .head(20)
    )

    print("\n" + "-" * 80)
    print("TOP 20 SPATIAL HOTSPOTS")
    print("-" * 80)

    print(
        hotspots[
            [
                "lat",
                "lon",
                "active_days",
                "active_day_pct",
                "mean_co2",
                "total_co2",
                "max_co2"
            ]
        ].to_string(index=False)
    )

    hotspots.to_csv(
        OUTPUT_DIR / "top_20_hotspots.csv",
        index=False
    )

    # --------------------------------------------------------
    # Spatial mean map
    # --------------------------------------------------------

    spatial_mean = (
        df.groupby(["lat", "lon"])["co2_tonnes"]
        .mean()
        .reset_index()
    )

    pivot = spatial_mean.pivot(
        index="lat",
        columns="lon",
        values="co2_tonnes"
    )

    plt.figure(figsize=(10, 8))

    plt.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[
            pivot.columns.min(),
            pivot.columns.max(),
            pivot.index.min(),
            pivot.index.max()
        ]
    )

    plt.colorbar(
        label="Mean CO₂ emissions"
    )

    plt.title(
        "Hyderabad Spatial Mean Road-Traffic CO₂ Emissions"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "hyderabad_spatial_mean_co2.png",
        dpi=200
    )

    plt.close()

    return daily_coverage, cell_activity


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("HYDERABAD CHETNA-ROAD EDA")
    print("=" * 80)

    if not CITY_FILE.exists():
        raise FileNotFoundError(
            f"Missing file:\n{CITY_FILE}"
        )

    if not GRID_FILE.exists():
        raise FileNotFoundError(
            f"Missing file:\n{GRID_FILE}"
        )

    analyze_city_data()

    analyze_grid_data()

    print("\n" + "=" * 80)
    print("EDA COMPLETED")
    print("=" * 80)

    print("\nOutput directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()