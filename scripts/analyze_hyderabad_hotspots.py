from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]

GRID_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "hyderabad"
    / "grid_daily_co2.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "hyderabad"
    / "spatial"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("HYDERABAD CHETNA-ROAD — SPATIAL HOTSPOT ANALYSIS")
print("=" * 70)

print("\nInput:")
print(GRID_FILE)

print("\nOutput:")
print(OUTPUT_DIR)


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

if not GRID_FILE.exists():
    raise FileNotFoundError(
        f"Grid dataset not found:\n{GRID_FILE}"
    )

df = pd.read_csv(GRID_FILE)

df["date"] = pd.to_datetime(df["date"])

print("\nDataset loaded.")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())


# ---------------------------------------------------------
# 2. BASIC VALIDATION
# ---------------------------------------------------------

required_columns = [
    "date",
    "lat",
    "lon",
    "co2_tonnes",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


print("\nDate range:")
print(df["date"].min(), "→", df["date"].max())

print("\nUnique spatial cells:")
print(df[["lat", "lon"]].drop_duplicates().shape[0])

print("\nMissing CO2 values:")
print(df["co2_tonnes"].isna().sum())

print(
    "Missing percentage:",
    round(df["co2_tonnes"].isna().mean() * 100, 2),
    "%"
)


# ---------------------------------------------------------
# 3. SPATIAL AGGREGATION
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CALCULATING SPATIAL STATISTICS")
print("=" * 70)

grouped = (
    df.groupby(["lat", "lon"], as_index=False)
    .agg(
        mean_co2_tonnes=("co2_tonnes", "mean"),
        total_co2_tonnes=("co2_tonnes", "sum"),
        max_daily_co2_tonnes=("co2_tonnes", "max"),
        active_days=("co2_tonnes", "count"),
    )
)

grouped["coverage_percent"] = (
    grouped["active_days"] / df["date"].nunique()
) * 100


# ---------------------------------------------------------
# 4. HOTSPOT RANKING
# ---------------------------------------------------------

hotspots = grouped.sort_values(
    "mean_co2_tonnes",
    ascending=False
).reset_index(drop=True)

hotspots["hotspot_rank"] = (
    hotspots.index + 1
)


print("\nTop 20 spatial hotspots:")
print(
    hotspots.head(20).to_string(index=False)
)


# ---------------------------------------------------------
# 5. SAVE COMPLETE SPATIAL STATISTICS
# ---------------------------------------------------------

spatial_file = (
    OUTPUT_DIR
    / "hyderabad_spatial_statistics.csv"
)

hotspots.to_csv(
    spatial_file,
    index=False
)


# ---------------------------------------------------------
# 6. SAVE TOP 20 HOTSPOTS
# ---------------------------------------------------------

top20 = hotspots.head(20).copy()

top20_file = (
    OUTPUT_DIR
    / "top_20_hyderabad_hotspots.csv"
)

top20.to_csv(
    top20_file,
    index=False
)


# ---------------------------------------------------------
# 7. TOP 10 HOTSPOTS
# ---------------------------------------------------------

top10 = hotspots.head(10).copy()

top10_file = (
    OUTPUT_DIR
    / "top_10_hyderabad_hotspots.csv"
)

top10.to_csv(
    top10_file,
    index=False
)


# ---------------------------------------------------------
# 8. HOTSPOT THRESHOLD ANALYSIS
# ---------------------------------------------------------

valid_mean = hotspots["mean_co2_tonnes"].dropna()

p90 = valid_mean.quantile(0.90)
p95 = valid_mean.quantile(0.95)

hotspots["hotspot_p90"] = (
    hotspots["mean_co2_tonnes"] >= p90
)

hotspots["hotspot_p95"] = (
    hotspots["mean_co2_tonnes"] >= p95
)

threshold_file = (
    OUTPUT_DIR
    / "hyderabad_hotspot_thresholds.csv"
)

hotspots.to_csv(
    threshold_file,
    index=False
)


print("\n" + "=" * 70)
print("HOTSPOT THRESHOLDS")
print("=" * 70)

print(f"90th percentile: {p90:.6f} tonnes/day")
print(f"95th percentile: {p95:.6f} tonnes/day")

print(
    "P90 hotspot cells:",
    hotspots["hotspot_p90"].sum()
)

print(
    "P95 hotspot cells:",
    hotspots["hotspot_p95"].sum()
)


# ---------------------------------------------------------
# 9. SPATIAL HEATMAP — MEAN CO2
# ---------------------------------------------------------

pivot_mean = hotspots.pivot(
    index="lat",
    columns="lon",
    values="mean_co2_tonnes"
)

plt.figure(figsize=(12, 8))

plt.imshow(
    pivot_mean.sort_index(ascending=True),
    aspect="auto",
    origin="lower"
)

plt.colorbar(
    label="Mean daily CO₂ emissions (tonnes/day)"
)

plt.xlabel("Longitude grid index")
plt.ylabel("Latitude grid index")

plt.title(
    "Hyderabad Road-Traffic CO₂ Emission Hotspots\n"
    "CHETNA-Road 2021"
)

plt.tight_layout()

heatmap_file = (
    OUTPUT_DIR
    / "hyderabad_mean_co2_spatial_heatmap.png"
)

plt.savefig(
    heatmap_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 10. TRUE LAT/LON SCATTER MAP
# ---------------------------------------------------------

plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    hotspots["lon"],
    hotspots["lat"],
    c=hotspots["mean_co2_tonnes"],
    s=35,
    alpha=0.85
)

plt.colorbar(
    scatter,
    label="Mean daily CO₂ emissions (tonnes/day)"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title(
    "Hyderabad Road-Traffic CO₂ Emission Spatial Distribution\n"
    "CHETNA-Road 2021"
)

plt.grid(alpha=0.2)

plt.tight_layout()

scatter_file = (
    OUTPUT_DIR
    / "hyderabad_co2_spatial_scatter.png"
)

plt.savefig(
    scatter_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 11. LABEL TOP 10 HOTSPOTS
# ---------------------------------------------------------

plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    hotspots["lon"],
    hotspots["lat"],
    c=hotspots["mean_co2_tonnes"],
    s=35,
    alpha=0.75
)

plt.colorbar(
    scatter,
    label="Mean daily CO₂ emissions (tonnes/day)"
)

for _, row in top10.iterrows():

    plt.annotate(
        str(int(row["hotspot_rank"])),
        (
            row["lon"],
            row["lat"]
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9
    )


plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title(
    "Top 10 Hyderabad Road-Traffic CO₂ Hotspots\n"
    "CHETNA-Road 2021"
)

plt.grid(alpha=0.2)

plt.tight_layout()

label_file = (
    OUTPUT_DIR
    / "hyderabad_top10_hotspots_map.png"
)

plt.savefig(
    label_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 12. FINAL SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SPATIAL HOTSPOT ANALYSIS COMPLETED")
print("=" * 70)

print("\nSpatial cells analyzed:", len(hotspots))

print(
    "\nHighest mean-emission cell:"
)

print(
    hotspots.iloc[0].to_string()
)

print("\nFiles created:")

print(spatial_file)
print(top20_file)
print(top10_file)
print(threshold_file)
print(heatmap_file)
print(scatter_file)
print(label_file)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)