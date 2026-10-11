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
print("HYDERABAD SPATIAL HOTSPOT VALIDATION")
print("=" * 70)


df = pd.read_csv(GRID_FILE)

df["date"] = pd.to_datetime(df["date"])

total_days = df["date"].nunique()

print("\nTotal days:", total_days)
print("Total spatial cells:", df[["lat", "lon"]].drop_duplicates().shape[0])


# ---------------------------------------------------------
# SPATIAL STATISTICS
# ---------------------------------------------------------

spatial = (
    df.groupby(["lat", "lon"], as_index=False)
    .agg(
        mean_co2_tonnes=("co2_tonnes", "mean"),
        total_co2_tonnes=("co2_tonnes", "sum"),
        max_daily_co2_tonnes=("co2_tonnes", "max"),
        active_days=("co2_tonnes", "count"),
    )
)

spatial["coverage_percent"] = (
    spatial["active_days"] / total_days * 100
)


# ---------------------------------------------------------
# COVERAGE FILTER
# ---------------------------------------------------------

MIN_COVERAGE = 90.0

eligible = spatial[
    spatial["coverage_percent"] >= MIN_COVERAGE
].copy()

excluded = spatial[
    spatial["coverage_percent"] < MIN_COVERAGE
].copy()


print("\n" + "=" * 70)
print("COVERAGE VALIDATION")
print("=" * 70)

print(
    f"Minimum required coverage: {MIN_COVERAGE:.1f}%"
)

print(
    "Minimum required active days:",
    int(np.ceil(total_days * MIN_COVERAGE / 100))
)

print(
    "Eligible cells:",
    len(eligible)
)

print(
    "Excluded cells:",
    len(excluded)
)

print(
    "Eligible percentage:",
    round(len(eligible) / len(spatial) * 100, 2),
    "%"
)


# ---------------------------------------------------------
# P95 THRESHOLD
# ---------------------------------------------------------

p95 = eligible["mean_co2_tonnes"].quantile(0.95)

eligible["p95_hotspot"] = (
    eligible["mean_co2_tonnes"] >= p95
)


hotspots = eligible[
    eligible["p95_hotspot"]
].copy()

hotspots = hotspots.sort_values(
    "mean_co2_tonnes",
    ascending=False
).reset_index(drop=True)

hotspots["hotspot_rank"] = (
    hotspots.index + 1
)


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

eligible_file = (
    OUTPUT_DIR
    / "hyderabad_eligible_spatial_cells.csv"
)

hotspot_file = (
    OUTPUT_DIR
    / "hyderabad_validated_p95_hotspots.csv"
)

excluded_file = (
    OUTPUT_DIR
    / "hyderabad_low_coverage_cells.csv"
)

eligible.to_csv(
    eligible_file,
    index=False
)

hotspots.to_csv(
    hotspot_file,
    index=False
)

excluded.to_csv(
    excluded_file,
    index=False
)


# ---------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATED P95 HOTSPOTS")
print("=" * 70)

print(
    f"P95 threshold: {p95:.6f} tonnes/day"
)

print(
    "Number of P95 hotspot cells:",
    len(hotspots)
)

print("\nTop 20 validated hotspots:")

print(
    hotspots.head(20).to_string(index=False)
)


# ---------------------------------------------------------
# TOP HOTSPOT
# ---------------------------------------------------------

if len(hotspots) > 0:

    top = hotspots.iloc[0]

    print("\n" + "=" * 70)
    print("HIGHEST VALIDATED HOTSPOT")
    print("=" * 70)

    print(
        f"Latitude          : {top['lat']:.6f}"
    )

    print(
        f"Longitude         : {top['lon']:.6f}"
    )

    print(
        f"Mean CO2          : {top['mean_co2_tonnes']:.6f} tonnes/day"
    )

    print(
        f"Total CO2         : {top['total_co2_tonnes']:.6f} tonnes"
    )

    print(
        f"Maximum daily CO2 : {top['max_daily_co2_tonnes']:.6f} tonnes/day"
    )

    print(
        f"Active days       : {int(top['active_days'])}"
    )

    print(
        f"Coverage          : {top['coverage_percent']:.2f}%"
    )


# ---------------------------------------------------------
# MAP
# ---------------------------------------------------------

plt.figure(figsize=(12, 8))

background = plt.scatter(
    eligible["lon"],
    eligible["lat"],
    c=eligible["mean_co2_tonnes"],
    s=25,
    alpha=0.35
)

plt.colorbar(
    background,
    label="Mean daily CO₂ emissions (tonnes/day)"
)

plt.scatter(
    hotspots["lon"],
    hotspots["lat"],
    s=55,
    marker="x",
    label="P95 hotspot"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title(
    "Validated Hyderabad Road-Traffic CO₂ Hotspots\n"
    "P95 threshold, minimum 90% temporal coverage"
)

plt.legend()

plt.grid(alpha=0.2)

plt.tight_layout()

map_file = (
    OUTPUT_DIR
    / "hyderabad_validated_p95_hotspot_map.png"
)

plt.savefig(
    map_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\n" + "=" * 70)
print("VALIDATION COMPLETED")
print("=" * 70)

print("\nSaved:")
print(eligible_file)
print(hotspot_file)
print(excluded_file)
print(map_file)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)