from pathlib import Path
import requests
import xarray as xr
import tempfile
import os

FILE_IDS = [
    52545962,
    52545953,
    52545956,
    52545959,
    52545968,
    52545965,
    52545971,
    52545974,
    52545980,
    52545977,
    52545986,
    52545983,
    52545989,
    52545995,
    52545992,
]

BASE_URL = "https://ndownloader.figshare.com/files/"

def inspect_file(file_id):
    print("=" * 70)
    print(f"FILE ID: {file_id}")

    temp_path = Path(tempfile.gettempdir()) / f"chetna_{file_id}.nc"

    try:
        print("Downloading...")

        response = requests.get(
            BASE_URL + str(file_id),
            timeout=180
        )
        response.raise_for_status()

        temp_path.write_bytes(response.content)

        with xr.open_dataset(temp_path) as ds:
            city = ds.attrs.get("city", "UNKNOWN")
            state = ds.attrs.get("state", "UNKNOWN")
            year = ds.attrs.get("year", "UNKNOWN")

            print(f"CITY : {city}")
            print(f"STATE: {state}")
            print(f"YEAR : {year}")

            if "lat" in ds.coords:
                print(
                    f"LAT  : {float(ds.lat.min()):.4f} "
                    f"to {float(ds.lat.max()):.4f}"
                )

            if "lon" in ds.coords:
                print(
                    f"LON  : {float(ds.lon.min()):.4f} "
                    f"to {float(ds.lon.max()):.4f}"
                )

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")

    finally:
        if temp_path.exists():
            temp_path.unlink()

def main():
    print("CHETNA-Road CO2 city identification")
    print("=" * 70)

    for file_id in FILE_IDS:
        inspect_file(file_id)

if __name__ == "__main__":
    main()