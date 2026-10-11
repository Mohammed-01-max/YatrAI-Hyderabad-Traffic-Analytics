from pathlib import Path
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "hyderabad"


def inspect_file(path: Path):
    print("=" * 80)
    print(f"FILE: {path.name}")
    print("=" * 80)

    try:
        with xr.open_dataset(path) as ds:

            print("\nDATASET:")
            print(ds)

            print("\n" + "=" * 80)
            print("DIMENSIONS")
            print("=" * 80)

            for name, size in ds.sizes.items():
                print(f"{name}: {size}")

            print("\n" + "=" * 80)
            print("COORDINATES")
            print("=" * 80)

            for name in ds.coords:
                values = ds[name].values

                print(f"\n{name}")
                print(f"  shape: {values.shape}")

                if values.size > 0:
                    print(f"  first: {values.flat[0]}")
                    print(f"  last : {values.flat[-1]}")

            print("\n" + "=" * 80)
            print("DATA VARIABLES")
            print("=" * 80)

            for name in ds.data_vars:
                var = ds[name]

                print(f"\n{name}")
                print(f"  dimensions : {var.dims}")
                print(f"  shape      : {var.shape}")
                print(f"  dtype      : {var.dtype}")
                print(f"  units      : {var.attrs.get('units')}")
                print(f"  long_name  : {var.attrs.get('long_name')}")
                print(f"  standard_name: {var.attrs.get('standard_name')}")

            print("\n" + "=" * 80)
            print("GLOBAL ATTRIBUTES")
            print("=" * 80)

            for key, value in ds.attrs.items():
                print(f"{key}: {value}")

    except Exception as exc:
        print("\nERROR WHILE READING FILE:")
        print(type(exc).__name__)
        print(exc)


def main():

    print("Project root:")
    print(PROJECT_ROOT)

    print("\nRaw Hyderabad directory:")
    print(RAW_DIR)

    files = sorted(RAW_DIR.glob("*.nc"))

    print("\nNetCDF files found:", len(files))

    if not files:
        print("\nNo .nc files found.")
        print("The Hyderabad CHETNA-Road data has not been downloaded yet.")
        return

    for file in files:
        inspect_file(file)


if __name__ == "__main__":
    main()