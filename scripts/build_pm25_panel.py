"""
build_pm25_panel.py
====================
Build an ADM2-year panel of mean surface PM2.5 concentrations from the
ACAG SatPM V5.GL.06 annual NetCDF files (van Donkelaar et al., 2021).

Source:  https://www.satpm.org/v5-gl-06
Files:   https://wustl.app.box.com/v/ACAG-V5GL06-GWRPM25/folder/349055735295
Format:  V5GL06.HybridPM25.Global.<YYYY>01-<YYYY>12.nc
Grid:    0.01° × 0.01°, WGS84, lat ∈ [-54.995, 67.995], lon ∈ [-179.995, 179.995]
Units:   µg/m³

Prerequisite
------------
Run replication.ipynb first so that data/gadm/gadm41_adm2.gpkg exists.

Usage
-----
1.  Download the annual .nc files for 1998–2024 from the Box link above
    and place them in data/pm25/ (or set PM25_DIR in this script).

2.  Run:
        python scripts/build_pm25_panel.py

3.  The output parquet is loaded in green_favoritism_pm25.ipynb via:
        PM25_PANEL_CACHE = DATA / "pm25_adm2_acag_panel.parquet"

Dependencies
------------
    pip install numpy pandas geopandas netCDF4 rasterstats pyproj tqdm
    (rasterio is used indirectly by rasterstats)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats
from rasterio.transform import from_bounds
import netCDF4 as nc
from tqdm import tqdm

try:
    from gid_utils import valid_gid_mask
except ModuleNotFoundError:
    from scripts.gid_utils import valid_gid_mask

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent   # project root
DATA      = ROOT / "data"
PM25_DIR  = DATA / "pm25"                            # put your .nc files here
GADM_DIR  = DATA / "gadm"
ADM2_CACHE = GADM_DIR / "gadm41_adm2.gpkg"
OUT_PATH  = DATA / "pm25_adm2_acag_panel.parquet"

ANALYSIS_YEARS = range(1998, 2025)   # 1998–2024 inclusive


# ── Expected filename pattern ─────────────────────────────────────────────────
# V5GL06.HybridPM25.Global.200501-200512.nc  (for year 2005)
def pm25_path(year: int) -> Path:
    return PM25_DIR / f"V5GL06.HybridPM25.Global.{year}01-{year}12.nc"


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_pm25_array(nc_path: Path) -> tuple[np.ndarray, object, float]:
    """
    Read the PM2.5 variable from a V5.GL.06 NetCDF file and return a
    georeferenced array suitable for rasterstats.

    The V5.GL.06 files store data as:
        float32 PM25(lat, lon)   or   float32 GWRPM25(lat, lon)
    with lat descending (north→south) and lon ascending (west→east).
    """
    with nc.Dataset(nc_path, "r") as ds:
        # ── Find the PM2.5 variable ──────────────────────────────────────────
        # The variable is named differently across minor releases; try in order.
        pm25_varnames = ["PM25", "GWRPM25", "PM2.5", "pm25"]
        var_name = next((v for v in pm25_varnames if v in ds.variables), None)
        if var_name is None:
            raise KeyError(
                f"Cannot find PM2.5 variable in {nc_path.name}. "
                f"Available: {list(ds.variables.keys())}"
            )

        data   = ds.variables[var_name][:]        # (lat, lon) — float32 masked array
        lats   = ds.variables["lat"][:]           # 1-D ascending or descending
        lons   = ds.variables["lon"][:]           # 1-D ascending

    # Handle masked / fill values — clamp unrealistic values to nodata
        if hasattr(data, "fill_value"):
            fill = float(data.fill_value)
        else:
            fill = -9999.0
        data = np.ma.filled(data.astype(np.float32), fill_value=-9999.0)
        data[data > 1000] = -9999.0   # sentinel values like 1e20 → nodata
        data[data < 0]    = -9999.0   # negative fill values → nodata
        fill = -9999.0

    # ── Ensure north-up orientation (lat must be descending for north-up) ──
    if lats[0] < lats[-1]:          # ascending → need to flip
        data = data[::-1, :]
        lats = lats[::-1]

    nrows, ncols = data.shape
    res_lat = abs(float(lats[0] - lats[1]))  # cell size ≈ 0.01°
    res_lon = abs(float(lons[1] - lons[0]))

    west  = float(lons[0])  - res_lon / 2
    east  = float(lons[-1]) + res_lon / 2
    north = float(lats[0])  + res_lat / 2
    south = float(lats[-1]) - res_lat / 2

    transform = from_bounds(west, south, east, north, ncols, nrows)

    return data, transform, fill


def compute_zonal_mean(
    adm2: gpd.GeoDataFrame,
    data: np.ndarray,
    transform,
    nodata: float,
) -> list:
    """Return list of mean PM2.5 values (one per ADM2 polygon)."""
    stats = zonal_stats(
        adm2.geometry,
        data,
        affine=transform,
        stats   = ["mean"],
        nodata  = nodata,
        all_touched = False,   # only pixels whose centroid falls in polygon
    )
    return [s["mean"] for s in stats]


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:

    # 1. Load ADM2 boundaries ─────────────────────────────────────────────────
    assert ADM2_CACHE.exists(), (
        f"ADM2 cache not found at {ADM2_CACHE}.\n"
        "Run replication.ipynb first to build it."
    )
    print(f"Loading ADM2 boundaries from {ADM2_CACHE} …")
    adm2 = gpd.read_file(ADM2_CACHE)
    adm2 = adm2.loc[valid_gid_mask(adm2["GID_2"])].copy()
    print(f"  {len(adm2):,} ADM2 regions loaded.")

    # 2. Check which years are already cached ─────────────────────────────────
    if OUT_PATH.exists():
        cached_years = set(
            pd.read_parquet(OUT_PATH, columns=["year"])["year"].unique()
        )
        missing_years = [y for y in ANALYSIS_YEARS if y not in cached_years]
        if not missing_years:
            print(f"Cache already covers all years {list(ANALYSIS_YEARS)}. Nothing to do.")
            return
        print(f"Cache exists but missing years: {missing_years}. Appending …")
        existing_frames = [pd.read_parquet(OUT_PATH)]
    else:
        missing_years  = list(ANALYSIS_YEARS)
        existing_frames = []

    # 3. Verify all required .nc files are present ────────────────────────────
    missing_nc = [y for y in missing_years if not pm25_path(y).exists()]
    if missing_nc:
        print("\n⚠  The following annual NetCDF files were not found in", PM25_DIR)
        for y in missing_nc:
            print(f"     {pm25_path(y).name}")
        print(
            "\nDownload them from:\n"
            "  https://wustl.app.box.com/v/ACAG-V5GL06-GWRPM25/folder/349055735295\n"
            "and place them in:", PM25_DIR
        )
        raise FileNotFoundError(
            f"{len(missing_nc)} NetCDF file(s) missing. See list above."
        )

    # 4. Process each year ────────────────────────────────────────────────────
    new_frames = []

    for year in tqdm(missing_years, desc="Processing PM2.5 years"):
        nc_path = pm25_path(year)

        data, transform, nodata = load_pm25_array(nc_path)
        means = compute_zonal_mean(adm2, data, transform, nodata)

        df_year = pd.DataFrame({
            "GID_2":    adm2["GID_2"].values,
            "GID_1":    adm2["GID_1"].values,
            "GID_0":    adm2["GID_0"].values,
            "year":     year,
            "pm25_mean": means,
        })

        n_valid = df_year["pm25_mean"].notna().sum()
        print(f"  {year}: {n_valid:,} non-null ADM2 values")
        new_frames.append(df_year)

    # 5. Combine, clean, and save ──────────────────────────────────────────────
    all_frames = existing_frames + new_frames
    panel = pd.concat(all_frames, ignore_index=True)

    # Basic sanity filters
    panel = panel.loc[valid_gid_mask(panel["GID_2"])].copy()
    panel = panel[panel["year"].isin(ANALYSIS_YEARS)]
    panel = panel.dropna(subset=["pm25_mean"])
    panel = panel[panel["pm25_mean"] >= 0]        # remove negative fill values
    panel = panel[panel["pm25_mean"] < 1000]      # remove sentinel values (realistic max ~999 µg/m³)

    panel = panel.sort_values(["GID_2", "year"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved PM2.5 panel → {OUT_PATH}")
    print(
        f"  {len(panel):,} obs | "
        f"{panel['GID_2'].nunique():,} regions | "
        f"years: {sorted(panel['year'].unique())}"
    )
    print(
        f"  pm25_mean (µg/m³): "
        f"mean={panel['pm25_mean'].mean():.2f}, "
        f"median={panel['pm25_mean'].median():.2f}, "
        f"max={panel['pm25_mean'].max():.1f}"
    )


if __name__ == "__main__":
    main()
