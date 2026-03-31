#!/usr/bin/env python3
"""Download ACAG surface NO2 NetCDF files and build an ADM2 panel.

The workflow is designed to survive interruptions:
- downloads are resumable via HTTP range requests (`.part` files)
- each continent-year is processed into its own parquet checkpoint
- reruns skip completed checkpoints and rebuild the final panel

Dataset source:
https://zenodo.org/records/5424752
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.errors import NotGeoreferencedWarning, RasterioIOError
from rasterio.transform import from_bounds
from rasterstats import zonal_stats
from shapely.geometry import box

try:
    from gid_utils import valid_gid_mask
except ModuleNotFoundError:
    from scripts.gid_utils import valid_gid_mask


BASE_URL = "https://zenodo.org/records/5424752/files"
CONTINENTS = (
    "africa",
    "asia",
    "europe",
    "northamerica",
    "oceania",
    "southamerica",
)
DEFAULT_YEARS = range(2005, 2020)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GADM_PATH = DATA / "gadm" / "gadm41_adm2.gpkg"
RAW_DIR = DATA / "acag_no2_raw"
PARTS_DIR = DATA / "acag_no2_parts"
OUTPUT_PATH = DATA / "no2_adm2_acag_panel.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an ADM2 NO2 panel from ACAG continent-year NetCDF files."
    )
    parser.add_argument("--start-year", type=int, default=min(DEFAULT_YEARS))
    parser.add_argument("--end-year", type=int, default=max(DEFAULT_YEARS))
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep downloaded .nc files after each continent-year is processed.",
    )
    parser.add_argument(
        "--force-process",
        action="store_true",
        help="Recompute existing continent-year parquet checkpoints.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request read timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=8,
        help="Max download retries before giving up.",
    )
    return parser.parse_args()


def year_range(start_year: int, end_year: int) -> list[int]:
    if start_year > end_year:
        raise ValueError("--start-year must be <= --end-year")
    return list(range(start_year, end_year + 1))


def filename(continent: str, year: int) -> str:
    return f"OMI_downscaled-inferred_surface_no2_{continent}_{year}.nc"


def download_url(continent: str, year: int) -> str:
    return f"{BASE_URL}/{filename(continent, year)}?download=1"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)


def load_adm2() -> gpd.GeoDataFrame:
    if not GADM_PATH.exists():
        raise FileNotFoundError(
            f"GADM ADM2 cache not found at {GADM_PATH}. "
            "Run the replication notebook first to build it."
        )
    adm2 = gpd.read_file(GADM_PATH, columns=["GID_0", "GID_1", "GID_2", "geometry"])
    adm2 = adm2.loc[valid_gid_mask(adm2["GID_2"])].copy()
    if adm2.crs is None:
        raise ValueError(f"{GADM_PATH} has no CRS; expected EPSG:4326.")
    if adm2.crs.to_epsg() != 4326:
        adm2 = adm2.to_crs(4326)
    return adm2


def download_with_resume(
    url: str,
    dest: Path,
    timeout: int,
    retries: int,
    chunk_size: int = 1024 * 1024,
) -> None:
    if dest.exists():
        return

    temp_path = dest.with_suffix(dest.suffix + ".part")
    base_delay = 2
    resume_from = temp_path.stat().st_size if temp_path.exists() else 0

    for attempt in range(1, retries + 1):
        headers = {}
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"

        try:
            with requests.get(
                url,
                stream=True,
                timeout=(15, timeout),
                headers=headers,
            ) as response:
                if response.status_code == 416 and temp_path.exists():
                    temp_path.rename(dest)
                    return

                if response.status_code not in (200, 206):
                    response.raise_for_status()

                if resume_from and response.status_code == 200:
                    temp_path.unlink(missing_ok=True)
                    resume_from = 0
                    continue

                mode = "ab" if resume_from and response.status_code == 206 else "wb"
                with temp_path.open(mode) as fh:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            fh.write(chunk)

            temp_path.rename(dest)
            return
        except (requests.RequestException, OSError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}") from exc
            sleep_for = min(60, base_delay ** attempt)
            print(
                f"    Download failed (attempt {attempt}/{retries}): {exc}. "
                f"Retrying in {sleep_for}s..."
            )
            time.sleep(sleep_for)
            resume_from = temp_path.stat().st_size if temp_path.exists() else 0


def choose_raster_source(nc_path: Path) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(nc_path) as src:
                subdatasets = list(src.subdatasets)
    except RasterioIOError:
        subdatasets = []

    if not subdatasets:
        return str(nc_path)

    def var_name(source: str) -> str:
        return source.rsplit(":", 1)[-1].lower()

    preferred = [s for s in subdatasets if "no2" in var_name(s)]
    if not preferred:
        preferred = subdatasets

    preferred.sort(
        key=lambda item: (
            0 if "surface" in var_name(item) else 1,
            len(var_name(item)),
        )
    )
    return preferred[0]


def read_vector_subdataset(nc_path: Path, var_name: str) -> np.ndarray:
    source = f"netcdf:{nc_path}:{var_name}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(source) as src:
            arr = src.read(1)
    return np.asarray(arr).reshape(-1)


def load_georeferenced_array(nc_path: Path) -> tuple[np.ndarray, object, float]:
    raster_source = choose_raster_source(nc_path)
    if not raster_source.endswith(":surface_no2_ppb"):
        raise RuntimeError(
            f"Unexpected NO2 subdataset for {nc_path.name}: {raster_source}. "
            "Expected surface_no2_ppb."
        )

    lon = read_vector_subdataset(nc_path, "LON_CENTER")
    lat = read_vector_subdataset(nc_path, "LAT_CENTER")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(raster_source) as src:
            data = src.read(1, masked=True)

    if data.shape == (len(lat), len(lon)):
        pass
    elif data.shape == (len(lon), len(lat)):
        data = data.T
    else:
        raise RuntimeError(
            f"Unexpected grid shape for {nc_path.name}: data={data.shape}, "
            f"lat={len(lat)}, lon={len(lon)}"
        )

    if lon[0] > lon[-1]:
        lon = lon[::-1]
        data = data[:, ::-1]

    if lat[0] < lat[-1]:
        lat = lat[::-1]
        data = data[::-1, :]

    lon_step = float(np.median(np.abs(np.diff(lon))))
    lat_step = float(np.median(np.abs(np.diff(lat))))
    if not np.isfinite(lon_step) or not np.isfinite(lat_step) or lon_step <= 0 or lat_step <= 0:
        raise RuntimeError(f"Invalid coordinate spacing for {nc_path.name}")

    left = float(lon[0] - lon_step / 2)
    right = float(lon[-1] + lon_step / 2)
    top = float(lat[0] + lat_step / 2)
    bottom = float(lat[-1] - lat_step / 2)
    transform = from_bounds(left, bottom, right, top, data.shape[1], data.shape[0])

    nodata = -9999.0
    filled = np.ma.filled(data.astype("float32"), nodata)
    return filled, transform, nodata


def compute_part(
    adm2: gpd.GeoDataFrame,
    continent: str,
    year: int,
    nc_path: Path,
    part_path: Path,
) -> None:
    data, transform, nodata = load_georeferenced_array(nc_path)
    bounds = rasterio.transform.array_bounds(data.shape[0], data.shape[1], transform)
    raster_box = box(bounds[0], bounds[1], bounds[2], bounds[3])
    subset = adm2.loc[adm2.intersects(raster_box), ["GID_0", "GID_1", "GID_2", "geometry"]].copy()
    if subset.empty:
        raise RuntimeError(f"No ADM2 geometries intersect raster extent for {nc_path.name}")

    stats = zonal_stats(
        subset.geometry,
        data,
        affine=transform,
        nodata=nodata,
        stats=["mean"],
    )
    frame = pd.DataFrame(
        {
            "GID_0": subset["GID_0"].values,
            "GID_1": subset["GID_1"].values,
            "GID_2": subset["GID_2"].values,
            "continent": continent,
            "year": year,
            "no2_mean": [row["mean"] for row in stats],
        }
    )

    frame = frame.dropna(subset=["no2_mean"])
    frame = frame[frame["no2_mean"] >= 0]
    frame.to_parquet(part_path, index=False)


def combine_parts(years: set[int]) -> None:
    part_files = sorted(PARTS_DIR.glob("acag_no2_*.parquet"))
    if not part_files:
        raise RuntimeError(f"No continent-year parquet files found in {PARTS_DIR}")

    frames = [pd.read_parquet(path) for path in part_files]
    panel = pd.concat(frames, ignore_index=True)
    panel = panel[panel["year"].isin(years)].copy()
    panel = panel.loc[valid_gid_mask(panel["GID_2"])].copy()
    if panel.empty:
        raise RuntimeError("No rows found for the requested year range.")

    panel = (
        panel.groupby(["GID_0", "GID_1", "GID_2", "year"], as_index=False)["no2_mean"]
        .mean()
        .sort_values(["GID_2", "year"])
    )
    panel.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nSaved ACAG NO2 panel to {OUTPUT_PATH}")
    print(
        f"Panel: {len(panel):,} obs | "
        f"{panel['GID_2'].nunique():,} regions | "
        f"years: {sorted(panel['year'].unique())}"
    )


def main() -> int:
    args = parse_args()
    years = year_range(args.start_year, args.end_year)

    ensure_dirs()
    adm2 = load_adm2()

    print(f"Loaded {len(adm2):,} ADM2 regions from {GADM_PATH}")
    print(f"Years: {years[0]}-{years[-1]}")
    print(f"Raw download dir: {RAW_DIR}")
    print(f"Checkpoint dir:   {PARTS_DIR}")

    for year in years:
        for continent in CONTINENTS:
            nc_name = filename(continent, year)
            nc_path = RAW_DIR / nc_name
            part_path = PARTS_DIR / f"acag_no2_{continent}_{year}.parquet"

            if part_path.exists() and not args.force_process:
                print(f"[skip] {continent} {year}: checkpoint already exists")
                continue

            print(f"[run]  {continent} {year}: downloading {nc_name}")
            download_with_resume(
                url=download_url(continent, year),
                dest=nc_path,
                timeout=args.timeout,
                retries=args.retries,
            )

            size_gb = nc_path.stat().st_size / 1e9
            print(f"       downloaded ({size_gb:.2f} GB), computing zonal stats...")
            compute_part(
                adm2=adm2,
                continent=continent,
                year=year,
                nc_path=nc_path,
                part_path=part_path,
            )
            print(f"       saved checkpoint -> {part_path.name}")

            if not args.keep_raw:
                nc_path.unlink(missing_ok=True)
                part_file_mb = part_path.stat().st_size / 1e6
                print(f"       removed raw .nc, checkpoint size {part_file_mb:.1f} MB")

    combine_parts(set(years))
    return 0


if __name__ == "__main__":
    sys.exit(main())
