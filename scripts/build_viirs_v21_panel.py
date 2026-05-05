"""
Build a standalone ADM2 VIIRS nightlights cache from annual VNL V2 rasters.

Expected inputs:
- data/nightlights/viirs_v21_annual/nightlights.average_viirs.v21_m_500m_s_YYYY0101_YYYY1231_go_epsg4326_v20250904.tif
- data/gadm/gadm41_adm2.gpkg

Outputs:
- per-year checkpoints in data/tmp_viirs_v21_parts/
- combined panel in data/nightlights_adm2_viirs_v21_2012_2024_panel.parquet
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats

try:
    from gid_utils import valid_gid_mask
except ModuleNotFoundError:
    from scripts.gid_utils import valid_gid_mask


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
GADM_PATH = DATA / "gadm" / "gadm41_adm2.gpkg"
GADM_SHP_PATH = DATA / "gadm" / "gadm41_adm2_shp" / "gadm41_adm2_shp.shp"
RAW_DIR = DATA / "nightlights" / "viirs_v21_annual"
PARTS_DIR = DATA / "tmp_viirs_v21_parts"
OUT_PATH = DATA / "nightlights_adm2_viirs_v21_2012_2024_panel.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def raster_path(year: int) -> Path:
    return RAW_DIR / (
        f"nightlights.average_viirs.v21_m_500m_s_{year}0101_{year}1231_"
        "go_epsg4326_v20250904.tif"
    )


def load_adm2() -> gpd.GeoDataFrame:
    if GADM_SHP_PATH.exists():
        adm2 = gpd.read_file(GADM_SHP_PATH)
    elif GADM_PATH.exists():
        adm2 = gpd.read_file(GADM_PATH, layer="gadm41_adm2")
    else:
        raise FileNotFoundError(
            f"GADM ADM2 geometry not found at {GADM_SHP_PATH} or {GADM_PATH}"
        )
    adm2 = adm2.loc[valid_gid_mask(adm2["GID_2"])].copy()
    return adm2[["GID_0", "GID_1", "GID_2", "geometry"]]


def ensure_inputs(start_year: int, end_year: int) -> None:
    missing = [str(raster_path(year)) for year in range(start_year, end_year + 1) if not raster_path(year).exists()]
    if missing:
        raise FileNotFoundError("Missing annual VIIRS rasters:\n" + "\n".join(missing))
    PARTS_DIR.mkdir(parents=True, exist_ok=True)


def build_year(task: tuple[int, Path]) -> str:
    year, part_path = task
    adm2 = load_adm2()
    stats = zonal_stats(adm2.geometry, str(raster_path(year)), stats=["mean"])
    out = adm2[["GID_0", "GID_1", "GID_2"]].copy()
    out["year"] = year
    out["ntl_mean"] = [row["mean"] for row in stats]
    out = out.dropna(subset=["ntl_mean"])
    out.to_parquet(part_path, index=False)
    return f"{year}: {len(out):,} rows"


def combine_parts(start_year: int, end_year: int) -> pd.DataFrame:
    part_files = [PARTS_DIR / f"viirs_v21_{year}.parquet" for year in range(start_year, end_year + 1)]
    missing = [str(path) for path in part_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint files:\n" + "\n".join(missing))
    panel = pd.concat([pd.read_parquet(path) for path in part_files], ignore_index=True)
    panel = panel.sort_values(["GID_2", "year"]).reset_index(drop=True)
    panel.to_parquet(OUT_PATH, index=False)
    return panel


def main() -> int:
    args = parse_args()
    ensure_inputs(args.start_year, args.end_year)

    adm2 = load_adm2()

    tasks: list[tuple[int, Path]] = []
    for year in range(args.start_year, args.end_year + 1):
        part_path = PARTS_DIR / f"viirs_v21_{year}.parquet"
        if part_path.exists() and not args.force:
            print(f"[skip] {year}: checkpoint exists")
            continue
        tasks.append((year, part_path))

    print(f"ADM2 regions: {len(adm2):,}")
    print(f"Years: {args.start_year}-{args.end_year}")
    print(f"Processes: {args.processes}")
    print(f"Parts dir: {PARTS_DIR}")

    if tasks:
        with Pool(processes=args.processes) as pool:
            for msg in pool.imap_unordered(build_year, tasks):
                print(f"[done] {msg}")
    else:
        print("No new yearly parts needed.")

    panel = combine_parts(args.start_year, args.end_year)
    print(
        f"Saved VIIRS panel to {OUT_PATH}\n"
        f"Rows: {len(panel):,} | Regions: {panel['GID_2'].nunique():,} | "
        f"Years: {panel['year'].min()}-{panel['year'].max()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
