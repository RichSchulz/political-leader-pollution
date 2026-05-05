"""Build the canonical GADM 4.1-aligned PLAD file.

PLAD codes its gid_2 against GADM 3.6, but every panel in this project is
built on GADM 4.1. This produces silent ~7% match-rate loss and ~30% of
leaders mis-coded by version drift. To fix this, we spatial-join PLAD's
lat/lon birthplace coordinates onto GADM 4.1 ADM2 polygons and overwrite
`gid_2` with the result. The original code is preserved as `gid_2_orig`.

Output: data/plad_april2024_gadm41.parquet — drop-in replacement for
PLAD_April_2024.dta. All other columns and dtypes are preserved as
returned by pd.read_stata.

Run once. Subsequent notebooks / scripts should load the parquet instead
of the .dta file.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GADM_PATH = DATA / "gadm" / "gadm41_adm2.gpkg"
PLAD_DTA = DATA / "political leaders" / "PLAD_April_2024.dta"
OUT = DATA / "political leaders" / "PLAD_April_2024_gadm41.parquet"

sys.path.append(str(ROOT / "scripts"))
from gid_utils import valid_gid_mask  # noqa: E402


def main():
    print(f">> Reading PLAD .dta ({PLAD_DTA.name})...")
    plad = pd.read_stata(PLAD_DTA)
    print(f"   {len(plad)} rows, {len(plad.columns)} cols")

    print(f">> Reading GADM 4.1 ADM2 ({GADM_PATH.name})...")
    adm2 = gpd.read_file(GADM_PATH)
    adm2 = adm2.loc[valid_gid_mask(adm2["GID_2"])][["GID_2", "geometry"]].to_crs("EPSG:4326").copy()
    print(f"   {len(adm2):,} ADM2 polygons")

    plad["latitude"]  = pd.to_numeric(plad["latitude"],  errors="coerce")
    plad["longitude"] = pd.to_numeric(plad["longitude"], errors="coerce")
    has_xy = plad["latitude"].notna() & plad["longitude"].notna()
    print(f">> {has_xy.sum()} / {len(plad)} leaders have lat/lon and can be re-joined")

    pts = gpd.GeoDataFrame(
        plad.loc[has_xy].copy(),
        geometry=gpd.points_from_xy(plad.loc[has_xy, "longitude"],
                                    plad.loc[has_xy, "latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, adm2, how="left", predicate="within")
    missing = joined["GID_2"].isna()
    if missing.any():
        print(f">> {missing.sum()} points outside any polygon — snapping to nearest")
        miss_pts = pts.loc[missing.values].to_crs("EPSG:3857")
        adm2_proj = adm2.to_crs("EPSG:3857")
        nearest = gpd.sjoin_nearest(miss_pts, adm2_proj, how="left",
                                    distance_col="snap_m")
        joined.loc[missing.values, "GID_2"] = nearest["GID_2"].values

    new_gid2 = pd.Series(pd.NA, index=plad.index, dtype="object")
    new_gid2.loc[has_xy] = joined["GID_2"].values

    # Preserve the original code; overwrite gid_2 with the spatial-join result.
    plad["gid_2_orig"] = plad["gid_2"].astype("object")
    # Keep PLAD's "." sentinel for missing in the new column (matches old behavior
    # when valid_gid_mask gets called downstream).
    plad["gid_2"] = new_gid2.where(new_gid2.notna(), ".")

    # Diagnostic
    orig_valid = plad["gid_2_orig"].apply(lambda s: isinstance(s, str) and s.strip() != ".")
    new_valid  = plad["gid_2"].apply(lambda s: isinstance(s, str) and s.strip() != ".")
    same = orig_valid & new_valid & (plad["gid_2"] == plad["gid_2_orig"])
    diff = orig_valid & new_valid & (plad["gid_2"] != plad["gid_2_orig"])
    only_new = (~orig_valid) & new_valid
    print(f">> orig valid: {int(orig_valid.sum())}  |  new valid: {int(new_valid.sum())}")
    print(f"   identical:    {int(same.sum())}")
    print(f"   reassigned:   {int(diff.sum())}")
    print(f"   recovered:    {int(only_new.sum())}  (orig blank, lat/lon present)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plad.to_parquet(OUT, index=False)
    print(f">> Wrote {OUT}")


if __name__ == "__main__":
    main()
