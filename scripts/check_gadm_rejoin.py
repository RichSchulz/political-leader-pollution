"""Spatial re-join of PLAD birthplaces onto GADM 4.1 ADM2 polygons.

PLAD codes its gid_2 field against GADM 3.6 (per its codebook), but every
other panel in this project (DMSP, harmonized DMSP-VIIRS, standalone VIIRS,
ACAG NO2, ACAG PM2.5) is built on GADM 4.1. The replication notebook joins
PLAD to nightlights via gid_2 string equality, which silently drops 41/579
PLAD codes (~7%) due to GADM version drift.

This script bypasses that mismatch by spatial-joining PLAD's lat/lon birth
coordinates onto GADM 4.1 polygons directly, producing a fresh GID_2 for
each leader. We then re-run the main lag-0 spec (DMSP 1992-2013) and the
DMSP 5-year bin sequence under both the original and re-joined assignments
and report the comparison.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GADM_PATH = DATA / "gadm" / "gadm41_adm2.gpkg"
PLAD_PATH = DATA / "political leaders" / "PLAD_April_2024.dta"
PANEL_CACHE = DATA / "nightlights_adm2_panel.parquet"

sys.path.append(str(ROOT / "scripts"))
from gid_utils import valid_gid_mask  # noqa: E402


def load_plad():
    plad = pd.read_stata(PLAD_PATH)
    plad = plad[plad["foreign_leader"] == "0"].copy()
    plad = plad[plad["archigos_id"].str.strip() != "."].copy()
    plad["startyear"] = plad["startyear"].astype(int)
    plad["endyear"] = plad["endyear"].astype(int)
    plad = plad.sort_values(["gid_0", "startyear"]).reset_index(drop=True)
    return plad


def clip_overlapping_spells(plad):
    p = plad.copy().reset_index(drop=True)
    for _, group in p.groupby("gid_0"):
        idxs = group.index.tolist()
        for i in range(len(idxs) - 1):
            curr, nxt = idxs[i], idxs[i + 1]
            if p.loc[curr, "endyear"] >= p.loc[nxt, "startyear"]:
                p.loc[curr, "endyear"] = p.loc[nxt, "startyear"] - 1
    return p


def spatial_rejoin(plad, adm2):
    """Return PLAD with a new column `gid_2_sj` from a point-in-polygon join."""
    p = plad.copy()
    p["latitude"] = pd.to_numeric(p["latitude"], errors="coerce")
    p["longitude"] = pd.to_numeric(p["longitude"], errors="coerce")
    has_xy = p["latitude"].notna() & p["longitude"].notna()
    pts = gpd.GeoDataFrame(
        p.loc[has_xy].copy(),
        geometry=gpd.points_from_xy(p.loc[has_xy, "longitude"], p.loc[has_xy, "latitude"]),
        crs="EPSG:4326",
    )
    adm2_small = adm2[["GID_2", "geometry"]].to_crs("EPSG:4326")
    joined = gpd.sjoin(pts, adm2_small, how="left", predicate="within")
    # If a point lies on a boundary or just outside (coastal coords), fall back
    # to nearest polygon. Project to a metric CRS for `sjoin_nearest`.
    missing = joined["GID_2"].isna()
    if missing.any():
        miss_pts = pts.loc[missing.values].to_crs("EPSG:3857")
        adm2_proj = adm2_small.to_crs("EPSG:3857")
        nearest = gpd.sjoin_nearest(miss_pts, adm2_proj, how="left",
                                    distance_col="snap_m")
        joined.loc[missing.values, "GID_2"] = nearest["GID_2"].values
    out = p.copy()
    out["gid_2_sj"] = pd.NA
    out.loc[has_xy, "gid_2_sj"] = joined["GID_2"].values
    return out


def build_leader_years(plad_fixed, gid_col, year_lo, year_hi):
    rows = []
    for idx, row in plad_fixed.iterrows():
        gid = row[gid_col]
        if pd.isna(gid) or str(gid).strip() == "." or str(gid).strip() == "":
            continue
        for y in range(max(int(row["startyear"]), year_lo),
                       min(int(row["endyear"]), year_hi) + 1):
            rows.append({"GID_2": gid, "GID_0": row["gid_0"],
                         "year": y, "spell_id": idx})
    ly = pd.DataFrame(rows)
    if ly.empty:
        return ly
    ly = ly.loc[valid_gid_mask(ly["GID_2"])].copy()
    ly = ly.drop_duplicates(subset=["GID_2", "year"])
    ly["birth_region_leader"] = 1
    return ly


def fit_lag0(panel):
    p = panel.copy()
    if not isinstance(p.index, pd.MultiIndex):
        p = p.set_index(["GID_2", "year"])
    model = PanelOLS.from_formula(
        "ln_ntl ~ birth_region_leader + EntityEffects",
        data=p, other_effects=p["country_year"], drop_absorbed=True,
    )
    res = model.fit(cov_type="clustered", clusters=p["spell_id"])
    return {
        "coef": float(res.params["birth_region_leader"]),
        "se":   float(res.std_errors["birth_region_leader"]),
        "p":    float(res.pvalues["birth_region_leader"]),
        "n":    int(res.nobs),
        "treated": int(p["birth_region_leader"].sum()),
    }


def assemble_panel(ntl_panel, plad_fixed, gid_col, year_lo, year_hi):
    ly = build_leader_years(plad_fixed, gid_col, year_lo, year_hi)
    spell_map = ly[["GID_0", "year", "spell_id"]].drop_duplicates(subset=["GID_0", "year"])

    p = ntl_panel.loc[(ntl_panel["year"] >= year_lo) & (ntl_panel["year"] <= year_hi)].copy()
    p = p.merge(ly[["GID_2", "year", "birth_region_leader"]],
                on=["GID_2", "year"], how="left")
    p["birth_region_leader"] = p["birth_region_leader"].fillna(0).astype(int)
    p = p.merge(spell_map, on=["GID_0", "year"], how="left")
    p["spell_id"] = p["spell_id"].fillna(
        p["GID_0"] + "_" + p["year"].astype(str) + "_noleader"
    ).astype(str)
    p["ln_ntl"] = np.log(p["ntl_mean"] + 0.01)
    p["country_year"] = p["GID_0"] + "_" + p["year"].astype(str)
    p = p.dropna(subset=["ntl_mean"])
    return p


def main():
    print(">> Loading GADM 4.1 ADM2...")
    adm2 = gpd.read_file(GADM_PATH)
    adm2 = adm2.loc[valid_gid_mask(adm2["GID_2"])].copy()
    print(f"   {len(adm2):,} ADM2 polygons")

    print(">> Loading PLAD...")
    plad = load_plad()
    print(f"   {len(plad)} domestic leader rows")

    print(">> Spatial-joining PLAD lat/lon onto GADM 4.1...")
    plad_sj = spatial_rejoin(plad, adm2)
    have_orig = plad_sj["gid_2"].apply(lambda s: isinstance(s, str) and s.strip() != ".")
    have_sj = plad_sj["gid_2_sj"].notna()
    same = have_orig & have_sj & (plad_sj["gid_2"] == plad_sj["gid_2_sj"])
    diff = have_orig & have_sj & (plad_sj["gid_2"] != plad_sj["gid_2_sj"])
    only_sj = (~have_orig) & have_sj
    only_orig = have_orig & (~have_sj)
    print(f"   leaders with orig gid_2:                {have_orig.sum()}")
    print(f"   leaders with spatial-join gid_2:        {have_sj.sum()}")
    print(f"   identical:                              {same.sum()}")
    print(f"   reassigned (different GID_2):           {diff.sum()}")
    print(f"   recovered by spatial join (orig blank): {only_sj.sum()}")
    print(f"   spatial join failed (no lat/lon):       {only_orig.sum()}")

    print("\n   Examples of reassignments (first 15):")
    cols = ["leader", "country", "gid_2", "gid_2_sj"]
    print(plad_sj.loc[diff, cols].head(15).to_string(index=False))

    print("\n>> Loading cached nightlights panel (DMSP 1992-2013)...")
    ntl_panel = pd.read_parquet(PANEL_CACHE)
    ntl_panel = ntl_panel.loc[valid_gid_mask(ntl_panel["GID_2"])].copy()

    plad_fixed = clip_overlapping_spells(plad_sj)

    print("\n>> Fitting main spec (DMSP 1992-2013, lag 0):\n")
    print(f"{'spec':30s}  {'coef':>8s}  {'se':>8s}  {'p':>6s}  {'N':>10s}  {'treated':>7s}")
    rows = []
    for label, gid_col in [("Original (gid_2 from PLAD)", "gid_2"),
                           ("Spatial re-join (gid_2_sj)", "gid_2_sj")]:
        p = assemble_panel(ntl_panel, plad_fixed, gid_col, 1992, 2013)
        r = fit_lag0(p)
        r["spec"] = label
        rows.append(r)
        print(f"{label:30s}  {r['coef']:8.4f}  {r['se']:8.4f}  {r['p']:6.3f}  "
              f"{r['n']:10,}  {r['treated']:7d}")

    print("\n>> 5-year bin comparison (DMSP):\n")
    bins = [(1992, 1996), (1997, 2001), (2002, 2006), (2007, 2011), (2012, 2013)]
    print(f"{'bin':>12s}  {'orig coef':>10s}  {'orig p':>7s}  "
          f"{'sj coef':>10s}  {'sj p':>7s}  {'orig N':>8s}  {'sj N':>8s}")
    for lo, hi in bins:
        po = assemble_panel(ntl_panel, plad_fixed, "gid_2", lo, hi)
        ps = assemble_panel(ntl_panel, plad_fixed, "gid_2_sj", lo, hi)
        ro = fit_lag0(po)
        rs = fit_lag0(ps)
        print(f"{lo}-{hi:>4d}  {ro['coef']:10.4f}  {ro['p']:7.3f}  "
              f"{rs['coef']:10.4f}  {rs['p']:7.3f}  {ro['treated']:8d}  {rs['treated']:8d}")

    out = DATA / "plad_gadm41_rejoin.csv"
    plad_sj[["plad_id", "leader", "country", "gid_0", "gid_2",
             "gid_2_sj", "latitude", "longitude", "geo_precision"]].to_csv(out, index=False)
    print(f"\n>> Wrote {out}")


if __name__ == "__main__":
    main()
