"""Re-run all headline specs in the paper under the spatial-rejoined PLAD.

For each (panel × window × subsample), reports the lag-0 birth-region
coefficient under both:
  - 'orig'  : PLAD's gid_2 (GADM 3.6) string-matched to GADM 4.1 panels
  - 'sj'    : PLAD's lat/lon spatially joined onto GADM 4.1 polygons

Specs covered (mirrors what's reported in final report/main.tex):
  Nightlights:
    DMSP 1992-2013      [Pooled, SSA, Non-SSA]
    DMSP 2005-2013      [Pooled, SSA, Non-SSA]
    Harmonized 2005-2019[Pooled, SSA, Non-SSA]
    VIIRS v21 2012-2024 [Pooled, SSA, Non-SSA]
    VIIRS v21 2014-2024 [Pooled]
    VIIRS v21 2014-2016 [Pooled]
  Pollution (2005-2019):
    log(NO2)            [Pooled]
    NO2 intensity       [Pooled]   (= log(NO2) - log(NL) on common support)
"""
from __future__ import annotations

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

sys.path.append(str(ROOT / "scripts"))
from gid_utils import valid_gid_mask  # noqa: E402

SSA = set("AGO BDI BEN BFA BWA CAF CIV CMR COD COG COM CPV DJI ERI ETH GAB GHA GIN GMB GNB GNQ KEN LBR LSO MDG MLI MOZ MRT MUS MWI NAM NER NGA RWA SDN SEN SLE SOM SSD STP SWZ TCD TGO TZA UGA ZAF ZMB ZWE".split())


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
        if pd.isna(gid) or str(gid).strip() in (".", ""):
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


def assemble_panel(base_panel, value_col, plad_fixed, gid_col,
                   year_lo, year_hi, ssa_filter=None):
    """Return a regression-ready panel for `value_col` over [year_lo, year_hi].

    ssa_filter: None | 'ssa' | 'nonssa'
    """
    ly = build_leader_years(plad_fixed, gid_col, year_lo, year_hi)
    spell_map = ly[["GID_0", "year", "spell_id"]].drop_duplicates(subset=["GID_0", "year"])

    p = base_panel.loc[(base_panel["year"] >= year_lo) & (base_panel["year"] <= year_hi)].copy()
    if ssa_filter == "ssa":
        p = p[p["GID_0"].isin(SSA)]
    elif ssa_filter == "nonssa":
        p = p[~p["GID_0"].isin(SSA)]
    p = p.merge(ly[["GID_2", "year", "birth_region_leader"]],
                on=["GID_2", "year"], how="left")
    p["birth_region_leader"] = p["birth_region_leader"].fillna(0).astype(int)
    p = p.merge(spell_map, on=["GID_0", "year"], how="left")
    p["spell_id"] = p["spell_id"].fillna(
        p["GID_0"] + "_" + p["year"].astype(str) + "_noleader"
    ).astype(str)
    p = p.dropna(subset=[value_col])
    p["ln_y"] = np.log(p[value_col] + 0.01)
    p["country_year"] = p["GID_0"] + "_" + p["year"].astype(str)
    return p


def fit_lag0(panel):
    p = panel.copy()
    if not isinstance(p.index, pd.MultiIndex):
        p = p.set_index(["GID_2", "year"])
    model = PanelOLS.from_formula(
        "ln_y ~ birth_region_leader + EntityEffects",
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


def fit_intensity(no2_panel, nl_panel, plad_fixed, gid_col, year_lo, year_hi):
    """Fit log(NO2)-log(NL) outcome on the common GID_2-year support."""
    no2 = no2_panel[(no2_panel["year"] >= year_lo) & (no2_panel["year"] <= year_hi)][
        ["GID_2", "GID_0", "year", "no2_mean"]
    ]
    nl = nl_panel[(nl_panel["year"] >= year_lo) & (nl_panel["year"] <= year_hi)][
        ["GID_2", "year", "ntl_mean"]
    ]
    m = no2.merge(nl, on=["GID_2", "year"], how="inner").dropna()
    m["intensity"] = np.log(m["no2_mean"] + 0.01) - np.log(m["ntl_mean"] + 0.01)

    ly = build_leader_years(plad_fixed, gid_col, year_lo, year_hi)
    spell_map = ly[["GID_0", "year", "spell_id"]].drop_duplicates(subset=["GID_0", "year"])
    m = m.merge(ly[["GID_2", "year", "birth_region_leader"]],
                on=["GID_2", "year"], how="left")
    m["birth_region_leader"] = m["birth_region_leader"].fillna(0).astype(int)
    m = m.merge(spell_map, on=["GID_0", "year"], how="left")
    m["spell_id"] = m["spell_id"].fillna(
        m["GID_0"] + "_" + m["year"].astype(str) + "_noleader"
    ).astype(str)
    m["country_year"] = m["GID_0"] + "_" + m["year"].astype(str)

    if not isinstance(m.index, pd.MultiIndex):
        m = m.set_index(["GID_2", "year"])
    model = PanelOLS.from_formula(
        "intensity ~ birth_region_leader + EntityEffects",
        data=m, other_effects=m["country_year"], drop_absorbed=True,
    )
    res = model.fit(cov_type="clustered", clusters=m["spell_id"])
    return {
        "coef": float(res.params["birth_region_leader"]),
        "se":   float(res.std_errors["birth_region_leader"]),
        "p":    float(res.pvalues["birth_region_leader"]),
        "n":    int(res.nobs),
        "treated": int(m["birth_region_leader"].sum()),
    }


def run_all():
    print(">> Loading GADM 4.1 ADM2...")
    adm2 = gpd.read_file(GADM_PATH)
    adm2 = adm2.loc[valid_gid_mask(adm2["GID_2"])].copy()

    print(">> Loading PLAD and spatial-joining...")
    plad = load_plad()
    plad_sj = spatial_rejoin(plad, adm2)
    plad_fixed = clip_overlapping_spells(plad_sj)
    print(f"   leaders with orig gid_2:        {plad_fixed['gid_2'].apply(lambda s: isinstance(s,str) and s.strip()!='.').sum()}")
    print(f"   leaders with sj gid_2_sj:       {plad_fixed['gid_2_sj'].notna().sum()}")

    print(">> Loading panels...")
    dmsp = pd.read_parquet(DATA / "nightlights_adm2_panel.parquet")
    dmsp = dmsp.loc[valid_gid_mask(dmsp["GID_2"])].copy()
    harm = pd.read_parquet(DATA / "nightlights_adm2_green_favoritism_panel.parquet")
    harm = harm.loc[valid_gid_mask(harm["GID_2"])].copy()
    viirs = pd.read_parquet(DATA / "nightlights_adm2_viirs_v21_2012_2024_panel.parquet")
    viirs = viirs.loc[valid_gid_mask(viirs["GID_2"])].copy()
    no2 = pd.read_parquet(DATA / "no2_adm2_acag_panel.parquet")
    no2 = no2.loc[valid_gid_mask(no2["GID_2"])].copy()

    rows = []

    def add(label, base, value, lo, hi, ssa=None):
        for tag, col in [("orig", "gid_2"), ("sj", "gid_2_sj")]:
            p = assemble_panel(base, value, plad_fixed, col, lo, hi, ssa_filter=ssa)
            r = fit_lag0(p)
            rows.append({"spec": label, "join": tag, **r})

    # Nightlights
    print(">> Fitting nightlights specs...")
    add("DMSP 1992-2013 [Pooled]",       dmsp, "ntl_mean", 1992, 2013)
    add("DMSP 1992-2013 [SSA]",          dmsp, "ntl_mean", 1992, 2013, ssa="ssa")
    add("DMSP 1992-2013 [Non-SSA]",      dmsp, "ntl_mean", 1992, 2013, ssa="nonssa")
    add("DMSP 2005-2013 [Pooled]",       dmsp, "ntl_mean", 2005, 2013)
    add("DMSP 2005-2013 [SSA]",          dmsp, "ntl_mean", 2005, 2013, ssa="ssa")
    add("DMSP 2005-2013 [Non-SSA]",      dmsp, "ntl_mean", 2005, 2013, ssa="nonssa")
    add("Harm 2005-2019 [Pooled]",       harm, "ntl_mean", 2005, 2019)
    add("Harm 2005-2019 [SSA]",          harm, "ntl_mean", 2005, 2019, ssa="ssa")
    add("Harm 2005-2019 [Non-SSA]",      harm, "ntl_mean", 2005, 2019, ssa="nonssa")
    add("VIIRS21 2012-2024 [Pooled]",    viirs, "ntl_mean", 2012, 2024)
    add("VIIRS21 2012-2024 [SSA]",       viirs, "ntl_mean", 2012, 2024, ssa="ssa")
    add("VIIRS21 2012-2024 [Non-SSA]",   viirs, "ntl_mean", 2012, 2024, ssa="nonssa")
    add("VIIRS21 2014-2024 [Pooled]",    viirs, "ntl_mean", 2014, 2024)
    add("VIIRS21 2014-2016 [Pooled]",    viirs, "ntl_mean", 2014, 2016)

    # Pollution
    print(">> Fitting pollution specs...")
    add("NO2 2005-2019 [Pooled]",        no2, "no2_mean", 2005, 2019)
    for tag, col in [("orig", "gid_2"), ("sj", "gid_2_sj")]:
        r = fit_intensity(no2, harm, plad_fixed, col, 2005, 2019)
        rows.append({"spec": "NO2 intensity 2005-2019 [Pooled]", "join": tag, **r})

    df = pd.DataFrame(rows)

    # Pivot for side-by-side comparison
    pivot_coef = df.pivot(index="spec", columns="join", values="coef")
    pivot_p    = df.pivot(index="spec", columns="join", values="p")
    pivot_n    = df.pivot(index="spec", columns="join", values="treated")

    spec_order = [
        "DMSP 1992-2013 [Pooled]", "DMSP 1992-2013 [SSA]", "DMSP 1992-2013 [Non-SSA]",
        "DMSP 2005-2013 [Pooled]", "DMSP 2005-2013 [SSA]", "DMSP 2005-2013 [Non-SSA]",
        "Harm 2005-2019 [Pooled]", "Harm 2005-2019 [SSA]", "Harm 2005-2019 [Non-SSA]",
        "VIIRS21 2012-2024 [Pooled]", "VIIRS21 2012-2024 [SSA]", "VIIRS21 2012-2024 [Non-SSA]",
        "VIIRS21 2014-2024 [Pooled]", "VIIRS21 2014-2016 [Pooled]",
        "NO2 2005-2019 [Pooled]", "NO2 intensity 2005-2019 [Pooled]",
    ]

    print("\n" + "=" * 95)
    print(f"{'spec':36s}  {'orig coef':>10s} {'orig p':>7s}  "
          f"{'sj coef':>10s} {'sj p':>7s}   {'orig tr':>7s} {'sj tr':>7s}")
    print("-" * 95)
    for s in spec_order:
        if s not in pivot_coef.index:
            continue
        oc, sc = pivot_coef.loc[s, "orig"], pivot_coef.loc[s, "sj"]
        op, sp = pivot_p.loc[s, "orig"], pivot_p.loc[s, "sj"]
        ot, st = pivot_n.loc[s, "orig"], pivot_n.loc[s, "sj"]
        print(f"{s:36s}  {oc:10.4f} {op:7.3f}  {sc:10.4f} {sp:7.3f}   "
              f"{int(ot):7d} {int(st):7d}")
    print("=" * 95)

    out = DATA / "gadm_rejoin_full_results.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    run_all()
