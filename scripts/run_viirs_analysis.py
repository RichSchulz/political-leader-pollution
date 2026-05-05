"""Run main VIIRS v21 specs on the rebuilt panel (nodata bug fixed).

Outputs results to stdout and overwrites:
  data/viirs_v21_2012_2024_analysis_summary.csv
  data/viirs_v21_window_grid_results.csv
  data/viirs_v21_2014_2024_bins.csv
"""
from __future__ import annotations

import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.append(str(ROOT / "scripts"))
from gid_utils import valid_gid_mask  # noqa: E402

PLAD41_PATH = DATA / "political leaders" / "PLAD_April_2024_gadm41.parquet"

SSA = set(
    "AGO BDI BEN BFA BWA CAF CIV CMR COD COG COM CPV DJI ERI ETH GAB GHA GIN GMB "
    "GNB GNQ KEN LBR LSO MDG MLI MOZ MRT MUS MWI NAM NER NGA RWA SDN SEN SLE SOM "
    "SSD STP SWZ TCD TGO TZA UGA ZAF ZMB ZWE".split()
)


def load_plad():
    plad = pd.read_parquet(PLAD41_PATH)
    plad = plad[plad["foreign_leader"] == "0"].copy()
    plad = plad.loc[valid_gid_mask(plad["gid_2"])].copy()
    plad["startyear"] = plad["startyear"].astype(int)
    plad["endyear"] = plad["endyear"].astype(int)
    plad = plad[plad["archigos_id"].astype(str).str.strip() != "."].copy()
    plad = plad.sort_values(["gid_0", "startyear"]).reset_index(drop=True)
    fixed = plad.copy().reset_index(drop=True)
    for _, group in fixed.groupby("gid_0"):
        idxs = group.index.tolist()
        for i in range(len(idxs) - 1):
            if fixed.loc[idxs[i], "endyear"] >= fixed.loc[idxs[i + 1], "startyear"]:
                fixed.loc[idxs[i], "endyear"] = fixed.loc[idxs[i + 1], "startyear"] - 1
    return fixed


def build_panel(plad, ntl, year_lo, year_hi, ssa_filter=None):
    rows = []
    for idx, row in plad.iterrows():
        gid = row["gid_2"]
        if pd.isna(gid) or str(gid).strip() in (".", ""):
            continue
        for y in range(max(int(row["startyear"]), year_lo),
                       min(int(row["endyear"]), year_hi) + 1):
            rows.append({"GID_2": gid, "GID_0": row["gid_0"], "year": y, "spell_id": idx})
    if not rows:
        return pd.DataFrame()
    ly = (pd.DataFrame(rows)
          .drop_duplicates(["GID_2", "year"])
          .assign(birth_region_leader=1))
    sm = ly[["GID_0", "year", "spell_id"]].drop_duplicates(["GID_0", "year"])

    p = ntl[(ntl["year"] >= year_lo) & (ntl["year"] <= year_hi)].copy()
    if ssa_filter == "ssa":
        p = p[p["GID_0"].isin(SSA)]
    elif ssa_filter == "nonssa":
        p = p[~p["GID_0"].isin(SSA)]
    p = p.merge(ly[["GID_2", "year", "birth_region_leader"]], on=["GID_2", "year"], how="left")
    p["birth_region_leader"] = p["birth_region_leader"].fillna(0).astype(int)
    p = p.merge(sm, on=["GID_0", "year"], how="left")
    p["spell_id"] = p["spell_id"].fillna(
        p["GID_0"] + "_" + p["year"].astype(str) + "_noleader"
    ).astype(str)
    p["ln_y"] = np.log(p["ntl_mean"] + 0.01)
    p["country_year"] = p["GID_0"] + "_" + p["year"].astype(str)
    p = p.dropna(subset=["ln_y"])
    p = p[np.isfinite(p["ln_y"])]
    return p


def fit(panel):
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
        "pval": float(res.pvalues["birth_region_leader"]),
        "nobs": int(res.nobs),
        "treated": int(p["birth_region_leader"].sum()),
        "clusters": int(p["spell_id"].nunique()),
        "regions": int(p.index.get_level_values("GID_2").nunique()),
        "countries": int(p["GID_0"].nunique()),
    }


def main():
    print(">> Loading PLAD (GADM 4.1 spatial join)...")
    plad = load_plad()
    print(f"   {len(plad)} leader spells")

    print(">> Loading VIIRS v21 2012-2024 panel...")
    viirs = pd.read_parquet(DATA / "nightlights_adm2_viirs_v21_2012_2024_panel.parquet")
    viirs = viirs.loc[valid_gid_mask(viirs["GID_2"])].copy()
    # Drop nodata sentinel rows (will also be NaN after log, but filter early)
    viirs = viirs[viirs["ntl_mean"] >= 0].copy()
    print(f"   {len(viirs):,} obs, {viirs['GID_2'].nunique():,} regions, "
          f"{viirs['year'].min()}-{viirs['year'].max()}")

    # --- Main specs ---
    summary_rows = []

    specs = [
        ("VIIRS-only 2012-2024 lag 0",        2012, 2024, None),
        ("VIIRS-only 2012-2024 lag 0 [SSA]",   2012, 2024, "ssa"),
        ("VIIRS-only 2012-2024 lag 0 [NonSSA]",2012, 2024, "nonssa"),
        ("VIIRS-only 2014-2024 lag 0",         2014, 2024, None),
        ("VIIRS-only 2014-2016 lag 0",         2014, 2016, None),
    ]

    print("\n>> Main specs:\n")
    print(f"{'spec':45s}  {'coef':>8s}  {'se':>7s}  {'p':>6s}  {'N':>9s}  {'treat':>5s}")
    print("-" * 90)
    for label, lo, hi, ssa in specs:
        p = build_panel(plad, viirs, lo, hi, ssa_filter=ssa)
        if p.empty:
            print(f"  {label:45s}  [no data]")
            continue
        r = fit(p)
        summary_rows.append({"spec": label, **r})
        print(f"  {label:45s}  {r['coef']:+8.4f}  {r['se']:7.4f}  "
              f"{r['pval']:6.3f}  {r['nobs']:9,}  {r['treated']:5d}")

    # --- Rolling window grid (2012 onwards, min 3 years) ---
    print("\n>> Rolling window grid (all windows ≥3 yrs starting 2012-2020):\n")
    print(f"{'window':>12s}  {'coef':>8s}  {'se':>7s}  {'p':>6s}  {'N':>9s}  {'treat':>5s}")
    print("-" * 65)
    window_rows = []
    for start in range(2012, 2021):
        for end in range(start + 2, 2025):
            p = build_panel(plad, viirs, start, end)
            if p.empty or p["birth_region_leader"].sum() < 10:
                continue
            try:
                r = fit(p)
            except Exception:
                continue
            window_rows.append({"Start": start, "End": end,
                                 "window_years": end - start + 1, **r})
            print(f"  {start}-{end}  {r['coef']:+8.4f}  {r['se']:7.4f}  "
                  f"{r['pval']:6.3f}  {r['nobs']:9,}  {r['treated']:5d}")

    # Save outputs
    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(DATA / "viirs_v21_2012_2024_analysis_summary.csv", index=False)
    print(f"\nWrote viirs_v21_2012_2024_analysis_summary.csv ({len(sum_df)} rows)")

    win_df = pd.DataFrame(window_rows)
    win_df.to_csv(DATA / "viirs_v21_window_grid_results.csv", index=False)
    print(f"Wrote viirs_v21_window_grid_results.csv ({len(win_df)} rows)")

    # 2014-2024 bins for reference
    bins_rows = []
    for lo, hi in [(2014, 2016), (2017, 2019), (2020, 2022), (2023, 2024)]:
        p = build_panel(plad, viirs, lo, hi)
        if p.empty:
            continue
        try:
            r = fit(p)
            bins_rows.append({"bin": f"{lo}-{hi}", **r})
        except Exception:
            pass
    bins_df = pd.DataFrame(bins_rows)
    bins_df.to_csv(DATA / "viirs_v21_2014_2024_bins.csv", index=False)
    print(f"Wrote viirs_v21_2014_2024_bins.csv ({len(bins_df)} rows)")


if __name__ == "__main__":
    main()
