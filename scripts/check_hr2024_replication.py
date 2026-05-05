"""Decompose the gap between our 0.0090 and H&R 2024 Table 2 Li column (0.048).

Builds a ladder of specifications that each move one knob toward H&R's spec,
so we can see which design choices drive the magnitude difference. Both
datasets are versions of Li et al.'s harmonized DMSP-VIIRS series in DMSP
units, so the data input is comparable.

Knobs:
  - lag of the birth-region indicator (0 vs 1)
  - time window (1992-2013 vs 1992-2019)
  - clustering (leader-spell vs country)
  - sample (with/without archigos_id filter)
  - PLAD GID_2 join (orig 3.6 string-match vs spatial re-join to 4.1)
"""
from __future__ import annotations

import sys
import warnings
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


def load_plad(filter_archigos=True):
    plad = pd.read_parquet(PLAD41_PATH)
    plad = plad[plad["foreign_leader"] == "0"].copy()
    plad = plad.loc[valid_gid_mask(plad["gid_2"])].copy()
    plad["startyear"] = plad["startyear"].astype(int)
    plad["endyear"] = plad["endyear"].astype(int)
    if filter_archigos:
        plad = plad[plad["archigos_id"].astype(str).str.strip() != "."].copy()
    plad = plad.sort_values(["gid_0", "startyear"]).reset_index(drop=True)
    fixed = plad.copy().reset_index(drop=True)
    for _, group in fixed.groupby("gid_0"):
        idxs = group.index.tolist()
        for i in range(len(idxs) - 1):
            if fixed.loc[idxs[i], "endyear"] >= fixed.loc[idxs[i + 1], "startyear"]:
                fixed.loc[idxs[i], "endyear"] = fixed.loc[idxs[i + 1], "startyear"] - 1
    return fixed


def build_panel(plad_fixed, ntl, gid_col, lag, year_lo, year_hi, cluster):
    """Return a regression-ready panel with the lag-`lag` indicator."""
    rows = []
    for idx, row in plad_fixed.iterrows():
        gid = row[gid_col]
        if pd.isna(gid) or str(gid).strip() in (".", ""):
            continue
        # leader is "in birth region" at calendar year y (lag 0).
        # For lag k, the indicator is 1 in year y+k for y in [start, end].
        for y in range(int(row["startyear"]), int(row["endyear"]) + 1):
            yy = y + lag
            if yy < year_lo or yy > year_hi:
                continue
            rows.append({"GID_2": gid, "GID_0": row["gid_0"],
                         "year": yy, "spell_id": idx})
    ly = (pd.DataFrame(rows)
            .drop_duplicates(["GID_2", "year"])
            .assign(birth_region_leader=1))
    sm = ly[["GID_0", "year", "spell_id"]].drop_duplicates(["GID_0", "year"])

    p = ntl[(ntl["year"] >= year_lo) & (ntl["year"] <= year_hi)].copy()
    p = p.merge(ly[["GID_2", "year", "birth_region_leader"]],
                on=["GID_2", "year"], how="left")
    p["birth_region_leader"] = p["birth_region_leader"].fillna(0).astype(int)
    p = p.merge(sm, on=["GID_0", "year"], how="left")
    p["spell_id"] = p["spell_id"].fillna(
        p["GID_0"] + "_" + p["year"].astype(str) + "_noleader"
    ).astype(str)
    p["ln_y"] = np.log(p["ntl_mean"] + 0.01)
    p["country_year"] = p["GID_0"] + "_" + p["year"].astype(str)
    p = p.dropna(subset=["ntl_mean"])
    return p


def fit(panel, cluster):
    p = panel.copy()
    if not isinstance(p.index, pd.MultiIndex):
        p = p.set_index(["GID_2", "year"])
    cluster_var = (p["spell_id"] if cluster == "spell" else
                   p.index.get_level_values("GID_2").map(
                       p.reset_index().groupby("GID_2")["GID_0"].first()))
    if cluster == "country":
        cluster_var = p.reset_index().set_index(["GID_2", "year"])["GID_0"]
    model = PanelOLS.from_formula(
        "ln_y ~ birth_region_leader + EntityEffects",
        data=p, other_effects=p["country_year"], drop_absorbed=True,
    )
    res = model.fit(cov_type="clustered", clusters=cluster_var)
    return {
        "coef": float(res.params["birth_region_leader"]),
        "se":   float(res.std_errors["birth_region_leader"]),
        "p":    float(res.pvalues["birth_region_leader"]),
        "n":    int(res.nobs),
        "treated": int(p["birth_region_leader"].sum()),
    }


def stitch_harmonized_1992_2019():
    """Concatenate the cached DMSP 1992-2013 panel with the harmonized 2005-2019
    panel, keeping only 2014-2019 from the latter to avoid year overlap."""
    dmsp = pd.read_parquet(DATA / "nightlights_adm2_panel.parquet")
    harm = pd.read_parquet(DATA / "nightlights_adm2_green_favoritism_panel.parquet")
    harm_late = harm[harm["year"] >= 2014].copy()
    full = pd.concat([dmsp, harm_late], ignore_index=True)
    full = full.loc[valid_gid_mask(full["GID_2"])].copy()
    return full


def main():
    print(">> Stitching harmonized 1992-2019 panel...")
    harm_full = stitch_harmonized_1992_2019()
    print(f"   {len(harm_full):,} obs, "
          f"{harm_full['year'].min()}-{harm_full['year'].max()}, "
          f"{harm_full['GID_2'].nunique():,} regions")

    dmsp_only = harm_full[harm_full["year"] <= 2013].copy()

    rows = []

    def step(label, plad, ntl, gid_col, lag, lo, hi, cluster):
        p = build_panel(plad, ntl, gid_col, lag, lo, hi, cluster)
        r = fit(p, cluster)
        rows.append({"step": label, **r})
        print(f"  {label:55s} coef={r['coef']:+.4f}  se={r['se']:.4f}  "
              f"p={r['p']:.3f}  N={r['n']:>9,}  treat={r['treated']:>5d}")

    print("\n>> Comparison ladder (each row changes ONE knob from previous)\n")

    plad_arch = load_plad(filter_archigos=True)
    plad_no_arch = load_plad(filter_archigos=False)

    print("[A] Our published spec (lag 0, 1992-2013, spell cluster, archigos filter, sj):")
    step("A. our paper spec", plad_arch, dmsp_only, "gid_2", 0, 1992, 2013, "spell")

    print("\n[B] + switch to lag 1 (H&R headline timing):")
    step("B. + lag 1", plad_arch, dmsp_only, "gid_2", 1, 1992, 2013, "spell")

    print("\n[C] + extend window to 1992-2019 (harmonized DMSP+simVIIRS):")
    step("C. + 1992-2019", plad_arch, harm_full, "gid_2", 1, 1992, 2019, "spell")

    print("\n[D] + cluster at country level (H&R convention):")
    step("D. + country cluster", plad_arch, harm_full, "gid_2", 1, 1992, 2019, "country")

    print("\n[E] + drop archigos_id filter (more leaders):")
    step("E. + drop archigos filter", plad_no_arch, harm_full, "gid_2", 1, 1992, 2019, "country")

    print("\n[F] Use orig PLAD GID_2 (3.6 string-match) instead of spatial re-join:")
    step("F. + orig gid_2 (3.6 match)", plad_no_arch, harm_full, "gid_2_orig", 1, 1992, 2019, "country")

    print("\n[REF] H&R 2024 Table 2 Panel A col (3) Li:                       coef=+0.0480  N=1,357,958")

    out = DATA / "hr2024_replication_ladder.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
