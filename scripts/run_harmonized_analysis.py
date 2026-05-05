"""Re-run all harmonized 2005-2019 specs against the rebuilt panel.

Computes:
  - Pooled / SSA / Non-SSA / Africa / Low-NO2-dispersion harm 2005-2019 NL
  - Joint NL+NO2+pollution-intensity (lag 0 & 1)
  - PM2.5 pollution intensity (lags 0-2)
  - Clustering robustness entry for harm 2005-2019 NL

Skips specs that need the DMSP 1992-2013 panel (long-run 1992-2019 window grid).
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

SSA = set(
    "AGO BDI BEN BFA BWA CAF CIV CMR COD COG COM CPV DJI ERI ETH GAB GHA GIN GMB "
    "GNB GNQ KEN LBR LSO MDG MLI MOZ MRT MUS MWI NAM NER NGA RWA SDN SEN SLE SOM "
    "SSD STP SWZ TCD TGO TZA UGA ZAF ZMB ZWE".split()
)
AFRICA = set(
    "AGO BDI BEN BFA BWA CAF CIV CMR COD COG COM CPV DJI DZA EGY ERI ETH GAB GHA "
    "GIN GMB GNB GNQ KEN LBR LSO LBY MAR MDG MLI MOZ MRT MUS MWI NAM NER NGA RWA "
    "SDN SEN SLE SOM SSD STP SWZ TCD TGO TUN TZA UGA ZAF ZMB ZWE".split()
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


def build_leader_years(plad, year_lo, year_hi):
    rows = []
    for idx, row in plad.iterrows():
        gid = row["gid_2"]
        if pd.isna(gid) or str(gid).strip() in (".", ""):
            continue
        for y in range(max(int(row["startyear"]), year_lo),
                       min(int(row["endyear"]), year_hi) + 1):
            rows.append({"GID_2": gid, "GID_0": row["gid_0"], "year": y, "spell_id": idx})
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    ly = (pd.DataFrame(rows)
          .drop_duplicates(["GID_2", "year"])
          .assign(birth_region_leader=1))
    sm = ly[["GID_0", "year", "spell_id"]].drop_duplicates(["GID_0", "year"])
    return ly, sm


def assemble(ntl, plad, year_lo, year_hi, extra_cols=None):
    """Return a panel with birth_region_leader, lag cols, country_year, etc."""
    ly, sm = build_leader_years(plad, year_lo, year_hi)
    p = ntl[(ntl["year"] >= year_lo) & (ntl["year"] <= year_hi)].copy()
    if ly.empty:
        p["birth_region_leader"] = 0
    else:
        p = p.merge(ly[["GID_2", "year", "birth_region_leader"]], on=["GID_2", "year"], how="left")
        p["birth_region_leader"] = p["birth_region_leader"].fillna(0).astype(int)
        p = p.merge(sm, on=["GID_0", "year"], how="left")
    p["spell_id"] = p.get("spell_id", pd.Series(pd.NA, index=p.index))
    p["spell_id"] = p["spell_id"].fillna(
        p["GID_0"] + "_" + p["year"].astype(str) + "_noleader"
    ).astype(str)
    p["ln_ntl"] = np.log(p["ntl_mean"] + 0.01)
    p["country_year"] = p["GID_0"] + "_" + p["year"].astype(str)
    p = p.sort_values(["GID_2", "year"])
    p["brl_lag1"] = p.groupby("GID_2")["birth_region_leader"].shift(1).fillna(0).astype(int)
    p["brl_lag2"] = p.groupby("GID_2")["birth_region_leader"].shift(2).fillna(0).astype(int)
    p = p.dropna(subset=["ntl_mean"])
    p = p[np.isfinite(p["ln_ntl"])]
    return p


def fit(panel, outcome="ln_ntl", treatment="birth_region_leader", cluster="spell"):
    p = panel.copy()
    p = p.dropna(subset=[outcome, "country_year", "spell_id"])
    if not isinstance(p.index, pd.MultiIndex):
        p = p.set_index(["GID_2", "year"])
    if cluster == "country":
        cluster_var = p.reset_index().set_index(["GID_2", "year"])["GID_0"]
    else:
        cluster_var = p["spell_id"]
    model = PanelOLS.from_formula(
        f"{outcome} ~ {treatment} + EntityEffects",
        data=p, other_effects=p["country_year"], drop_absorbed=True,
    )
    res = model.fit(cov_type="clustered", clusters=cluster_var)
    return {
        "coef": float(res.params[treatment]),
        "se":   float(res.std_errors[treatment]),
        "pval": float(res.pvalues[treatment]),
        "nobs": int(res.nobs),
        "treated": int(p[treatment].sum()),
        "clusters": int(p["spell_id"].nunique()),
        "countries": int(p["GID_0"].nunique()),
    }


def print_row(label, r):
    print(f"  {label:50s}  {r['coef']:+8.4f}  {r['se']:7.4f}  "
          f"{r['pval']:6.3f}  {r['nobs']:9,}  {r['treated']:5d}")


def main():
    print(">> Loading PLAD...")
    plad = load_plad()
    print(f"   {len(plad)} leader spells")

    print(">> Loading harmonized 2005-2019 panel...")
    harm = pd.read_parquet(DATA / "nightlights_adm2_green_favoritism_panel.parquet")
    harm = harm.loc[valid_gid_mask(harm["GID_2"])].copy()
    harm = harm.dropna(subset=["ntl_mean"])
    print(f"   {len(harm):,} obs, {harm['GID_2'].nunique():,} regions, "
          f"{harm['year'].min()}-{harm['year'].max()}")

    print(">> Loading NO2 and PM2.5 panels...")
    no2 = pd.read_parquet(DATA / "no2_adm2_acag_panel.parquet")
    no2 = no2.loc[valid_gid_mask(no2["GID_2"])].dropna(subset=["no2_mean"]).copy()
    no2 = no2[no2["no2_mean"] >= 0]

    pm25 = pd.read_parquet(DATA / "pm25_adm2_acag_panel.parquet")
    pm25 = pm25.loc[valid_gid_mask(pm25["GID_2"])].dropna(subset=["pm25_mean"]).copy()
    pm25 = pm25[(pm25["pm25_mean"] >= 0) & (pm25["pm25_mean"] < 1000)]

    # Low-NO2-dispersion mask: countries below median of region-SD of NO2 (2005-2007)
    no2_early = no2[no2["year"].isin(range(2005, 2008))]
    region_mean = no2_early.groupby(["GID_0", "GID_2"])["no2_mean"].mean().reset_index()
    country_sd = region_mean.groupby("GID_0")["no2_mean"].std().rename("disp").reset_index()
    med = country_sd["disp"].median()
    low_no2_disp_countries = set(country_sd[country_sd["disp"] < med]["GID_0"])

    p_full = assemble(harm, plad, 2005, 2019)

    # ── NL specs ──────────────────────────────────────────────────────────────
    print(f"\n{'spec':52s}  {'coef':>8s}  {'se':>7s}  {'p':>6s}  {'N':>9s}  {'treat':>5s}")
    print("-" * 97)

    results = {}

    def reg(label, mask=None, outcome="ln_ntl", treatment="birth_region_leader"):
        d = p_full if mask is None else p_full[mask]
        try:
            r = fit(d, outcome, treatment)
        except Exception as e:
            print(f"  {label:50s}  ERROR: {e}")
            return None
        results[label] = r
        print_row(label, r)
        return r

    reg("Harm 2005-2019 pooled")
    reg("Harm 2005-2019 SSA",      mask=p_full["GID_0"].isin(SSA))
    reg("Harm 2005-2019 Non-SSA",  mask=~p_full["GID_0"].isin(SSA))
    reg("Harm 2005-2019 Africa",   mask=p_full["GID_0"].isin(AFRICA))
    reg("Harm 2005-2019 Low-NO2-disp", mask=p_full["GID_0"].isin(low_no2_disp_countries))

    # Clustering robustness for pooled NL
    d_full = p_full.copy()
    try:
        r_spell = fit(d_full, "ln_ntl", cluster="spell")
        r_ctry  = fit(d_full, "ln_ntl", cluster="country")
        print(f"\n  Clustering robustness (harm 2005-2019 pooled NL):")
        print(f"    Spell-cluster SE : {r_spell['se']:.4f}  p={r_spell['pval']:.3f}")
        print(f"    Country-cluster SE: {r_ctry['se']:.4f}  p={r_ctry['pval']:.3f}")
        print(f"    (coef = {r_spell['coef']:+.4f})")
    except Exception as e:
        print(f"  Clustering robustness error: {e}")

    # ── Joint NL + NO2 panel ─────────────────────────────────────────────────
    print("\n>> Building joint NL+NO2 panel (inner merge)...")
    joint = p_full.merge(
        no2[no2["year"].between(2005, 2019)][["GID_2", "year", "no2_mean"]],
        on=["GID_2", "year"], how="inner"
    )
    joint = joint[joint["no2_mean"] >= 0].copy()
    joint["ln_no2"] = np.log(joint["no2_mean"] + 0.01)
    joint["pollution_intensity"] = joint["ln_no2"] - joint["ln_ntl"]
    print(f"   joint panel: {len(joint):,} obs")

    print(f"\n{'spec':52s}  {'coef':>8s}  {'se':>7s}  {'p':>6s}  {'N':>9s}  {'treat':>5s}")
    print("-" * 97)

    for label, outcome, treatment in [
        ("Joint NL lag 0",             "ln_ntl",              "birth_region_leader"),
        ("Joint NL lag 1",             "ln_ntl",              "brl_lag1"),
        ("NO2 lag 0",                  "ln_no2",              "birth_region_leader"),
        ("NO2 lag 1",                  "ln_no2",              "brl_lag1"),
        ("NO2 pollution intensity lag 0", "pollution_intensity", "birth_region_leader"),
        ("NO2 pollution intensity lag 1", "pollution_intensity", "brl_lag1"),
    ]:
        try:
            r = fit(joint, outcome, treatment)
            results[label] = r
            print_row(label, r)
        except Exception as e:
            print(f"  {label:50s}  ERROR: {e}")

    # ── PM2.5 joint panel ────────────────────────────────────────────────────
    print("\n>> Building joint NL+PM2.5 panel (inner merge)...")
    joint_pm = p_full.merge(
        pm25[pm25["year"].between(2005, 2019)][["GID_2", "year", "pm25_mean"]],
        on=["GID_2", "year"], how="inner"
    )
    joint_pm = joint_pm.dropna(subset=["pm25_mean"]).copy()
    joint_pm["ln_pm25"] = np.log(joint_pm["pm25_mean"] + 0.01)
    joint_pm["pm25_intensity"] = joint_pm["ln_pm25"] - joint_pm["ln_ntl"]
    print(f"   joint PM2.5 panel: {len(joint_pm):,} obs")

    print(f"\n{'spec':52s}  {'coef':>8s}  {'se':>7s}  {'p':>6s}  {'N':>9s}  {'treat':>5s}")
    print("-" * 97)

    for label, outcome, treatment in [
        ("PM2.5 intensity lag 0",   "pm25_intensity", "birth_region_leader"),
        ("PM2.5 intensity lag 1",   "pm25_intensity", "brl_lag1"),
        ("PM2.5 intensity lag 2",   "pm25_intensity", "brl_lag2"),
    ]:
        try:
            r = fit(joint_pm, outcome, treatment)
            results[label] = r
            print_row(label, r)
        except Exception as e:
            print(f"  {label:50s}  ERROR: {e}")

    # Summary for paper
    print("\n\n=== NUMBERS FOR PAPER ===")
    for k, r in results.items():
        print(f"{k}: coef={r['coef']:+.4f}, se={r['se']:.4f}, p={r['pval']:.3f}, "
              f"N={r['nobs']:,}, treat={r['treated']}")

    out = pd.DataFrame([{"spec": k, **v} for k, v in results.items()])
    out.to_csv(DATA / "harmonized_analysis_results.csv", index=False)
    print(f"\nWrote harmonized_analysis_results.csv")


if __name__ == "__main__":
    main()
