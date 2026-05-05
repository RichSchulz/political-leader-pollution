"""Produce three additions for the final paper (all restricted to 2005-2017).

Outputs
-------
data/summary_stats.csv          -- Table 1: mean/SD/N for key variables
data/lag_table_results.csv      -- Full lag-0/1/2 table for NTL, NO2 int, PM2.5 int
data/democracy_results.csv      -- Democracy heterogeneity (high/low V-Dem split)
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
YEAR_LO, YEAR_HI = 2005, 2017


# ── helpers (same as run_harmonized_analysis.py) ────────────────────────────

def load_plad():
    plad = pd.read_parquet(PLAD41_PATH)
    plad = plad[plad["foreign_leader"] == "0"].copy()
    plad = plad.loc[valid_gid_mask(plad["gid_2"])].copy()
    plad["startyear"] = plad["startyear"].astype(int)
    plad["endyear"]   = plad["endyear"].astype(int)
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
            rows.append({"GID_2": gid, "GID_0": row["gid_0"],
                         "year": y, "spell_id": idx})
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    ly = (pd.DataFrame(rows)
          .drop_duplicates(["GID_2", "year"])
          .assign(birth_region_leader=1))
    sm = ly[["GID_0", "year", "spell_id"]].drop_duplicates(["GID_0", "year"])
    return ly, sm


def assemble(ntl_raw, plad, year_lo, year_hi):
    ly, sm = build_leader_years(plad, year_lo, year_hi)
    p = ntl_raw[(ntl_raw["year"] >= year_lo) & (ntl_raw["year"] <= year_hi)].copy()
    p = p.merge(ly[["GID_2", "year", "birth_region_leader"]],
                on=["GID_2", "year"], how="left")
    p["birth_region_leader"] = p["birth_region_leader"].fillna(0).astype(int)
    p = p.merge(sm, on=["GID_0", "year"], how="left")
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


def fit(panel, outcome, treatment="birth_region_leader"):
    p = panel.dropna(subset=[outcome, "country_year", "spell_id"]).copy()
    if not isinstance(p.index, pd.MultiIndex):
        p = p.set_index(["GID_2", "year"])
    model = PanelOLS.from_formula(
        f"{outcome} ~ {treatment} + EntityEffects",
        data=p, other_effects=p["country_year"], drop_absorbed=True,
    )
    res = model.fit(cov_type="clustered", clusters=p["spell_id"])
    return {
        "coef":     float(res.params[treatment]),
        "se":       float(res.std_errors[treatment]),
        "pval":     float(res.pvalues[treatment]),
        "nobs":     int(res.nobs),
        "treated":  int(p[treatment].sum()),
        "clusters": int(p["spell_id"].nunique()),
    }


# ── load data ────────────────────────────────────────────────────────────────

print("Loading PLAD...")
plad = load_plad()

print("Loading harmonized NTL panel (2005-2019)...")
ntl_raw = pd.read_parquet(DATA / "nightlights_adm2_green_favoritism_panel.parquet")
ntl_raw = ntl_raw.loc[valid_gid_mask(ntl_raw["GID_2"])].copy()

print("Loading NO2 panel...")
no2_raw = pd.read_parquet(DATA / "no2_adm2_acag_panel.parquet")
no2_raw = no2_raw.loc[valid_gid_mask(no2_raw["GID_2"])].copy()
no2_raw = no2_raw[no2_raw["no2_mean"] >= 0].dropna(subset=["no2_mean"])

print("Loading PM2.5 panel...")
pm25_raw = pd.read_parquet(DATA / "pm25_adm2_acag_panel.parquet")
pm25_raw = pm25_raw.loc[valid_gid_mask(pm25_raw["GID_2"])].copy()
pm25_raw = pm25_raw[(pm25_raw["pm25_mean"] >= 0) & (pm25_raw["pm25_mean"] < 1000)].dropna(subset=["pm25_mean"])

# Restrict to 2005-2017
print("Assembling 2005-2017 panel...")
p = assemble(ntl_raw, plad, YEAR_LO, YEAR_HI)

no2_17  = no2_raw[no2_raw["year"].between(YEAR_LO, YEAR_HI)][["GID_2", "year", "no2_mean"]]
pm25_17 = pm25_raw[pm25_raw["year"].between(YEAR_LO, YEAR_HI)][["GID_2", "year", "pm25_mean"]]

joint = p.merge(no2_17, on=["GID_2", "year"], how="inner").copy()
joint["ln_no2"]            = np.log(joint["no2_mean"] + 0.01)
joint["no2_intensity"]     = joint["ln_no2"] - joint["ln_ntl"]

joint_pm = p.merge(pm25_17, on=["GID_2", "year"], how="inner").copy()
joint_pm["ln_pm25"]        = np.log(joint_pm["pm25_mean"] + 0.01)
joint_pm["pm25_intensity"] = joint_pm["ln_pm25"] - joint_pm["ln_ntl"]

print(f"  NTL panel:      {len(p):,} obs")
print(f"  NO2 joint:      {len(joint):,} obs")
print(f"  PM2.5 joint:    {len(joint_pm):,} obs")


# ── 1. SUMMARY STATISTICS ────────────────────────────────────────────────────

print("\n=== SUMMARY STATISTICS ===")

# Use the joint panel (all three variables present) for a consistent sample
# but report NTL stats from the full NTL panel for correct N
stats_rows = []
for label, df, col in [
    ("Log nightlights (harmonized)",  p,        "ln_ntl"),
    ("NO$_2$ (ppb)",                  joint,    "no2_mean"),
    ("PM$_{2.5}$ ($\\mu$g/m$^3$)",    joint_pm, "pm25_mean"),
    ("Birth-region indicator",        p,        "birth_region_leader"),
]:
    s = df[col].dropna()
    row = {
        "Variable": label,
        "Mean":  round(s.mean(), 3),
        "SD":    round(s.std(),  3),
        "Min":   round(s.min(),  3),
        "Max":   round(s.max(),  3),
        "N":     len(s),
    }
    stats_rows.append(row)
    print(f"  {label:45s}  mean={row['Mean']:.3f}  sd={row['SD']:.3f}  N={row['N']:,}")

pd.DataFrame(stats_rows).to_csv(DATA / "summary_stats.csv", index=False)
print("Wrote summary_stats.csv")


# ── 2. FULL LAG TABLE ────────────────────────────────────────────────────────

print("\n=== FULL LAG TABLE (2005-2017) ===")

lag_specs = [
    # (label,           panel,    outcome,            treatment)
    ("NTL lag 0",        p,        "ln_ntl",           "birth_region_leader"),
    ("NTL lag 1",        p,        "ln_ntl",           "brl_lag1"),
    ("NTL lag 2",        p,        "ln_ntl",           "brl_lag2"),
    ("NO2 int lag 0",    joint,    "no2_intensity",    "birth_region_leader"),
    ("NO2 int lag 1",    joint,    "no2_intensity",    "brl_lag1"),
    ("NO2 int lag 2",    joint,    "no2_intensity",    "brl_lag2"),
    ("PM25 int lag 0",   joint_pm, "pm25_intensity",   "birth_region_leader"),
    ("PM25 int lag 1",   joint_pm, "pm25_intensity",   "brl_lag1"),
    ("PM25 int lag 2",   joint_pm, "pm25_intensity",   "brl_lag2"),
]

lag_rows = []
for label, panel, outcome, treatment in lag_specs:
    try:
        r = fit(panel, outcome, treatment)
        r["spec"] = label
        lag_rows.append(r)
        print(f"  {label:25s}  coef={r['coef']:+.4f}  se={r['se']:.4f}  p={r['pval']:.3f}  N={r['nobs']:,}")
    except Exception as e:
        print(f"  {label:25s}  ERROR: {e}")

pd.DataFrame(lag_rows).to_csv(DATA / "lag_table_results.csv", index=False)
print("Wrote lag_table_results.csv")


# ── 3. DEMOCRACY HETEROGENEITY ───────────────────────────────────────────────

print("\n=== DEMOCRACY HETEROGENEITY ===")

vdem_files = list((DATA / "vdem").glob("*.csv"))
assert vdem_files, "No V-Dem CSV found in data/vdem/"
vdem = pd.read_csv(vdem_files[0], low_memory=False)
vdem = (vdem[["country_text_id", "year", "v2x_polyarchy"]]
        .rename(columns={"country_text_id": "GID_0", "v2x_polyarchy": "democracy"})
        .dropna(subset=["democracy"]))
vdem = vdem[(vdem["year"] >= YEAR_LO) & (vdem["year"] <= YEAR_HI)]

# Country-mean democracy over 2005-2017, then median split
country_dem = vdem.groupby("GID_0")["democracy"].mean().rename("mean_dem")
median_dem = country_dem.median()
high_dem = set(country_dem[country_dem >= median_dem].index)
low_dem  = set(country_dem[country_dem <  median_dem].index)
print(f"  Median democracy score: {median_dem:.3f}")
print(f"  High-democracy countries: {len(high_dem)}  |  Low-democracy: {len(low_dem)}")

dem_specs = [
    # (label,                     panel,    outcome,          mask)
    ("NTL high-dem",              p,        "ln_ntl",         p["GID_0"].isin(high_dem)),
    ("NTL low-dem",               p,        "ln_ntl",         p["GID_0"].isin(low_dem)),
    ("NO2 int high-dem",          joint,    "no2_intensity",  joint["GID_0"].isin(high_dem)),
    ("NO2 int low-dem",           joint,    "no2_intensity",  joint["GID_0"].isin(low_dem)),
    ("PM25 int high-dem",         joint_pm, "pm25_intensity", joint_pm["GID_0"].isin(high_dem)),
    ("PM25 int low-dem",          joint_pm, "pm25_intensity", joint_pm["GID_0"].isin(low_dem)),
]

dem_rows = []
for label, panel, outcome, mask in dem_specs:
    sub = panel[mask].copy()
    try:
        r = fit(sub, outcome)
        r["spec"] = label
        dem_rows.append(r)
        print(f"  {label:30s}  coef={r['coef']:+.4f}  se={r['se']:.4f}  p={r['pval']:.3f}  N={r['nobs']:,}")
    except Exception as e:
        print(f"  {label:30s}  ERROR: {e}")

pd.DataFrame(dem_rows).to_csv(DATA / "democracy_results.csv", index=False)
print("Wrote democracy_results.csv")

print("\nDone.")
