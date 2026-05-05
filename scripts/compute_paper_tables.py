"""Compute all numbers needed for the corrected paper tables.

Outputs:
  - SSA/Non-SSA splits: DMSP 1992-2013, DMSP 2005-2013, Harm 2005-2017
  - Mineral rent decomposition: DMSP 1992-2013
  - Clustering robustness: DMSP 1992-2013, DMSP 2005-2013, Harm 2005-2017
  - Pollution results: 2005-2017 (NL, NO2, intensity, PM2.5)
  - Africa split: DMSP 1992-2013
  - Low-NO2-dispersion: Harm 2005-2017
  - Subsample table rows with new data
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

SSA = set("AGO BDI BEN BFA BWA CAF CIV CMR COD COG COM CPV DJI ERI ETH GAB GHA GIN GMB "
          "GNB GNQ KEN LBR LSO MDG MLI MOZ MRT MUS MWI NAM NER NGA RWA SDN SEN SLE SOM "
          "SSD STP SWZ TCD TGO TZA UGA ZAF ZMB ZWE".split())
AFRICA = set("AGO BDI BEN BFA BWA CAF CIV CMR COD COG COM CPV DJI DZA EGY ERI ETH GAB GHA "
             "GIN GMB GNB GNQ KEN LBR LSO LBY MAR MDG MLI MOZ MRT MUS MWI NAM NER NGA RWA "
             "SDN SEN SLE SOM SSD STP SWZ TCD TGO TUN TZA UGA ZAF ZMB ZWE".split())


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
    ly = (pd.DataFrame(rows).drop_duplicates(["GID_2", "year"])
          .assign(birth_region_leader=1))
    sm = ly[["GID_0", "year", "spell_id"]].drop_duplicates(["GID_0", "year"])
    return ly, sm


def assemble(ntl, plad, year_lo, year_hi):
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
        p["GID_0"] + "_" + p["year"].astype(str) + "_noleader").astype(str)
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
    }


def pr(label, r, extra=""):
    print(f"  {label:55s}  {r['coef']:+8.4f}  {r['se']:7.4f}  {r['pval']:8.4f}  "
          f"{r['nobs']:9,}  {r['treated']:5d}  {extra}")


print(">> Loading data...")
plad = load_plad()

dmsp = pd.read_parquet(DATA / "nightlights_adm2_panel.parquet")
dmsp = dmsp.loc[valid_gid_mask(dmsp["GID_2"])].copy()
dmsp = dmsp.dropna(subset=["ntl_mean"])

harm = pd.read_parquet(DATA / "nightlights_adm2_green_favoritism_panel.parquet")
harm = harm.loc[valid_gid_mask(harm["GID_2"])].copy()
harm = harm.dropna(subset=["ntl_mean"])

no2 = pd.read_parquet(DATA / "no2_adm2_acag_panel.parquet")
no2 = no2.loc[valid_gid_mask(no2["GID_2"])].dropna(subset=["no2_mean"]).copy()
no2 = no2[no2["no2_mean"] >= 0]

pm25 = pd.read_parquet(DATA / "pm25_adm2_acag_panel.parquet")
pm25 = pm25.loc[valid_gid_mask(pm25["GID_2"])].dropna(subset=["pm25_mean"]).copy()
pm25 = pm25[(pm25["pm25_mean"] >= 0) & (pm25["pm25_mean"] < 1000)]

wdi = pd.read_csv(DATA / "wdi_resource_rents.csv")
wdi = wdi.rename(columns={"country_code": "GID_0", "mineral_rents_gdp": "mineral_rents_pct_gdp"})
country_mean_rents = wdi.groupby("GID_0")["mineral_rents_pct_gdp"].mean()
q75 = country_mean_rents.quantile(0.75)
top_q_countries = set(country_mean_rents[country_mean_rents >= q75].index)
print(f"   Top-quartile mineral rent threshold: {q75:.3f}%")

# Low-NO2-dispersion mask
no2_early = no2[no2["year"].isin(range(2005, 2008))]
region_mean = no2_early.groupby(["GID_0", "GID_2"])["no2_mean"].mean().reset_index()
country_sd = region_mean.groupby("GID_0")["no2_mean"].std().rename("disp").reset_index()
med = country_sd["disp"].median()
low_no2_disp_countries = set(country_sd[country_sd["disp"] < med]["GID_0"])

print(">> Assembling panels...")
p_dmsp1992 = assemble(dmsp, plad, 1992, 2013)
p_dmsp2005 = assemble(dmsp, plad, 2005, 2013)
p_harm2005 = assemble(harm, plad, 2005, 2017)

print(f"   DMSP 1992-2013: {len(p_dmsp1992):,} obs")
print(f"   DMSP 2005-2013: {len(p_dmsp2005):,} obs")
print(f"   Harm 2005-2017: {len(p_harm2005):,} obs")

hdr = f"  {'label':55s}  {'coef':>8s}  {'se':>7s}  {'p':>8s}  {'N':>9s}  {'treat':>5s}"
sep = "-" * 110

# ── SSA / Non-SSA / Africa decomposition ──────────────────────────────────────
print(f"\n{'='*110}")
print("SSA / Non-SSA / Africa decomposition")
print(hdr); print(sep)

for label, p, mask_ssa, mask_nonssa, mask_africa in [
    ("DMSP 1992-2013", p_dmsp1992,
     p_dmsp1992["GID_0"].isin(SSA), ~p_dmsp1992["GID_0"].isin(SSA), p_dmsp1992["GID_0"].isin(AFRICA)),
    ("DMSP 2005-2013", p_dmsp2005,
     p_dmsp2005["GID_0"].isin(SSA), ~p_dmsp2005["GID_0"].isin(SSA), p_dmsp2005["GID_0"].isin(AFRICA)),
    ("Harm 2005-2017", p_harm2005,
     p_harm2005["GID_0"].isin(SSA), ~p_harm2005["GID_0"].isin(SSA), p_harm2005["GID_0"].isin(AFRICA)),
]:
    for sublabel, mask in [("Pooled", None), ("SSA", mask_ssa), ("Non-SSA", mask_nonssa), ("Africa", mask_africa)]:
        d = p if mask is None else p[mask]
        try:
            r = fit(d)
            pr(f"{label} {sublabel}", r)
        except Exception as e:
            print(f"  {label} {sublabel}: ERROR {e}")

# ── Clustering robustness ──────────────────────────────────────────────────────
print(f"\n{'='*110}")
print("Clustering robustness")
print(f"  {'label':55s}  {'coef':>8s}  {'spell_se':>8s}  {'spell_p':>8s}  {'ctry_se':>7s}  {'ctry_p':>7s}")
print(sep)

for label, p in [("DMSP 1992-2013", p_dmsp1992), ("DMSP 2005-2013", p_dmsp2005), ("Harm 2005-2017", p_harm2005)]:
    try:
        r_spell = fit(p, cluster="spell")
        r_ctry  = fit(p, cluster="country")
        print(f"  {label:55s}  {r_spell['coef']:+8.4f}  {r_spell['se']:8.4f}  {r_spell['pval']:8.4f}  "
              f"{r_ctry['se']:7.4f}  {r_ctry['pval']:7.4f}")
    except Exception as e:
        print(f"  {label}: ERROR {e}")

# ── Mineral rent decomposition ─────────────────────────────────────────────────
print(f"\n{'='*110}")
print("Mineral rent decomposition (DMSP 1992-2013)")
print(hdr); print(sep)

p_wdi = p_dmsp1992[p_dmsp1992["GID_0"].isin(set(wdi["GID_0"].unique()))].copy()
masks_mineral = [
    ("All WDI-merged", None),
    ("Excl. top-q mineral-rent", ~p_wdi["GID_0"].isin(top_q_countries)),
    ("Top-q mineral-rent only", p_wdi["GID_0"].isin(top_q_countries)),
    ("SSA only", p_wdi["GID_0"].isin(SSA)),
    ("SSA excl. top-q", p_wdi["GID_0"].isin(SSA) & ~p_wdi["GID_0"].isin(top_q_countries)),
    ("SSA top-q only", p_wdi["GID_0"].isin(SSA) & p_wdi["GID_0"].isin(top_q_countries)),
    ("Non-SSA top-q only", ~p_wdi["GID_0"].isin(SSA) & p_wdi["GID_0"].isin(top_q_countries)),
    ("Non-SSA excl. top-q", ~p_wdi["GID_0"].isin(SSA) & ~p_wdi["GID_0"].isin(top_q_countries)),
]
for sublabel, mask in masks_mineral:
    d = p_wdi if mask is None else p_wdi[mask]
    try:
        r = fit(d)
        pr(sublabel, r)
    except Exception as e:
        print(f"  {sublabel}: ERROR {e}")

# ── Pollution results 2005-2017 ────────────────────────────────────────────────
print(f"\n{'='*110}")
print("Pollution results 2005-2017")
print(hdr); print(sep)

joint = p_harm2005.merge(
    no2[no2["year"].between(2005, 2017)][["GID_2", "year", "no2_mean"]],
    on=["GID_2", "year"], how="inner"
)
joint = joint[joint["no2_mean"] >= 0].copy()
joint["ln_no2"] = np.log(joint["no2_mean"] + 0.01)
joint["pollution_intensity"] = joint["ln_no2"] - joint["ln_ntl"]

joint_pm = p_harm2005.merge(
    pm25[pm25["year"].between(2005, 2017)][["GID_2", "year", "pm25_mean"]],
    on=["GID_2", "year"], how="inner"
)
joint_pm = joint_pm.dropna(subset=["pm25_mean"]).copy()
joint_pm["ln_pm25"] = np.log(joint_pm["pm25_mean"] + 0.01)
joint_pm["pm25_intensity"] = joint_pm["ln_pm25"] - joint_pm["ln_ntl"]

for label, d, outcome, treatment in [
    ("NL lag 0 (harm 2005-2017)", p_harm2005, "ln_ntl", "birth_region_leader"),
    ("NL lag 1 (harm 2005-2017)", p_harm2005, "ln_ntl", "brl_lag1"),
    ("NO2 lag 0", joint, "ln_no2", "birth_region_leader"),
    ("NO2 lag 1", joint, "ln_no2", "brl_lag1"),
    ("NO2 intensity lag 0", joint, "pollution_intensity", "birth_region_leader"),
    ("NO2 intensity lag 1", joint, "pollution_intensity", "brl_lag1"),
    ("PM2.5 intensity lag 0", joint_pm, "pm25_intensity", "birth_region_leader"),
    ("PM2.5 intensity lag 1", joint_pm, "pm25_intensity", "brl_lag1"),
    ("PM2.5 intensity lag 2", joint_pm, "pm25_intensity", "brl_lag2"),
]:
    try:
        r = fit(d, outcome, treatment)
        pr(label, r)
    except Exception as e:
        print(f"  {label}: ERROR {e}")

# Also: pollution clustering robustness for harm 2005-2017
print(f"\n  Clustering robustness for pollution outcomes (harm 2005-2017):")
print(f"  {'label':45s}  {'spell_se':>8s}  {'spell_p':>8s}  {'ctry_se':>7s}  {'ctry_p':>7s}")
for label, d, outcome in [
    ("NO2 lag 0", joint, "ln_no2"),
    ("NO2 intensity lag 0", joint, "pollution_intensity"),
    ("PM2.5 intensity lag 0", joint_pm, "pm25_intensity"),
]:
    try:
        r_s = fit(d, outcome, "birth_region_leader", "spell")
        r_c = fit(d, outcome, "birth_region_leader", "country")
        print(f"  {label:45s}  {r_s['se']:8.4f}  {r_s['pval']:8.4f}  {r_c['se']:7.4f}  {r_c['pval']:7.4f}")
    except Exception as e:
        print(f"  {label}: ERROR {e}")

# ── Harm 2005-2017 subsamples ──────────────────────────────────────────────────
print(f"\n{'='*110}")
print("Harm 2005-2017 subsamples")
print(hdr); print(sep)

for sublabel, mask in [
    ("Low NO2 disp", p_harm2005["GID_0"].isin(low_no2_disp_countries)),
    ("SSA", p_harm2005["GID_0"].isin(SSA)),
    ("Africa", p_harm2005["GID_0"].isin(AFRICA)),
    ("Non-SSA", ~p_harm2005["GID_0"].isin(SSA)),
]:
    d = p_harm2005[mask]
    try:
        r = fit(d)
        pr(f"Harm 2005-2017 {sublabel}", r)
    except Exception as e:
        print(f"  Harm 2005-2017 {sublabel}: ERROR {e}")

# ── Pollution subsamples (harm 2005-2017) ─────────────────────────────────────
print(f"\n  Pollution subsamples (NO2 intensity lag 0, harm 2005-2017 window):")
for sublabel, mask_fn in [
    ("SSA", lambda d: d["GID_0"].isin(SSA)),
    ("Africa", lambda d: d["GID_0"].isin(AFRICA)),
    ("Low NO2 disp", lambda d: d["GID_0"].isin(low_no2_disp_countries)),
]:
    for d_base, d_label in [(joint, "NO2"), (joint_pm, "PM2.5")]:
        m = mask_fn(d_base)
        d = d_base[m]
        outcome = "pollution_intensity" if d_label == "NO2" else "pm25_intensity"
        try:
            r = fit(d, outcome, "birth_region_leader")
            print(f"  {sublabel} {d_label} intensity:  coef={r['coef']:+.4f}  se={r['se']:.4f}  p={r['pval']:.4f}  N={r['nobs']:,}")
        except Exception as e:
            print(f"  {sublabel} {d_label}: ERROR {e}")

print("\n\nDONE")
