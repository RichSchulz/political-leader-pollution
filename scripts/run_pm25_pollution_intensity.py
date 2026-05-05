"""
Run a PM2.5 pollution-intensity specification on the overlapping
2005-2019 harmonized nightlights window.

The outcome mirrors the NO2 notebook's pollution-intensity measure:

    log(PM2.5 + 0.01) - log(Nightlights + 0.01)

This keeps the same treatment construction, fixed effects, and
leader-period clustering as the existing PM2.5 analysis.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

try:
    from gid_utils import valid_gid_mask
except ModuleNotFoundError:
    from scripts.gid_utils import valid_gid_mask

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLAD_PATH = DATA / "political leaders" / "PLAD_April_2024_gadm41.parquet"
NTL_CACHE = DATA / "nightlights_adm2_green_favoritism_panel.parquet"
PM25_PANEL_CACHE = DATA / "pm25_adm2_acag_panel.parquet"
OUT_PATH = DATA / "pm25_pollution_intensity_results.csv"

ANALYSIS_YEARS = range(2005, 2020)


def load_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    assert NTL_CACHE.exists(), f"Nightlights panel not found at {NTL_CACHE}"
    assert PM25_PANEL_CACHE.exists(), f"PM2.5 panel not found at {PM25_PANEL_CACHE}"

    ntl = pd.read_parquet(NTL_CACHE)
    ntl = ntl[ntl["year"].isin(ANALYSIS_YEARS)].copy()
    ntl = ntl.loc[valid_gid_mask(ntl["GID_2"])].copy()
    ntl = ntl.dropna(subset=["ntl_mean"])

    pm25 = pd.read_parquet(PM25_PANEL_CACHE)
    pm25 = pm25[pm25["year"].isin(ANALYSIS_YEARS)].copy()
    pm25 = pm25.loc[valid_gid_mask(pm25["GID_2"])].copy()
    pm25 = pm25.dropna(subset=["pm25_mean"])
    pm25 = pm25[(pm25["pm25_mean"] >= 0) & (pm25["pm25_mean"] < 1000)].copy()

    return ntl, pm25


def build_treatment() -> tuple[pd.DataFrame, pd.DataFrame]:
    assert PLAD_PATH.exists(), f"PLAD data not found at {PLAD_PATH}"

    plad = pd.read_parquet(PLAD_PATH)
    plad = plad[plad["foreign_leader"] == "0"].copy()

    birth_gid = "gid_2" if "gid_2" in plad.columns else "gid_1"
    plad = plad.loc[valid_gid_mask(plad[birth_gid])].copy()
    plad["startyear"] = plad["startyear"].astype(int)
    plad["endyear"] = plad["endyear"].astype(int)
    plad = plad[plad["archigos_id"].str.strip() != "."].copy()
    plad = plad.sort_values(["gid_0", "startyear"]).reset_index(drop=True)

    # Match the notebook logic that removes overlapping transition years.
    plad_fixed = plad.copy().reset_index(drop=True)
    for gid_0, group in plad_fixed.groupby("gid_0"):
        idxs = group.index.tolist()
        for i in range(len(idxs) - 1):
            curr, nxt = idxs[i], idxs[i + 1]
            if plad_fixed.loc[curr, "endyear"] >= plad_fixed.loc[nxt, "startyear"]:
                plad_fixed.loc[curr, "endyear"] = plad_fixed.loc[nxt, "startyear"] - 1

    year_min, year_max = min(ANALYSIS_YEARS), max(ANALYSIS_YEARS)
    rows: list[dict[str, object]] = []
    for idx, row in plad_fixed.iterrows():
        for year in range(
            max(int(row["startyear"]), year_min),
            min(int(row["endyear"]), year_max) + 1,
        ):
            rows.append(
                {
                    "GID_2": row[birth_gid],
                    "GID_0": row["gid_0"],
                    "year": year,
                    "spell_id": idx,
                }
            )

    leader_years = pd.DataFrame(rows)
    leader_years = leader_years.loc[valid_gid_mask(leader_years["GID_2"])].copy()
    leader_years = leader_years.drop_duplicates(subset=["GID_2", "year"])
    leader_years["birth_region_leader"] = 1

    spell_map = (
        leader_years[["GID_0", "year", "spell_id"]]
        .drop_duplicates(subset=["GID_0", "year"])
        .copy()
    )
    return leader_years, spell_map


def build_analysis_panel() -> pd.DataFrame:
    ntl, pm25 = load_panels()
    leader_years, spell_map = build_treatment()

    panel = ntl.merge(pm25[["GID_2", "year", "pm25_mean"]], on=["GID_2", "year"], how="inner")
    panel = panel.merge(
        leader_years[["GID_2", "year", "birth_region_leader"]],
        on=["GID_2", "year"],
        how="left",
    )
    panel["birth_region_leader"] = panel["birth_region_leader"].fillna(0).astype(int)
    panel = panel.merge(spell_map[["GID_0", "year", "spell_id"]], on=["GID_0", "year"], how="left")
    panel["spell_id"] = panel["spell_id"].fillna(
        panel["GID_0"] + "_" + panel["year"].astype(str) + "_noleader"
    ).astype(str)

    panel = panel.sort_values(["GID_2", "year"])
    panel["brl_lag1"] = panel.groupby("GID_2")["birth_region_leader"].shift(1).fillna(0).astype(int)
    panel["brl_lag2"] = panel.groupby("GID_2")["birth_region_leader"].shift(2).fillna(0).astype(int)

    panel["ln_ntl"] = np.log(panel["ntl_mean"] + 0.01)
    panel["ln_pm25"] = np.log(panel["pm25_mean"] + 0.01)
    panel["pm25_pollution_intensity"] = panel["ln_pm25"] - panel["ln_ntl"]
    panel["country_year"] = panel["GID_0"] + "_" + panel["year"].astype(str)

    panel = panel.dropna(subset=["ntl_mean", "pm25_mean"]).copy()
    panel = panel.loc[valid_gid_mask(panel["GID_2"])].copy()
    return panel


def run_regressions(panel: pd.DataFrame) -> pd.DataFrame:
    panel_idx = panel.set_index(["GID_2", "year"])
    specs = [
        ("Lag 0", "birth_region_leader"),
        ("Lag 1", "brl_lag1"),
        ("Lag 2", "brl_lag2"),
    ]

    rows: list[dict[str, object]] = []
    for lag_label, var in specs:
        model = PanelOLS.from_formula(
            f"pm25_pollution_intensity ~ {var} + EntityEffects",
            data=panel_idx,
            other_effects=panel_idx["country_year"],
            drop_absorbed=True,
        )
        res = model.fit(cov_type="clustered", clusters=panel_idx["spell_id"])
        coef = float(res.params[var])
        se = float(res.std_errors[var])
        pval = float(res.pvalues[var])
        rows.append(
            {
                "Spec": lag_label,
                "coef": coef,
                "se": se,
                "pval": pval,
                "nobs": int(res.nobs),
                "treated": int(panel[var].sum()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    panel = build_analysis_panel()
    results = run_regressions(panel)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_PATH, index=False)

    print("PM2.5 pollution-intensity panel")
    print(
        f"Obs: {len(panel):,} | Regions: {panel['GID_2'].nunique():,} | "
        f"Years: {panel['year'].min()}-{panel['year'].max()}"
    )
    print(
        "Outcome: log(PM2.5 + 0.01) - log(Nightlights + 0.01) "
        "(negative = cleaner growth in leader birth regions)\n"
    )
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved results to {OUT_PATH}")


if __name__ == "__main__":
    main()
