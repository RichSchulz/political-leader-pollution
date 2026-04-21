"""
Run autocracy-only nightlights first-stage regressions for selected windows.

This mirrors the replication notebook's first-stage setup:
- outcome: log(nightlights + 0.01)
- region FE + country-year FE
- leader-period clustering

and restricts the sample to country-years with V-Dem polyarchy < 0.3.
"""

from __future__ import annotations

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

PLAD_PATH = DATA / "political leaders" / "PLAD_April_2024.dta"
DMSP_NTL_CACHE = DATA / "nightlights_adm2_panel.parquet"
VDEM_PATH = DATA / "vdem" / "V-Dem-CY-Core-v15.csv"
OUT_PATH = DATA / "autocracy_first_stage_results.csv"

AUT_THRESHOLD = 0.3
WINDOWS = [
    ("DMSP full window", 1992, 2013),
    ("DMSP later window", 2005, 2013),
]


def load_ntl() -> pd.DataFrame:
    assert DMSP_NTL_CACHE.exists(), f"Nightlights panel not found at {DMSP_NTL_CACHE}"
    ntl = pd.read_parquet(DMSP_NTL_CACHE)
    ntl = ntl.loc[valid_gid_mask(ntl["GID_2"])].copy()
    return ntl


def load_plad() -> tuple[pd.DataFrame, str]:
    assert PLAD_PATH.exists(), f"PLAD data not found at {PLAD_PATH}"
    plad = pd.read_stata(PLAD_PATH)
    plad = plad[plad["foreign_leader"] == "0"].copy()
    birth_gid = "gid_2" if "gid_2" in plad.columns else "gid_1"
    plad = plad.loc[valid_gid_mask(plad[birth_gid])].copy()
    plad["startyear"] = plad["startyear"].astype(int)
    plad["endyear"] = plad["endyear"].astype(int)
    plad = plad[plad["archigos_id"].str.strip() != "."].copy()
    plad = plad.sort_values(["gid_0", "startyear"]).reset_index(drop=True)

    # Match replication notebook overlap fix.
    plad_fixed = plad.copy().reset_index(drop=True)
    for gid_0, group in plad_fixed.groupby("gid_0"):
        idxs = group.index.tolist()
        for i in range(len(idxs) - 1):
            curr, nxt = idxs[i], idxs[i + 1]
            if plad_fixed.loc[curr, "endyear"] >= plad_fixed.loc[nxt, "startyear"]:
                plad_fixed.loc[curr, "endyear"] = plad_fixed.loc[nxt, "startyear"] - 1
    return plad_fixed, birth_gid


def load_vdem(start_year: int, end_year: int) -> pd.DataFrame:
    assert VDEM_PATH.exists(), f"V-Dem data not found at {VDEM_PATH}"
    vdem = pd.read_csv(VDEM_PATH, usecols=["country_text_id", "year", "v2x_polyarchy"])
    vdem = vdem.rename(columns={"country_text_id": "GID_0", "v2x_polyarchy": "democracy"})
    vdem = vdem.dropna(subset=["democracy"])
    vdem = vdem[(vdem["year"] >= start_year) & (vdem["year"] <= end_year)].copy()
    return vdem


def build_panel(
    ntl_source: pd.DataFrame,
    plad_fixed: pd.DataFrame,
    birth_gid: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    base = ntl_source.loc[
        (ntl_source["year"] >= start_year) & (ntl_source["year"] <= end_year)
    ].copy()
    base = base.loc[valid_gid_mask(base["GID_2"])].copy()

    rows: list[dict[str, object]] = []
    for idx, row in plad_fixed.iterrows():
        spell_start = max(int(row["startyear"]), start_year)
        spell_end = min(int(row["endyear"]), end_year)
        if spell_start > spell_end:
            continue
        for year in range(spell_start, spell_end + 1):
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

    if birth_gid == "gid_1":
        adm1_to_adm2 = base[["GID_2", "GID_1"]].drop_duplicates()
        leader_years = (
            leader_years.rename(columns={"GID_2": "GID_1"})
            .merge(adm1_to_adm2, on="GID_1", how="inner")
            [["GID_2", "GID_0", "year", "spell_id"]]
        )

    leader_years = leader_years.drop_duplicates(subset=["GID_2", "year"])
    leader_years["birth_region_leader"] = 1

    spell_map = leader_years[["GID_0", "year", "spell_id"]].drop_duplicates(
        subset=["GID_0", "year"]
    )

    panel = base.merge(
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
    panel["country_year"] = panel["GID_0"] + "_" + panel["year"].astype(str)
    panel = panel.dropna(subset=["ntl_mean"]).copy()
    return panel


def run_window(window_label: str, start_year: int, end_year: int, ntl_source: pd.DataFrame, plad_fixed: pd.DataFrame, birth_gid: str) -> pd.DataFrame:
    panel = build_panel(ntl_source, plad_fixed, birth_gid, start_year, end_year)
    vdem = load_vdem(start_year, end_year)
    panel = panel.merge(vdem, on=["GID_0", "year"], how="left").dropna(subset=["democracy"])
    panel = panel[panel["democracy"] < AUT_THRESHOLD].copy()

    panel_idx = panel.set_index(["GID_2", "year"])
    specs = [
        ("Lag 0", "birth_region_leader"),
        ("Lag 1", "brl_lag1"),
        ("Lag 2", "brl_lag2"),
    ]

    rows: list[dict[str, object]] = []
    for spec_label, var in specs:
        model = PanelOLS.from_formula(
            f"ln_ntl ~ {var} + EntityEffects",
            data=panel_idx,
            other_effects=panel_idx["country_year"],
            drop_absorbed=True,
        )
        res = model.fit(cov_type="clustered", clusters=panel_idx["spell_id"])
        rows.append(
            {
                "window": window_label,
                "start_year": start_year,
                "end_year": end_year,
                "threshold": AUT_THRESHOLD,
                "spec": spec_label,
                "coef": float(res.params[var]),
                "se": float(res.std_errors[var]),
                "pval": float(res.pvalues[var]),
                "nobs": int(res.nobs),
                "treated": int(panel[var].sum()),
                "clusters": int(panel["spell_id"].nunique()),
                "regions": int(panel["GID_2"].nunique()),
                "countries": int(panel["GID_0"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ntl_source = load_ntl()
    plad_fixed, birth_gid = load_plad()

    results = []
    for label, start_year, end_year in WINDOWS:
        results.append(run_window(label, start_year, end_year, ntl_source, plad_fixed, birth_gid))

    out = pd.concat(results, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Autocracy-only first stage (v2x_polyarchy < {AUT_THRESHOLD})")
    print("Dep. var: log(nightlights + 0.01)")
    print("FE: Region (ADM2) + Country x Year  |  Clustering: Leader-period\n")
    print(out.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f"\nSaved results to {OUT_PATH}")


if __name__ == "__main__":
    main()
