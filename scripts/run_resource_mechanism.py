import warnings
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from gid_utils import valid_gid_mask
except ModuleNotFoundError:
    from scripts.gid_utils import valid_gid_mask

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

PLAD_PATH = DATA / "political leaders" / "PLAD_April_2024.dta"
DMSP_NTL_CACHE = DATA / "nightlights_adm2_panel.parquet"
WDI_PATH = DATA / "wdi_resource_rents.csv"

VDEM_PATH = DATA / "vdem" / "V-Dem-CY-Core-v15.csv"
AUT_THRESHOLD = 0.3

def load_vdem(start_year: int, end_year: int) -> pd.DataFrame:
    vdem = pd.read_csv(VDEM_PATH, usecols=["country_text_id", "year", "v2x_polyarchy"])
    vdem = vdem.rename(columns={"country_text_id": "GID_0", "v2x_polyarchy": "democracy"})
    vdem = vdem.dropna(subset=["democracy"])
    vdem = vdem[(vdem["year"] >= start_year) & (vdem["year"] <= end_year)].copy()
    return vdem

def load_ntl() -> pd.DataFrame:
    ntl = pd.read_parquet(DMSP_NTL_CACHE)
    ntl = ntl.loc[valid_gid_mask(ntl["GID_2"])].copy()
    return ntl

def load_plad() -> tuple[pd.DataFrame, str]:
    plad = pd.read_stata(PLAD_PATH)
    plad = plad[plad["foreign_leader"] == "0"].copy()
    birth_gid = "gid_2" if "gid_2" in plad.columns else "gid_1"
    plad = plad.loc[valid_gid_mask(plad[birth_gid])].copy()
    plad["startyear"] = plad["startyear"].astype(int)
    plad["endyear"] = plad["endyear"].astype(int)
    plad = plad[plad["archigos_id"].str.strip() != "."].copy()
    plad = plad.sort_values(["gid_0", "startyear"]).reset_index(drop=True)

    plad_fixed = plad.copy().reset_index(drop=True)
    for gid_0, group in plad_fixed.groupby("gid_0"):
        idxs = group.index.tolist()
        for i in range(len(idxs) - 1):
            curr, nxt = idxs[i], idxs[i + 1]
            if plad_fixed.loc[curr, "endyear"] >= plad_fixed.loc[nxt, "startyear"]:
                plad_fixed.loc[curr, "endyear"] = plad_fixed.loc[nxt, "startyear"] - 1
    return plad_fixed, birth_gid

def load_rents() -> pd.DataFrame:
    rents = pd.read_csv(WDI_PATH)
    # Use 1992-2013 average
    rents_period = rents[(rents["year"] >= 1992) & (rents["year"] <= 2013)]
    avg_rents = rents_period.groupby("country_code")["total_resource_rents_gdp"].mean().reset_index()
    avg_rents = avg_rents.dropna(subset=["total_resource_rents_gdp"])
    
    # Split at median
    median_rent = avg_rents["total_resource_rents_gdp"].median()
    avg_rents["high_rent"] = (avg_rents["total_resource_rents_gdp"] > median_rent).astype(int)
    avg_rents = avg_rents.rename(columns={"country_code": "GID_0"})
    return avg_rents

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

    panel["ln_ntl"] = np.log(panel["ntl_mean"] + 0.01)
    panel["country_year"] = panel["GID_0"] + "_" + panel["year"].astype(str)
    panel = panel.dropna(subset=["ntl_mean"]).copy()
    return panel

