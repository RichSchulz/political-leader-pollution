import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WDI_PATH = DATA / "wdi_resource_rents.csv"
VDEM_PATH = DATA / "vdem" / "V-Dem-CY-Core-v15.csv"

def load_mineral_rents():
    rents = pd.read_csv(WDI_PATH)
    rents_92_13 = rents[(rents["year"] >= 1992) & (rents["year"] <= 2013)]
    avg_rents = rents_92_13.groupby("country_code")["mineral_rents_gdp"].mean().reset_index()
    q75_min = avg_rents["mineral_rents_gdp"].quantile(0.75)
    avg_rents["quart_min"] = (avg_rents["mineral_rents_gdp"] > q75_min).astype(int)
    avg_rents = avg_rents.rename(columns={"country_code": "GID_0"})
    return avg_rents

def load_vdem():
    vdem = pd.read_csv(VDEM_PATH, usecols=["country_text_id", "year", "v2x_polyarchy"])
    vdem = vdem.rename(columns={"country_text_id": "GID_0", "v2x_polyarchy": "democracy"})
    return vdem

def run_regression(subset_panel, label, formula="ln_ntl ~ birth_region_leader + EntityEffects"):
    if subset_panel.empty or "birth_region_leader" not in subset_panel.columns or subset_panel["birth_region_leader"].nunique() < 2:
        return {"Sample": label, "Coef": np.nan, "SE": np.nan, "p-value": np.nan, "N": 0}
        
    model = PanelOLS.from_formula(
        formula,
        data=subset_panel,
        other_effects=subset_panel["country_year"],
        drop_absorbed=True,
    )
    res = model.fit(cov_type="clustered", clusters=subset_panel["spell_id"])
    
    return {
        "Sample": label,
        "Coef": res.params["birth_region_leader"],
        "SE": res.std_errors["birth_region_leader"],
        "p-value": res.pvalues["birth_region_leader"],
        "N": int(res.nobs)
    }

def main():
    import sys
    sys.path.append(str(ROOT / "scripts"))
    from run_resource_mechanism import load_ntl, load_plad, build_panel

    print("Loading data...")
    ntl_source = load_ntl()
    plad_fixed, birth_gid = load_plad()
    rent_flags = load_mineral_rents()
    vdem = load_vdem()
    
    print("Building panel...")
    panel = build_panel(ntl_source, plad_fixed, birth_gid, 1992, 2013)
    panel = panel.merge(rent_flags, on="GID_0", how="inner")
    panel = panel.merge(vdem, on=["GID_0", "year"], how="left")
    
    # Calculate tenure
    # First, let's fix spell_id to exactly match plad_fixed indices
    plad_fixed["spell_id_str"] = plad_fixed.index.astype(str)
    panel["spell_id_str"] = panel["spell_id"].astype(str).str.replace(".0", "", regex=False)
    
    plad_spell_starts = plad_fixed[["spell_id_str", "startyear"]].copy()
    panel = panel.merge(plad_spell_starts, on="spell_id_str", how="left")
    
    panel["tenure"] = panel["year"] - panel["startyear"] + 1
    # For non-leaders, set tenure to 0
    panel["tenure"] = panel["tenure"].fillna(0)
    panel["long_tenure"] = (panel["tenure"] >= 5).astype(int) # 5+ years
    
    # Define low democracy
    med_democ = panel["democracy"].median()
    panel["low_democ"] = (panel["democracy"] < med_democ).astype(int)
    panel["autocracy"] = (panel["democracy"] < 0.3).astype(int)

    # Restrict to mineral rent quartile for the main heterogeneity tests
    mineral_panel = panel[panel["quart_min"] == 1].copy()
    mineral_idx = mineral_panel.set_index(["GID_2", "year"])
    
    print("\n--- 1. Democracy / Accountability (Top Quartile Mineral Rents, 1992-2013) ---")
    res_democ = []
    res_democ.append(run_regression(mineral_idx[mineral_idx["autocracy"] == 1], "Autocracy (v2x_polyarchy < 0.3)"))
    res_democ.append(run_regression(mineral_idx[mineral_idx["autocracy"] == 0], "Non-Autocracy (v2x_polyarchy >= 0.3)"))
    df_democ = pd.DataFrame(res_democ)
    print(df_democ.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # Interaction test
    mineral_idx_sub = mineral_idx.dropna(subset=["autocracy"]).copy()
    mineral_idx_sub["br_autocracy"] = mineral_idx_sub["birth_region_leader"] * mineral_idx_sub["autocracy"]
    
    model_int1 = PanelOLS.from_formula(
        "ln_ntl ~ birth_region_leader + br_autocracy + EntityEffects", 
        data=mineral_idx_sub, 
        other_effects=mineral_idx_sub["country_year"], 
        drop_absorbed=True
    )
    res_int1 = model_int1.fit(cov_type="clustered", clusters=mineral_idx_sub["spell_id"])
    print("\nInteraction with Autocracy (< 0.3):")
    print(f"Base (Non-Autoc):  {res_int1.params['birth_region_leader']:.4f} (p={res_int1.pvalues['birth_region_leader']:.4f})")
    print(f"Autocracy Diff:    {res_int1.params['br_autocracy']:.4f} (p={res_int1.pvalues['br_autocracy']:.4f})")
    print(f"Total Autoc Coef:  {res_int1.params['birth_region_leader'] + res_int1.params['br_autocracy']:.4f}")

    print("\n--- 2. Leader Tenure (Top Quartile Mineral Rents, 1992-2013) ---")
    
    # Since tenure is defined dynamically, we can't easily split the sample and compare treating the base group.
    # Instead, we split the treatment dummy:
    mineral_idx["br_long_tenure"] = ((mineral_idx["birth_region_leader"] == 1) & (mineral_idx["long_tenure"] == 1)).astype(int)
    mineral_idx["br_short_tenure"] = ((mineral_idx["birth_region_leader"] == 1) & (mineral_idx["long_tenure"] == 0)).astype(int)
    
    model_tenure = PanelOLS.from_formula(
        "ln_ntl ~ br_short_tenure + br_long_tenure + EntityEffects", 
        data=mineral_idx, 
        other_effects=mineral_idx["country_year"], 
        drop_absorbed=True
    )
    res_tenure = model_tenure.fit(cov_type="clustered", clusters=mineral_idx["spell_id"])
    print("Base Favoritism by Tenure Phase:")
    print(f"Short Tenure (< 5 yrs): {res_tenure.params['br_short_tenure']:.4f} (p={res_tenure.pvalues['br_short_tenure']:.4f})")
    print(f"Long Tenure (>= 5 yrs): {res_tenure.params['br_long_tenure']:.4f} (p={res_tenure.pvalues['br_long_tenure']:.4f})")

if __name__ == "__main__":
    main()
