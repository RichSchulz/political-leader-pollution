import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WDI_PATH = DATA / "wdi_resource_rents.csv"
SSA = set("AGO BDI BEN BFA BWA CAF CIV CMR COD COG COM CPV DJI ERI ETH GAB GHA GIN GMB GNB GNQ KEN LBR LSO MDG MLI MOZ MRT MUS MWI NAM NER NGA RWA SDN SEN SLE SOM SSD STP SWZ TCD TGO TZA UGA ZAF ZMB ZWE".split())

def load_mineral_rents():
    rents = pd.read_csv(WDI_PATH)
    rents_92_13 = rents[(rents["year"] >= 1992) & (rents["year"] <= 2013)]
    avg_rents = rents_92_13.groupby("country_code")["mineral_rents_gdp"].mean().reset_index()
    q75_min = avg_rents["mineral_rents_gdp"].quantile(0.75)
    avg_rents["quart_min"] = (avg_rents["mineral_rents_gdp"] > q75_min).astype(int)
    avg_rents = avg_rents.rename(columns={"country_code": "GID_0"})
    return avg_rents

def run_regression(subset_panel, label):
    if subset_panel.empty or subset_panel["birth_region_leader"].nunique() < 2:
        return {"Sample": label, "Coef": np.nan, "SE": np.nan, "p-value": np.nan, "N": 0}
        
    model = PanelOLS.from_formula(
        "ln_ntl ~ birth_region_leader + EntityEffects",
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

    ntl_source = load_ntl()
    plad_fixed, birth_gid = load_plad()
    rent_flags = load_mineral_rents()
    
    panel = build_panel(ntl_source, plad_fixed, birth_gid, 1992, 2013)
    panel = panel.merge(rent_flags, on="GID_0", how="inner")
    panel["is_ssa"] = panel["GID_0"].isin(SSA).astype(int)
    panel_idx = panel.set_index(["GID_2", "year"])
    
    results = []
    
    # 1. All countries
    results.append(run_regression(panel_idx, "All countries"))
    
    # 2. Excluding top-quartile mineral-rent countries
    results.append(run_regression(panel_idx[panel_idx["quart_min"] == 0], "Excluding top-quartile mineral-rent countries"))
    
    # 3. Only top-quartile mineral-rent countries
    results.append(run_regression(panel_idx[panel_idx["quart_min"] == 1], "Only top-quartile mineral-rent countries"))
    
    # 4. SSA only
    results.append(run_regression(panel_idx[panel_idx["is_ssa"] == 1], "SSA only"))
    
    # 5. SSA excluding top-quartile mineral-rent countries
    results.append(run_regression(panel_idx[(panel_idx["is_ssa"] == 1) & (panel_idx["quart_min"] == 0)], "SSA excluding top-quartile mineral-rent countries"))
    
    # 6. SSA top-quartile mineral-rent countries
    results.append(run_regression(panel_idx[(panel_idx["is_ssa"] == 1) & (panel_idx["quart_min"] == 1)], "SSA top-quartile mineral-rent countries"))
    
    # 7. Non-SSA top-quartile mineral-rent countries
    results.append(run_regression(panel_idx[(panel_idx["is_ssa"] == 0) & (panel_idx["quart_min"] == 1)], "Non-SSA top-quartile mineral-rent countries"))
    
    # 8. Non-SSA excluding top-quartile mineral-rent countries
    results.append(run_regression(panel_idx[(panel_idx["is_ssa"] == 0) & (panel_idx["quart_min"] == 0)], "Non-SSA excluding top-quartile mineral-rent countries"))
    
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    res_df.to_csv(DATA / "longrun_mineral_table.csv", index=False)

if __name__ == "__main__":
    main()
