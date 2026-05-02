import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

from run_resource_mechanism import load_ntl, load_plad, build_panel

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WDI_PATH = DATA / "wdi_resource_rents.csv"

def load_all_rents():
    rents = pd.read_csv(WDI_PATH)
    rents_92_13 = rents[(rents["year"] >= 1992) & (rents["year"] <= 2013)]
    avg_rents = rents_92_13.groupby("country_code")[["total_resource_rents_gdp", "oil_rents_gdp", "mineral_rents_gdp"]].mean().reset_index()
    
    # 1. Baseline median
    q50_tot = avg_rents["total_resource_rents_gdp"].median()
    avg_rents["med_tot"] = (avg_rents["total_resource_rents_gdp"] > q50_tot).astype(int)
    
    # 2. Top-tercile total
    q66_tot = avg_rents["total_resource_rents_gdp"].quantile(0.6666)
    avg_rents["terc_tot"] = (avg_rents["total_resource_rents_gdp"] > q66_tot).astype(int)
    
    # 3. Top-quartile total
    q75_tot = avg_rents["total_resource_rents_gdp"].quantile(0.75)
    avg_rents["quart_tot"] = (avg_rents["total_resource_rents_gdp"] > q75_tot).astype(int)
    
    # 4. Top-quartile oil
    q75_oil = avg_rents["oil_rents_gdp"].quantile(0.75)
    avg_rents["quart_oil"] = (avg_rents["oil_rents_gdp"] > q75_oil).astype(int)
    
    # 5. Top-quartile mineral
    q75_min = avg_rents["mineral_rents_gdp"].quantile(0.75)
    avg_rents["quart_min"] = (avg_rents["mineral_rents_gdp"] > q75_min).astype(int)
    
    # 6. Pre-2007 average total (Top quartile)
    rents_pre07 = rents[(rents["year"] >= 1992) & (rents["year"] < 2007)]
    avg_pre07 = rents_pre07.groupby("country_code")["total_resource_rents_gdp"].mean().reset_index()
    q75_pre = avg_pre07["total_resource_rents_gdp"].quantile(0.75)
    avg_pre07["quart_pre07_tot"] = (avg_pre07["total_resource_rents_gdp"] > q75_pre).astype(int)
    
    avg_rents = avg_rents.merge(avg_pre07[["country_code", "quart_pre07_tot"]], on="country_code", how="left")
    avg_rents = avg_rents.rename(columns={"country_code": "GID_0"})
    
    return avg_rents

def main():
    print("Loading data...")
    ntl_source = load_ntl()
    plad_fixed, birth_gid = load_plad()
    rent_flags = load_all_rents()
    
    print("Building panel...")
    panel = build_panel(ntl_source, plad_fixed, birth_gid, 1992, 2013)
    panel = panel.merge(rent_flags, on="GID_0", how="inner")
    
    bins = [
        ("1992_1996", 1992, 1996),
        ("1997_2001", 1997, 2001),
        ("2002_2006", 2002, 2006),
        ("2007_2011", 2007, 2011),
        ("2012_2013", 2012, 2013),
    ]

    panel["bin"] = None
    for name, s, e in bins:
        mask = (panel["year"] >= s) & (panel["year"] <= e)
        panel.loc[mask, "bin"] = name
        
    panel = panel.dropna(subset=["bin"])
    
    # Base variables
    for name, _, _ in bins:
        base_var = f"br_{name}"
        panel[base_var] = ((panel["birth_region_leader"] == 1) & (panel["bin"] == name)).astype(int)

    panel_idx = panel.set_index(["GID_2", "year"])
    
    tests = [
        ("Above Median Total Rents", "med_tot"),
        ("Top Tercile Total Rents", "terc_tot"),
        ("Top Quartile Total Rents", "quart_tot"),
        ("Top Quartile Oil Rents", "quart_oil"),
        ("Top Quartile Mineral Rents", "quart_min"),
        ("Pre-2007 Top Quartile Total", "quart_pre07_tot")
    ]
    
    results = []
    
    for label, col in tests:
        print(f"\nTesting: {label}")
        # Create interactions
        test_vars = []
        for name, _, _ in bins:
            base_var = f"br_{name}"
            int_var = f"{base_var}_high"
            panel_idx[int_var] = panel_idx[base_var] * panel_idx[col].fillna(0)
            test_vars.extend([base_var, int_var])
            
        formula = "ln_ntl ~ " + " + ".join(test_vars) + " + EntityEffects"
        
        model = PanelOLS.from_formula(
            formula,
            data=panel_idx,
            other_effects=panel_idx["country_year"],
            drop_absorbed=True,
        )
        res = model.fit(cov_type="clustered", clusters=panel_idx["spell_id"])
        
        # We are most interested in the 2007-2011 interaction and base
        base_2007 = res.params["br_2007_2011"]
        base_se = res.std_errors["br_2007_2011"]
        base_pval = res.pvalues["br_2007_2011"]
        
        diff_2007 = res.params["br_2007_2011_high"]
        diff_se = res.std_errors["br_2007_2011_high"]
        diff_pval = res.pvalues["br_2007_2011_high"]
        
        results.append({
            "Definition": label,
            "LowRent_Base": base_2007,
            "LowRent_p": base_pval,
            "HighRent_Diff": diff_2007,
            "HighRent_Diff_SE": diff_se,
            "HighRent_Diff_p": diff_pval,
            "Total_HighRent": base_2007 + diff_2007
        })
        
    res_df = pd.DataFrame(results)
    print("\n--- RESULTS FOR 2007-2011 ---")
    print(res_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    res_df.to_csv(DATA / "rent_definitions_summary.csv", index=False)

if __name__ == "__main__":
    main()
