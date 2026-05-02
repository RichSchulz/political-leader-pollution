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
    # Full period (1992-2013)
    rents_92_13 = rents[(rents["year"] >= 1992) & (rents["year"] <= 2013)]
    avg_rents = rents_92_13.groupby("country_code")["mineral_rents_gdp"].mean().reset_index()
    q75_min = avg_rents["mineral_rents_gdp"].quantile(0.75)
    avg_rents["quart_min"] = (avg_rents["mineral_rents_gdp"] > q75_min).astype(int)
    
    # Pre-period (1992-2006)
    rents_pre07 = rents[(rents["year"] >= 1992) & (rents["year"] < 2007)]
    avg_pre07 = rents_pre07.groupby("country_code")["mineral_rents_gdp"].mean().reset_index()
    q75_min_pre = avg_pre07["mineral_rents_gdp"].quantile(0.75)
    
    # Map back to avg_rents
    pre_map = avg_pre07.set_index("country_code")["mineral_rents_gdp"]
    avg_rents["quart_min_pre07"] = (avg_rents["country_code"].map(pre_map) > q75_min_pre).astype(int)
    
    avg_rents = avg_rents.rename(columns={"country_code": "GID_0"})
    return avg_rents

def main():
    print("Loading data...")
    import sys
    sys.path.append(str(ROOT / "scripts"))
    from run_resource_mechanism import load_ntl, load_plad, build_panel

    ntl_source = load_ntl()
    plad_fixed, birth_gid = load_plad()
    rent_flags = load_mineral_rents()
    
    print("Building panel...")
    panel = build_panel(ntl_source, plad_fixed, birth_gid, 1992, 2013)
    panel = panel.merge(rent_flags, on="GID_0", how="inner")
    panel["is_ssa"] = panel["GID_0"].isin(SSA).astype(int)
    
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
    
    for name, _, _ in bins:
        base_var = f"br_{name}"
        panel[base_var] = ((panel["birth_region_leader"] == 1) & (panel["bin"] == name)).astype(int)

    panel_idx = panel.set_index(["GID_2", "year"])
    
    print("\n--- 1. Country-clustered standard errors for Top Quartile Mineral Rents ---")
    test_vars = []
    for name, _, _ in bins:
        base_var = f"br_{name}"
        int_var = f"{base_var}_high"
        panel_idx[int_var] = panel_idx[base_var] * panel_idx["quart_min"]
        test_vars.extend([base_var, int_var])
        
    formula = "ln_ntl ~ " + " + ".join(test_vars) + " + EntityEffects"
    model = PanelOLS.from_formula(formula, data=panel_idx, other_effects=panel_idx["country_year"], drop_absorbed=True)
    res_ctry = model.fit(cov_type="clustered", clusters=panel_idx["GID_0"])
    print("2007-2011 Differential (country cluster):")
    print(f"Coef: {res_ctry.params['br_2007_2011_high']:.4f}")
    print(f"SE:   {res_ctry.std_errors['br_2007_2011_high']:.4f}")
    print(f"P-val:{res_ctry.pvalues['br_2007_2011_high']:.4f}")
    
    print("\n--- 2. Pre-period (1992-2006) Top-quartile Mineral Rents classification ---")
    test_vars_pre = []
    for name, _, _ in bins:
        base_var = f"br_{name}"
        int_var = f"{base_var}_highpre"
        panel_idx[int_var] = panel_idx[base_var] * panel_idx["quart_min_pre07"]
        test_vars_pre.extend([base_var, int_var])
    formula_pre = "ln_ntl ~ " + " + ".join(test_vars_pre) + " + EntityEffects"
    model_pre = PanelOLS.from_formula(formula_pre, data=panel_idx, other_effects=panel_idx["country_year"], drop_absorbed=True)
    res_pre = model_pre.fit(cov_type="clustered", clusters=panel_idx["spell_id"])
    print("2007-2011 Differential (pre-2007 mineral):")
    print(f"Coef: {res_pre.params['br_2007_2011_highpre']:.4f}")
    print(f"SE:   {res_pre.std_errors['br_2007_2011_highpre']:.4f}")
    print(f"P-val:{res_pre.pvalues['br_2007_2011_highpre']:.4f}")

    print("\n--- 3. Leave-one-country-out for Top Quartile Mineral Rents ---")
    mineral_countries = panel[panel["quart_min"] == 1]["GID_0"].unique()
    positive_count = 0
    sig_10_count = 0
    total_runs = len(mineral_countries)
    for c in mineral_countries:
        subset = panel_idx[panel_idx["GID_0"] != c]
        model_loo = PanelOLS.from_formula(formula, data=subset, other_effects=subset["country_year"], drop_absorbed=True)
        res_loo = model_loo.fit(cov_type="clustered", clusters=subset["spell_id"])
        diff_coef = res_loo.params["br_2007_2011_high"]
        diff_pval = res_loo.pvalues["br_2007_2011_high"]
        if diff_coef > 0:
            positive_count += 1
            if diff_pval < 0.10:
                sig_10_count += 1
    print(f"Leave-one-out for {total_runs} mineral countries:")
    print(f"Positive in {positive_count}/{total_runs}")
    print(f"Significant (p < 0.10) in {sig_10_count}/{total_runs}")

    print("\n--- 4. SSA versus non-SSA within Mineral-rent countries ---")
    panel_min = panel_idx[panel_idx["quart_min"] == 1].copy()
    test_vars_ssa = []
    for name, _, _ in bins:
        base_var = f"br_{name}"
        int_var = f"{base_var}_ssa"
        panel_min[int_var] = panel_min[base_var] * panel_min["is_ssa"]
        test_vars_ssa.extend([base_var, int_var])
    formula_ssa = "ln_ntl ~ " + " + ".join(test_vars_ssa) + " + EntityEffects"
    model_ssa = PanelOLS.from_formula(formula_ssa, data=panel_min, other_effects=panel_min["country_year"], drop_absorbed=True)
    try:
        res_ssa = model_ssa.fit(cov_type="clustered", clusters=panel_min["spell_id"])
        print("Within top quartile mineral rents, 2007-2011:")
        print(f"Base (Non-SSA Mineral) Coef: {res_ssa.params['br_2007_2011']:.4f}, p={res_ssa.pvalues['br_2007_2011']:.4f}")
        print(f"SSA Differential Coef:       {res_ssa.params['br_2007_2011_ssa']:.4f}, p={res_ssa.pvalues['br_2007_2011_ssa']:.4f}")
        print(f"Total SSA Mineral Coef:      {res_ssa.params['br_2007_2011'] + res_ssa.params['br_2007_2011_ssa']:.4f}")
    except ValueError as e:
        print("ValueError running SSA subset within mineral:", e)

if __name__ == "__main__":
    main()
