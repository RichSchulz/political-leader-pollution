import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

print("Loading PLAD...")
plad = pd.read_parquet("data/political leaders/PLAD_April_2024_gadm41.parquet")
plad = plad[plad["foreign_leader"] == "0"].copy()
plad = plad[~plad["idacr"].isin(["GDR", "SSD", "YPR", "YUG"])].copy()
plad = plad[plad["gid_2"].str.strip() != "."].copy()
plad["startyear"] = plad["startyear"].astype(int)
plad["endyear"] = plad["endyear"].astype(int)

# Expand plad to region-years
rows = []
for _, row in plad.iterrows():
    for y in range(row["startyear"], row["endyear"] + 1):
        rows.append({
            "leader": row["leader"],
            "gid_0": row["gid_0"],
            "gid_1": row["gid_1"],
            "gid_2": row["gid_2"],
            "year": y,
        })
plad_yr = pd.DataFrame(rows)
plad_yr["total_tenure"] = plad_yr.groupby("leader")["year"].transform("count")
plad_yr = plad_yr[(plad_yr["year"] >= 1989) & (plad_yr["year"] <= 2023)].copy()
plad_yr = plad_yr.sort_values("total_tenure", ascending=False).drop_duplicates(["gid_0", "gid_1", "gid_2", "year"])

regions = plad_yr[["gid_0", "gid_1", "gid_2"]].drop_duplicates()
years = list(range(1989, 2024))
regions_years = regions.assign(key=1).merge(pd.DataFrame({"year": years, "key": 1}), on="key").drop("key", axis=1)

plad_yr["is_birthregion"] = 1
plad_yr = regions_years.merge(plad_yr[["gid_0", "gid_1", "gid_2", "year", "is_birthregion"]], on=["gid_0", "gid_1", "gid_2", "year"], how="left")
plad_yr["is_birthregion"] = plad_yr["is_birthregion"].fillna(0)

print("Loading nightlights...")
df_ols = pd.read_parquet("data/nightlights_adm2_panel.parquet") # 1992-2013
df_ols = df_ols.rename(columns={"GID_0": "gid_0", "GID_1": "gid_1", "GID_2": "gid_2"})

# Build full comb panel of all regions from df_ols * years 1989-2023
all_regions = df_ols[["gid_0", "gid_1", "gid_2"]].drop_duplicates()
comb = all_regions.assign(key=1).merge(pd.DataFrame({"year": years, "key": 1}), on="key").drop("key", axis=1)

# merge plad
comb = comb.merge(plad_yr[["gid_0", "gid_1", "gid_2", "year", "is_birthregion"]], on=["gid_0", "gid_1", "gid_2", "year"], how="left")
comb["is_birthregion"] = comb["is_birthregion"].fillna(0)

cntry_had_leader = comb.groupby("gid_0")["is_birthregion"].max()
cntry_had_leader = cntry_had_leader[cntry_had_leader == 1].index
comb = comb[comb["gid_0"].isin(cntry_had_leader)].copy()

any_leader = comb.groupby(["gid_0", "year"])["is_birthregion"].max().reset_index().rename(columns={"is_birthregion": "any_leader"})
comb = comb.merge(any_leader, on=["gid_0", "year"], how="left")
comb.loc[comb["any_leader"] == 0, "is_birthregion"] = np.nan

# SHIFT
comb = comb.sort_values(["gid_2", "year"])
comb["brl_lag1"] = comb.groupby("gid_2")["is_birthregion"].shift(1)

# Now merge lights
comb = comb.merge(df_ols[["gid_2", "year", "ntl_mean"]], on=["gid_2", "year"], how="left")

comb = comb.dropna(subset=["ntl_mean"])
comb["ln_ntl"] = np.log(comb["ntl_mean"] + 0.01)
comb["country_year"] = comb["gid_0"] + "_" + comb["year"].astype(str)

reg_df = comb.dropna(subset=["brl_lag1", "ln_ntl"]).copy()

print("OLS N:", len(reg_df))
print("Treated sum:", reg_df["brl_lag1"].sum())
reg_df = reg_df.set_index(["gid_2", "year"])
model = PanelOLS.from_formula("ln_ntl ~ brl_lag1 + EntityEffects", data=reg_df, other_effects=reg_df["country_year"], drop_absorbed=True)
res = model.fit(cov_type="clustered", clusters=reg_df["gid_0"])
print(res)
