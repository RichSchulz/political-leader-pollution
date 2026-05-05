import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

DATA = Path("data")
NTL_DIR = DATA / "nightlights"
ADM2_CACHE = DATA / "gadm" / "gadm41_adm2.gpkg"

adm2 = gpd.read_file(ADM2_CACHE)
from gid_utils import valid_gid_mask
adm2 = adm2.loc[valid_gid_mask(adm2["GID_2"])].copy()

rpath = NTL_DIR / "Harmonized_DN_NTL_1992_calDMSP.tif"

print("Computing with nodata=0...")
stats_0 = zonal_stats(adm2.geometry, str(rpath), stats=["mean"], nodata=0)
means_0 = [s["mean"] for s in stats_0]

print("Computing with nodata=None...")
stats_none = zonal_stats(adm2.geometry, str(rpath), stats=["mean"], nodata=None)
means_none = [s["mean"] for s in stats_none]

df = pd.DataFrame({"GID_2": adm2["GID_2"], "mean_nodata0": means_0, "mean_nodata_none": means_none})
print(df.head(20))
print("Mean of means (nodata=0):", df["mean_nodata0"].mean())
print("Mean of means (nodata=None):", df["mean_nodata_none"].mean())
