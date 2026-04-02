# Political Leader Pollution

Project for the Big Data and Development class.

See the [Preliminary Proposal](proposal/main.pdf).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Requirements

This project uses the following datasets:

1.  **Political Leaders (PLAD):** Included in the `data/` folder.
2.  **Harmonized Nighttime Lights (1992-2024):** Due to size, this dataset is not included in the repository. Download from Figshare:
    *   **Source:** [Harmonization of DMSP and VIIRS nighttime light data (Li et al., 2020)](https://figshare.com/articles/dataset/Harmonization_of_DMSP_and_VIIRS_nighttime_light_data_from_1992-2018_at_the_global_scale/9828827)
    *   **For this project window (2005-2019):**
        * `Harmonized_DN_NTL_2005_calDMSP.tif` through `Harmonized_DN_NTL_2013_calDMSP.tif`
        * `Harmonized_DN_NTL_2014_simVIIRS.tif` through `Harmonized_DN_NTL_2019_simVIIRS.tif`
    *   Place files in `data/nightlights/`.
3.  **ACAG SatPM PM2.5 (1998-2024):** Used for the PM2.5 extension. Download the annual NetCDF files named `V5GL06.HybridPM25.Global.<YYYY>01-<YYYY>12.nc` from [SatPM](https://www.satpm.org/v5-gl-06) / [ACAG Box](https://wustl.app.box.com/v/ACAG-V5GL06-GWRPM25/folder/349055735295) and place them in `data/pm25/`.
4.  **V-Dem (Varieties of Democracy):** Used for the democracy interaction specification. Download the **Country-Year Core** CSV from the [V-Dem website](https://v-dem.net/data/the-v-dem-dataset/) (free registration required) and place it in the `data/vdem/` directory.
5.  **GADM Admin Boundaries:** Downloaded automatically by the reproduction notebook on first run. Cached in `data/gadm/`.

## Build ACAG NO2 Panel (2005-2019)

To build the proposal-aligned ACAG surface NO2 panel without using the notebook, run:

```bash
./venv/bin/python scripts/build_acag_no2_panel.py
```

The script:
- downloads ACAG continent-year NetCDF files from Zenodo
- resumes partial downloads after connection drops
- checkpoints each continent-year in `data/acag_no2_parts/`
- writes the combined ADM2 panel to `data/no2_adm2_acag_panel.parquet`

By default it deletes each raw `.nc` after processing to keep disk use down. Use `--keep-raw` if you want to retain the downloads.

Useful options:

```bash
# partial run
./venv/bin/python scripts/build_acag_no2_panel.py --start-year 2005 --end-year 2010

# force recompute of existing continent-year checkpoints
./venv/bin/python scripts/build_acag_no2_panel.py --force-process
```

## Build ACAG PM2.5 Panel (1998-2024)

To build the long-run ADM2 PM2.5 panel used by the PM2.5 extension, run:

```bash
./venv/bin/python scripts/build_pm25_panel.py
```

The script:
- reads annual ACAG SatPM NetCDF files from `data/pm25/`
- computes ADM2 zonal means using the cached GADM boundaries in `data/gadm/gadm41_adm2.gpkg`
- writes the combined panel to `data/pm25_adm2_acag_panel.parquet`

Before running it, make sure the annual PM2.5 files for 1998-2024 are present in `data/pm25/`.

## Run Analysis

Use these notebooks for the current analysis workflows:

- `analysis/green_favoritism.ipynb`: ACAG NO2 + nightlights analysis on the 2005-2019 window.
- `analysis/green_favoritism_pm25.ipynb`: PM2.5 extension on the 1998-2024 window.

`analysis/green_favoritism.ipynb` workflow:
1. Load ADM2 boundaries and build PLAD treatment.
2. Build/load nightlights panel for 2005-2019 (DMSP for 2005-2013, simVIIRS for 2014-2019).
3. Load `data/no2_adm2_acag_panel.parquet`.
4. Run:
   - main Green Favoritism regressions,
   - democracy interaction,
   - pollution-intensity outcome.

`analysis/green_favoritism_pm25.ipynb` workflow:
1. Load `data/pm25_adm2_acag_panel.parquet`.
2. Build PLAD treatment variables.
3. Run:
   - main PM2.5 regressions,
   - democracy interaction,
   - autocracy-only subsamples,
   - PM2.5 event study.

For the PM2.5 analogue of the NO2 pollution-intensity outcome on the overlapping `2005-2019` nightlights window, run:

```bash
./venv/bin/python scripts/run_pm25_pollution_intensity.py
```

This writes the regression output to `data/pm25_pollution_intensity_results.csv`.

## Results Snapshot (Current)

Current estimates on the 2005-2019 ACAG window do not show the expected Green Favoritism pattern:

- Main pooled spec (`ln(Nightlights)`): near zero effect (Lag 0: -0.0026, p=0.778).
- Main pooled spec (`ln(NO2)`): small positive Lag 0 effect (0.0308, p=0.045).
- Pollution intensity `ln(NO2) - ln(Nightlights)`: positive at Lag 0 (0.0333, p=0.063), not significant at Lag 1.
- Democracy interaction: no statistically meaningful interaction effects in this restricted window.
- Autocracy-only subsample (`v2x_polyarchy < 0.3`): pollution intensity is positive at Lag 0 (0.1136, p=0.032), which points away from cleaner growth.
- Event-study results: NO2 and pollution-intensity coefficients turn positive around 3-4 years after ascension, but there is still some pre-period movement, so the dynamic evidence is not yet clean.
- PM2.5 extension: the longer windows show positive and statistically significant pollution effects (`1998-2024`: Lag 0 = 0.0078, p=0.011; `1998-2019`: Lag 0 = 0.0091, p=0.002), while the later `2005-2024` window is positive but null.
- PM2.5 pollution intensity `ln(PM2.5) - ln(Nightlights)` on the overlapping `2005-2019` sample is not negative either: Lag 0 = `0.0041` (`p=0.665`), Lag 1 = `0.0122` (`p=0.120`), and Lag 2 = `0.0175` (`p=0.019`).
- Replication diagnostics: the nightlights favoritism coefficient stays positive in `1995-2013` (0.0187, p=0.001) and `2000-2013` (0.0110, p=0.056), then collapses only in `2005-2013` (0.0017, p=0.768).
- Decade interaction check: pooled `BirthRegion × post2000` and `BirthRegion × post2010` interactions are not statistically significant, so the cleaner evidence comes from the nested-window comparison rather than a sharp decade break.

For reference, the replication-style long DMSP sample (1992-2013) still shows a positive nightlights favoritism coefficient (about 0.0128, p=0.030). The new diagnostics sharpen that result: attenuation is not just a “late 1990s” story, because the coefficient remains positive through `2000-2013` and only really breaks in the post-2005 window. Taken together, the PM2.5 results line up more with the NO2 results than against them: neither outcome shows evidence of cleaner birth-region growth, and the stronger positive PM2.5 results appear in the longer windows rather than the later restricted sample.
