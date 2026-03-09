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
3.  **V-Dem (Varieties of Democracy):** Used for the democracy interaction specification. Download the **Country-Year Core** CSV from the [V-Dem website](https://v-dem.net/data/the-v-dem-dataset/) (free registration required) and place it in the `data/vdem/` directory.
4.  **GADM Admin Boundaries:** Downloaded automatically by the reproduction notebook on first run. Cached in `data/gadm/`.

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

## Run Analysis

Use `analysis/green_favoritism.ipynb` for the ACAG 2005-2019 analysis.

Notebook workflow:
1. Load ADM2 boundaries and build PLAD treatment.
2. Build/load nightlights panel for 2005-2019 (DMSP for 2005-2013, simVIIRS for 2014-2019).
3. Load `data/no2_adm2_acag_panel.parquet`.
4. Run:
   - main Green Favoritism regressions,
   - democracy interaction,
   - pollution-intensity outcome.

## Results Snapshot (Current)

Current estimates on the 2005-2019 ACAG window do not show the expected Green Favoritism pattern:

- Main pooled spec (`ln(Nightlights)`): near zero effect (Lag 0: -0.0026, p=0.778).
- Main pooled spec (`ln(NO2)`): small positive Lag 0 effect (0.0308, p=0.045).
- Pollution intensity `ln(NO2) - ln(Nightlights)`: positive at Lag 0 (0.0333, p=0.063), not significant at Lag 1.
- Democracy interaction: no statistically meaningful interaction effects in this restricted window.

For reference, the replication-style long DMSP sample (1992-2013) still shows a positive nightlights favoritism coefficient (about 0.0128, p=0.030), but this fades in the restricted 2005+ sample.
