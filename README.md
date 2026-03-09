# Political Leader Pollution

Project for the Big Data and Development class.

See the [Preliminary Proposal](proposal/main.pdf).

## Data

This project uses the following datasets:

1.  **Political Leaders (PLAD):** Included in the `data/` folder.
2.  **Harmonized Nighttime Lights (1992-2024):** Due to its size (~1GB), this dataset is not included in the repository. It can be downloaded from Figshare:
    *   **Source:** [Harmonization of DMSP and VIIRS nighttime light data (Li et al., 2020)](https://figshare.com/articles/dataset/Harmonization_of_DMSP_and_VIIRS_nighttime_light_data_from_1992-2018_at_the_global_scale/9828827)
    *   **Instruction:** Download and place the files in the `data/nightlights/` directory.
3.  **V-Dem (Varieties of Democracy):** Used for the democracy interaction specification. Download the **Country-Year Core** CSV from the [V-Dem website](https://v-dem.net/data/the-v-dem-dataset/) (free registration required) and place it in the `data/vdem/` directory.
4.  **GADM Admin Boundaries:** Downloaded automatically by the reproduction notebook on first run. Cached in `data/gadm/`.

## ACAG NO2 Script

To build the proposal-aligned ACAG surface NO2 panel (2005-2019) without using the notebook, run:

```bash
python3 scripts/build_acag_no2_panel.py
```

The script:
- downloads ACAG continent-year NetCDF files from Zenodo
- resumes partial downloads after connection drops
- checkpoints each continent-year in `data/acag_no2_parts/`
- writes the combined ADM2 panel to `data/no2_adm2_acag_panel.parquet`

By default it deletes each raw `.nc` after processing to keep disk use down. Use `--keep-raw` if you want to retain the downloads.
