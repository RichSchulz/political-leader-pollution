# Political Leader Pollution

Project for the Big Data and Development class.

See the [Preliminary Proposal](proposal/main.pdf), the in-progress [final report source](<final report/main.tex>), and the public GitHub repository: <https://github.com/RichSchulz/political-leader-pollution>.

## Project Status

The project began as an environmental extension of the regional-favoritism literature: do leaders direct cleaner growth to their birth regions, not just more growth? The current evidence does not support that "green favoritism" pattern in the post-2005 pollution window.

The main empirical development is instead the attenuation of the nightlights first stage in later years. The long DMSP-era replication still works, but the effect weakens through later windows and is essentially zero once attention shifts to the post-2005 sample that overlaps with the pollution data. That attenuation now drives the final-stage analysis and report framing.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Requirements

This project uses the following datasets:

1. **Political Leaders (PLAD):** Included in the `data/` folder.
2. **Harmonized Nighttime Lights (1992-2024):** Due to size, this dataset is not included in the repository. Download from Figshare:
   - **Source:** [Harmonization of DMSP and VIIRS nighttime light data (Li et al., 2020)](https://figshare.com/articles/dataset/Harmonization_of_DMSP_and_VIIRS_nighttime_light_data_from_1992-2018_at_the_global_scale/9828827)
   - **For this project window (2005-2019):**
     - `Harmonized_DN_NTL_2005_calDMSP.tif` through `Harmonized_DN_NTL_2013_calDMSP.tif`
     - `Harmonized_DN_NTL_2014_simVIIRS.tif` through `Harmonized_DN_NTL_2019_simVIIRS.tif`
   - Place files in `data/nightlights/`.
3. **ACAG SatPM PM2.5 (1998-2024):** Used for the PM2.5 extension. Download the annual NetCDF files named `V5GL06.HybridPM25.Global.<YYYY>01-<YYYY>12.nc` from [SatPM](https://www.satpm.org/v5-gl-06) / [ACAG Box](https://wustl.app.box.com/v/ACAG-V5GL06-GWRPM25/folder/349055735295) and place them in `data/pm25/`.
4. **V-Dem (Varieties of Democracy):** Used for the democracy interaction specification. Download the **Country-Year Core** CSV from the [V-Dem website](https://v-dem.net/data/the-v-dem-dataset/) (free registration required) and place it in the `data/vdem/` directory.
5. **GADM Admin Boundaries:** Downloaded automatically by the reproduction notebook on first run. Cached in `data/gadm/`.

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
- `analysis/final_subsample_search.ipynb`: targeted post-midterm subsample search for places where the nightlights first stage might survive.

`analysis/green_favoritism.ipynb` workflow:
1. Load ADM2 boundaries and build PLAD treatment.
2. Build/load nightlights panel for 2005-2019 (DMSP for 2005-2013, simVIIRS for 2014-2019).
3. Load `data/no2_adm2_acag_panel.parquet`.
4. Run:
   - main Green Favoritism regressions
   - democracy interaction
   - pollution-intensity outcome

`analysis/green_favoritism_pm25.ipynb` workflow:
1. Load `data/pm25_adm2_acag_panel.parquet`.
2. Build PLAD treatment variables.
3. Run:
   - main PM2.5 regressions
   - democracy interaction
   - autocracy-only subsamples
   - PM2.5 event study

For the PM2.5 analogue of the NO2 pollution-intensity outcome on the overlapping `2005-2019` nightlights window, run:

```bash
./venv/bin/python scripts/run_pm25_pollution_intensity.py
```

This writes the regression output to `data/pm25_pollution_intensity_results.csv`.

For the current autocracy-only first-stage check, run:

```bash
./venv/bin/python scripts/run_autocracy_first_stage.py
```

This writes the regression output to `data/autocracy_first_stage_results.csv`.

## Current Results Snapshot

The long DMSP replication still works, but the later-period first stage does not. In the key nested windows, the lag-0 nightlights coefficient is about `0.0128` (`p = 0.030`) in `1992-2013`, `0.0187` (`p = 0.001`) in `1995-2013`, `0.0110` (`p = 0.056`) in `2000-2013`, and then collapses to `0.0017` (`p = 0.768`) in `2005-2013`. The harmonized `2005-2019` lag-0 nightlights coefficient is essentially zero at `0.0004` (`p = 0.966`).

That attenuation matters because the pollution exercise is only compelling if a later-period favoritism first stage still exists. In the current estimates, it does not. The autocracy-only first stage also does not rescue the result: the lag-0 coefficient is `0.0111` (`p = 0.574`) in the DMSP `1992-2013` autocracy sample and `0.0076` (`p = 0.552`) in the DMSP `2005-2013` autocracy sample.

On the pollution side, the pooled `2005-2019` ACAG results do not support green favoritism. The pooled lag-0 NO2 coefficient is `0.0308` (`p = 0.045`), and the lag-0 pollution-intensity coefficient `ln(NO2) - ln(Nightlights)` is `0.0333` (`p = 0.063`). The PM2.5 extension points in the same direction: longer windows show positive pollution effects, and the overlapping `2005-2019` PM2.5 pollution-intensity specification is not negative either, with lag 2 equal to `0.0175` (`p = 0.019`).

The targeted subsample search adds one new result: the historical replication is strongly concentrated in Africa/Sub-Saharan Africa, but that does not rescue the post-2005 window. In the full DMSP `1992-2013` window, Africa has a lag-0 coefficient of `0.0428` (`p = 0.009`) and Sub-Saharan Africa has `0.0442` (`p = 0.011`), while non-SSA is small (`0.0053`, `p = 0.370`). In `2005-2013`, Africa and SSA turn negative and insignificant. In the harmonized `2005-2019` panel they are positive again, but imprecise: Africa is `0.0262` (`p = 0.167`) and SSA is `0.0315` (`p = 0.135`). No post-2005 subsample clears the notebook's pre-specified credible-first-stage rule.

Current headline takeaways:

- The classic nightlights replication works in the long `1992-2013` DMSP sample.
- The historical first stage is concentrated in Africa/Sub-Saharan Africa, but that pattern does not become a statistically credible post-2005 rescue.
- The first stage weakens sharply through later windows and is near zero in `2005-2013` and `2005-2019`.
- The autocracy-only first stage does not recover the result in current runs.
- Pooled NO2 and pollution-intensity results do not support cleaner birth-region growth.
- PM2.5 results are positive in longer windows and do not support green favoritism either.

## Recommended Final Narrative

The evidence now fits an attenuation story better than a green-favoritism story. The final report therefore treats the main empirical puzzle as the post-2005 collapse of the first-stage nightlights effect, not as affirmative evidence that leaders channel cleaner development to their birth regions.

That framing is more defensible than continuing to search for pooled positive results in the later period. The long-window replication remains useful and credible, but the overlapping pollution window appears to be a different empirical environment: later-window favoritism is weaker or harder to detect, and the pollution estimates are zero to positive rather than negative.

## Final Report

The submission-style manuscript scaffold now lives in `final report/` and is built around the official Overleaf PNAS research-article template structure:

- manuscript source: `final report/main.tex`
- public code repository: <https://github.com/RichSchulz/political-leader-pollution>
- class/style files: `final report/pnas-new.cls` and `final report/pnasresearcharticle.sty`
- bibliography: `final report/references.bib`
- current figure assets: `final report/figures/first_stage_window_grid.png` and `final report/figures/final_subsample_first_stage_plot.png`
- new subsample outputs: `analysis/final_subsample_first_stage.csv`, `analysis/final_subsample_pollution_followthrough.csv`, and `analysis/final_subsample_first_stage_plot.png`

The report is pre-filled with the current replication, subsample-search, and pollution results and is framed around explaining the post-2005 first-stage attenuation.
