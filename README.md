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
3. **Standalone annual VIIRS V2 (2012-2024):** Used for the single-sensor VIIRS checks. Place the annual GeoTIFFs in `data/nightlights/viirs_v21_annual/`. The cache builder expects files named `nightlights.average_viirs.v21_m_500m_s_<YYYY>0101_<YYYY>1231_go_epsg4326_v20250904.tif`.
4. **ACAG SatPM PM2.5 (1998-2024):** Used for the PM2.5 extension. Download the annual NetCDF files named `V5GL06.HybridPM25.Global.<YYYY>01-<YYYY>12.nc` from [SatPM](https://www.satpm.org/v5-gl-06) / [ACAG Box](https://wustl.app.box.com/v/ACAG-V5GL06-GWRPM25/folder/349055735295) and place them in `data/pm25/`.
5. **V-Dem (Varieties of Democracy):** Used for the democracy interaction specification. Download the **Country-Year Core** CSV from the [V-Dem website](https://v-dem.net/data/the-v-dem-dataset/) (free registration required) and place it in the `data/vdem/` directory.
6. **GADM Admin Boundaries:** Downloaded automatically by the reproduction notebook on first run. Cached in `data/gadm/`.

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

## Build Standalone VIIRS Panel (2012-2024)

To build the ADM2 standalone VIIRS cache used for the single-sensor checks, run:

```bash
./venv/bin/python scripts/build_viirs_v21_panel.py
```

The script:
- reads annual VIIRS GeoTIFFs from `data/nightlights/viirs_v21_annual/`
- checkpoints each processed year in `data/tmp_viirs_v21_parts/`
- writes the combined ADM2 panel to `data/nightlights_adm2_viirs_v21_2012_2024_panel.parquet`

Useful option:

```bash
# recompute yearly checkpoints
./venv/bin/python scripts/build_viirs_v21_panel.py --force
```

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

The long DMSP replication still works, but the later-period pooled nightlights effect does not. In the key nested windows, the lag-0 birth-region coefficient is about `0.0128` (`p = 0.030`) in `1992-2013`, `0.0187` (`p = 0.001`) in `1995-2013`, `0.0110` (`p = 0.056`) in `2000-2013`, and then collapses to `0.0017` (`p = 0.768`) in `2005-2013`. The harmonized `2005-2019` lag-0 coefficient is essentially zero at `0.0004` (`p = 0.966`).

That attenuation matters because the pollution exercise is only compelling if a later-period favoritism first stage still exists. In the current estimates, it does not. The autocracy-only first stage also does not rescue the result: the lag-0 coefficient is `0.0111` (`p = 0.574`) in the DMSP `1992-2013` autocracy sample and `0.0076` (`p = 0.552`) in the DMSP `2005-2013` autocracy sample.

On the pollution side, the pooled `2005-2019` ACAG results do not support green favoritism. The pooled lag-0 NO2 coefficient is `0.0308` (`p = 0.045`), and the lag-0 pollution-intensity coefficient `ln(NO2) - ln(Nightlights)` is `0.0333` (`p = 0.063`). The PM2.5 extension points in the same direction: longer windows show positive pollution effects, and the overlapping `2005-2019` PM2.5 pollution-intensity specification is not negative either, with lag 2 equal to `0.0175` (`p = 0.019`).

The targeted subsample search adds one historical concentration result: the long DMSP replication is strongly concentrated in Africa/Sub-Saharan Africa, but that does not become a clean short-window post-2005 rescue. In the full DMSP `1992-2013` window, Africa has a lag-0 coefficient of `0.0428` (`p = 0.009`) and Sub-Saharan Africa has `0.0442` (`p = 0.011`), while non-SSA is small (`0.0053`, `p = 0.370`). In `2005-2013`, Africa and SSA turn negative and insignificant. In the harmonized `2005-2019` panel they are positive again, but imprecise: Africa is `0.0262` (`p = 0.167`) and SSA is `0.0315` (`p = 0.135`). A longer SSA-only harmonized grid strengthens that distinction: the SSA long window stays positive in `1992-2019` at `0.0448` (`p = 0.010`), even though the short SSA `2005-2013` window is near zero and the SSA `2005-2019` estimate remains imprecise.

Crucially, the historical SSA concentration is entirely explained by a commodity-rent fiscal capacity mechanism. The canonical baseline favoritism effect in the full 1992-2013 DMSP sample (`0.0125`, `p = 0.035`) collapses to zero (`0.0038`, `p = 0.526`) when excluding countries in the top quartile of mineral rents. The same is true within Sub-Saharan Africa: when excluding top-quartile mineral-rent economies, the SSA effect vanishes (`0.0090`, `p = 0.659`). The entire classic favoritism effect is driven by the mineral-rent quartile globally (`0.0456`, `p = 0.006`), and specifically by the SSA mineral-rent countries, where the effect is massive (`0.0967`, `p < 0.001`). This indicates that the long-run favoritism result is not a universal law of political economy but a phenomenon localized to mineral-dependent economies riding a commodity boom (peaking in 2007-2011).

Within those mineral-rich economies, the favoritism mechanism displays striking political heterogeneity. It operates entirely independent of democratic institutions: there is no statistically significant difference between strict autocracies (`0.0657`, `p = 0.098`) and non-autocracies (`0.0498`, `p = 0.003`). Discretionary mineral wealth appears to override formal accountability measures. However, the effect is strongly gated by leader consolidation. Leaders in their first four years of tenure show only suggestive favoritism (`0.0313`, `p = 0.060`), but the effect doubles and becomes highly robust once a leader survives into their fifth year in power (`0.0641`, `p = 0.004`), consistent with the time required to plan and direct large-scale regional investment.

The new standalone VIIRS checks add a different later-period pattern. In the single-sensor VIIRS cache, the pooled `2012-2024` lag-0 coefficient is negative at `-0.0287` (`p = 0.010`). The negative estimate is concentrated in the early VIIRS years: `2012-2016` is `-0.0385` (`p = 0.003`), while `2017-2021` is `-0.0060` (`p = 0.674`) and `2022-2024` is `0.0016` (`p = 0.955`). Dropping the first two VIIRS years does not remove the pattern: pooled `2014-2024` is `-0.0354` (`p = 0.002`), and `2014-2016` is `-0.0495` (`p = 0.001`).

That VIIRS-only reversal is not driven by Sub-Saharan Africa. In the standalone `2012-2024` panel, Africa is positive but imprecise at `0.0193` (`p = 0.506`) and SSA is also positive but imprecise at `0.0270` (`p = 0.378`), while non-SSA is clearly negative at `-0.0473` (`p < 0.001`). The same is true in the early VIIRS bin: SSA `2012-2016` is `0.0392` (`p = 0.268`), but non-SSA `2012-2016` is `-0.0633` (`p < 0.001`).

Current headline takeaways:

- The classic nightlights replication works in the long `1992-2013` DMSP sample.
- The historical replication is concentrated in Africa/Sub-Saharan Africa, and long SSA windows remain positive, but the short post-2005 SSA window does not provide a clean rescue.
- The canonical 1992-2013 result and its SSA concentration are entirely driven by countries in the top quartile of mineral rents, indicating a commodity-boom fiscal capacity mechanism rather than a universal law of favoritism.
- Within mineral economies, this favoritism effect is independent of democratic accountability (V-Dem) but heavily gated by leader consolidation (requiring 5+ years of tenure).
- The first stage weakens sharply through later windows and is near zero in `2005-2013` and `2005-2019`.
- The autocracy-only first stage does not recover the result in current runs.
- Pooled NO2 and pollution-intensity results do not support cleaner birth-region growth.
- PM2.5 results are positive in longer windows and do not support green favoritism either.
- The standalone VIIRS-only cache does not revive the classic positive result; it instead produces negative early-period coefficients, especially outside Sub-Saharan Africa.

## Recommended Final Narrative

The evidence still fits an attenuation story better than a green-favoritism story, but it is no longer a simple pooled-collapse story. The long DMSP replication works, short post-2005 pooled windows weaken sharply, long SSA windows remain positive, and standalone VIIRS-only windows turn negative outside Sub-Saharan Africa. Furthermore, the original positive finding is entirely explained by a mineral-rent fiscal capacity mechanism peaking in the late 2000s, rather than reflecting universal distributive politics. The final report therefore treats the main empirical puzzle as instability in the later-period birth-region nightlights effect—driven by the end of the commodity boom—rather than as affirmative evidence that leaders channel cleaner development to their birth regions.

That framing is more defensible than continuing to search for pooled positive results in the later period. The long-window replication remains useful and credible, but the later annual windows appear to be a different empirical environment: short pooled windows weaken, SSA retains some long-window signal, and the pollution estimates are zero to positive rather than negative.

## Final Report

The submission-style manuscript scaffold now lives in `final report/` and is built around the official Overleaf PNAS research-article template structure:

- manuscript source: `final report/main.tex`
- public code repository: <https://github.com/RichSchulz/political-leader-pollution>
- class/style files: `final report/pnas-new.cls` and `final report/pnasresearcharticle.sty`
- bibliography: `final report/references.bib`
- current figure assets: `final report/figures/first_stage_window_grid.png` and `final report/figures/final_subsample_first_stage_plot.png`
- new subsample outputs: `analysis/final_subsample_first_stage.csv`, `analysis/final_subsample_pollution_followthrough.csv`, and `analysis/final_subsample_first_stage_plot.png`

The report is pre-filled with the current replication, subsample-search, and pollution results and is framed around explaining the post-2005 first-stage attenuation.
