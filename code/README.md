# Replication Package

This is the replication package of the **Regional favoritism: New data, larger sample, same pattern** paper.

## Directory Structure

The directory structure adheres to the [SoDa Replicator](https://github.com/sodalabsio/soda_replicator) template. Accordingly, we read raw inputs from `../data/raw`, write intermediate data to `../data/interim`, save analysis-ready data to `../data/analysis`, and export tables to `../paper/results/tables`. If you use a different structure, amend the file paths in the code accordingly.

## Software

The code is written in R and C++. So, you do not only have to install [R](https://cloud.r-project.org), but need to make sure that it is able to compile C++. You achieve this by installing build-essential (from your package manager) on Linux, RTools (from the R website) on Windows, and Xcode (from the app store) on macOS.

To install all required R packages, open the repository's directory in R on your computer, install `renv` (`install.packages("renv")`), and restore the environment with `renv::restore()`.

## Data

All input data are publicly available. The links in the subsequent table direct to the respective download sites. To replicate our results, download the data sets and place them the respective `data/raw` directories.

<div align="center">

| Variables | Source | Accessed | Local Directory |
|:----------|:-------|:---------|:----------------|
|Light |[Chen et al. (2024)](https://figshare.com/articles/dataset/A_history_reconstructed_time_series_1992-2011_of_annual_global_NPP-VIIRS-V2-like_nighttime_light_data_through_Super-resolution_U-Net_model/22262545/8) |October 21, 2025 |../data/raw/nighttime_light/chen |
|Light |[Chiovelli et al. (2026)](https://drive.google.com/drive/folders/1cAFp_1Fn3ntwj_Ps3rRU0--srP_UWVgx) |October 21, 2025 |../data/raw/nighttime_light/chiovelli |
|Light |[Li et al. (2020)](https://doi.org/10.6084/m9.figshare.9828827.v10) |October 07, 2025 |../data/raw/nighttime_light/li |
|Light |[Nechaev et al. (2021)](https://eogdata.mines.edu/wwwdata/viirs_products/dvnl)|October 21, 2025 |../data/raw/nighttime_light/nechaev |
|Light |[DMSP-OLS](https://www.ncei.noaa.gov/products/dmsp-operational-linescan-system) |December 17, 2025 |../data/raw/nighttime_light/ols |
|Leader birth regions |[Bomprezzi et al. (2025)](https://doi.org/10.7910/DVN/YUS575) |October 07, 2025 |../data/raw/plad |
|Ethnic fractionalization, Polity2 score, liberal democracy index, schooling, GDP per capita |[Teorell et al. (2026)](https://www.gu.se/en/quality-government/qog-data/data-downloads/standard-dataset) |February 12, 2026 |../data/raw/qog |
|Administrative borders |[GADM 3.6](https://geodata.ucdavis.edu/gadm/gadm3.6/gadm36_levels_shp.zip) |October 07, 2025 |../data/raw/gadm36 |
|Population |[Schiavina et al. (2023)](https://human-settlement.emergency.copernicus.eu/download.php?ds=pop) |December 12 - 16, 2025 |../data/raw/ghs |

</div>

## Script Execution

The R scripts in `dataprep` convert the raw input data to intermediate and analysis-ready outputs. As long as you run `join.R` last, you are free to run the other files in any random order. Do not run `interpolate.cpp` independently. `qog.R` calls that code internally.

The code in the `analysis` folder produces the summary statistics and regression tables in the paper. You may also execute these scripts in any order of your choosing.

## Computational Requirements

`dataprep/nightlights.R` and `dataprep/population.R` run rather expensive geo-spatial computations. Expect execution time of up to multiple days (on smaller devices). All other scripts complete in seconds.
