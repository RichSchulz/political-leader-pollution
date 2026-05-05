# Population pre-processing

# Clear environment
rm(list = ls())

# Load packages
require("data.table")
require("terra")

# Load boundaries
adm <- vect("../data/raw/gadm36/gadm36_2.shp", what = "geoms")
n_adm <- nrow(adm)
adm$id <- 1:n_adm
adm <- rast("../data/raw/ghs/GHS_POP_E2010_GLOBE_R2023A_4326_3ss_V1_0/GHS_POP_E2010_GLOBE_R2023A_4326_3ss_V1_0.tif") |> 
  rasterize(adm, y = _, field = "id")

# Iterate over years of population grids
pop <- lapply(seq.int(1985L, 2025L, 5L), function(year) {
  pop_year <- paste0("../data/raw/ghs/GHS_POP_E", year, "_GLOBE_R2023A_4326_3ss_V1_0/GHS_POP_E", year, "_GLOBE_R2023A_4326_3ss_V1_0.tif") |>
    rast() |>
    zonal(adm, fun = "sum", na.rm = T)
  setDT(pop_year)
  setnames(pop_year, c("id", "population"))
  pop_year[, id := as.integer(id)]
  pop_year <- pop_year[data.table(id = 1:n_adm), on = "id"]
  pop_year[, year := year]
  return(pop_year)
}) |>
  rbindlist()
rm(adm)

# Interpolate population
pop <- lapply(seq.int(1985L, 2020L, 5L), function(begin_year) {
  begin_pop <- pop[year == begin_year, c("id", "population")]
  setnames(begin_pop, "population", "bpopulation")
  end_pop <- pop[year == begin_year + 5L, c("id", "population")]
  setnames(end_pop, "population", "epopulation")
  begin_pop <- end_pop[begin_pop, on = "id"]
  rm(end_pop)
  begin_pop[, annual_diff := (epopulation - bpopulation) / 5]
  begin_pop[, epopulation := NULL]
  interpolated_pop <- lapply(1:4, function(year_diff) data.table(id = begin_pop[["id"]],
    population = begin_pop[["bpopulation"]] + begin_pop[["annual_diff"]] * year_diff,
    year = begin_year + year_diff)) |>
    rbindlist()
  return(interpolated_pop)
}) |>
  rbindlist() |>
  rbind(pop, use.names = T)
pop[, id := NULL]

# Add regional identifiers
adm <- vect("../data/raw/gadm36/gadm36_2.shp", what = "attributes")
setDT(adm)
adm <- adm[, c("GID_0", "GID_1", "GID_2")]
pop[, c("GID_0", "GID_1", "GID_2") := adm[rep.int(1:n_adm, nrow(pop) / n_adm),]]

# Convert population count to integer type
pop[, population := round(population) |> as.integer()]

# Write adm level population to disk
fwrite(pop, "../data/interim/population_adm_2.csv.gz")
