# Zonal statistics using nighttime lights

# Clear environment
rm(list = ls())

# Load packages
require("terra")
require("data.table")
require("mirai")

# List nighttime light files
lights <- rbind(
  # Li
  data.table(file_path = list.files("../data/raw/nighttime_light/li/9828827", "Harmonized_.+[.]tif$", full.names = T), author = "li")[,
    c("year", "type") := list(
      gsub(".+NTL_", "", file_path) |> substr(1L, 4L) |> as.integer(),
      gsub(".+_", "", file_path) |> gsub("[.]tif$", "", x = _)
    )],
  # Chiovelli
  data.table(file_path = list.files("../data/raw/nighttime_light/chiovelli/bloomtopcode_fix", full.names = T), author = "chiovelli")[,
    c("year", "type") := list(
      gsub(".+DMSP", "", file_path) |> substr(1L, 4L) |> as.integer(),
      gsub(".+_", "", file_path) |> gsub("[.]tif$", "", x = _)
    )],
  # Chen
  data.table(file_path = list.files("../data/raw/nighttime_light/chen/22262545", "^SRUNet", full.names = T, recursive = T), author = "chen")[,
    c("year", "type") := list(
      gsub(".+_", "", file_path) |> substr(1L, 4L) |> as.integer(),
      "SRUNet"
    )],
  # Nechaev
  data.table(file_path = list.files("../data/raw/nighttime_light/nechaev/dvnl", full.names = T), author = "nechaev")[,
    c("year", "type") := list(
      gsub(".+_", "", file_path) |> substr(1L, 4L) |> as.integer(),
      "DVNL"
    )],
  # DMSP OLS
  data.table(file_path = list.files("../data/raw/nighttime_light/ols", "[.]stable_lights[.]avg_vis[.]tif$", full.names = T, recursive = T),
    author = "ols")[,
    c("year", "type") := list(
      gsub(".*/F\\d{2}", "", file_path) |> substr(1L, 4L) |> as.integer(),
      gsub(".*/", "", file_path) |> substr(1L, 3L)
    )]
)

# Subset dmsp ols to latest satellite per year
ols <- lights[author == "ols",]
ols[, sat_num := substr(type, 2L, 3L) |> as.integer()]
ols <- ols[ols[, .I[which.max(sat_num)], by = "year"][["V1"]], .SD, .SDcols = !"sat_num"]
lights <- lights[author != "ols",] |> 
  rbind(ols)
rm(ols)

# Write grids list to disk
fwrite(lights, "../data/interim/lights_grid_list.csv.gz")

# Initialize parallel workers
daemons(10L)
everywhere(vapply(c("terra", "data.table"), require, logical(1L), character.only = T))

# Obtain nighttime lights stats by region
lights_adm <- mirai_map(1:nrow(lights), function(i, lights) {
  # Load light grid
  light_i <- lights[i, "file_path"][["file_path"]] |> 
    rast()
  # Load boundaries (done inside loop to avoid terra serialization)
  adm <- vect("../data/raw/gadm36/gadm36_2.shp", what = "geoms")
  # Compute mean light output per adm region
  light_i <- data.table(
    light_mean = zonal(light_i, adm, fun = "mean", na.rm = T)[[1L]]
  )
  # Add grid number
  light_i[, grid := i]
  return(light_i)
}, .args = list(lights = lights[, "file_path"]))[.progress] |> 
  rbindlist()

# Close parallel workers
daemons(0L)

# Add grid meta data
lights_adm[, c("year", "author") := lights[grid, c("year", "author")]]
lights_adm[, grid := NULL]

# Add adm region identifiers
# Load adm identifiers
adm <- vect("../data/raw/gadm36/gadm36_2.shp", what = "attributes")[, c("GID_0", "GID_1", "GID_2")]
lights_adm[, c("GID_0", "GID_1", "GID_2") := adm[rep.int(1:nrow(adm), nrow(lights)),]]
fwrite(lights_adm, "../data/interim/lights_adm_2.csv.gz")
