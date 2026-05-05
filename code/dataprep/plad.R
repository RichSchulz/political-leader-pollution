# PLAD pre-processing

# Clear environment
rm(list = ls())

# Load packages
require("data.table")

# Load plad data
# plad_id: leader id
# startdate: office entry
# enddate: office exit
# gid_0: birth place gadm 3.6 adm0 alpha code
# gid_1: birth place gadm 3.6 adm1 alpha code
# gid_2: birth place gadm 3.6 adm2 alpha code
plad <- fread("../data/raw/plad/PLAD_April_2024.tab", select = c("plad_id", "idacr", "startdate", "enddate", "gid_0", "gid_1", "gid_2", "foreign_leader"))

# Recode missing values (somehow not supported in fread for tab-separated data)
character_cols <- names(plad)[vapply(plad, is.character, logical(1L), USE.NAMES = F)]
plad[, (character_cols) := lapply(.SD, function(x) fifelse(x == ".", NA_character_, x)), .SDcols = character_cols]
rm(character_cols)
plad[, foreign_leader := as.logical(as.integer(foreign_leader))]

# Drop foreign leaders
plad <- plad[foreign_leader == F, .SD, .SDcols = !"foreign_leader"]

# Drop east germany, south sudan, yugoslavia, and yemen
plad <- plad[!.(c("GDR", "SSD", "YPR", "YUG")), on = "idacr"]
plad[, idacr := NULL]

# Drop leaders with unknown adm0, adm1, or adm2 birth regions
plad <- plad[!is.na(gid_0) & !is.na(gid_1) & !is.na(gid_2),]

# Create unbalanced panel
leader_year <- mapply(function(plad_id, startdate, enddate) {
  gov <- data.table(year = year(startdate):year(enddate))
  gov[, plad_id := plad_id]
  return(gov)
}, plad[["plad_id"]], plad[["startdate"]], plad[["enddate"]], SIMPLIFY = F, USE.NAMES = F) |>
  rbindlist()

# Merge other variables to panel structure
plad[, c("startdate", "enddate") := NULL]
plad <- plad[leader_year, on = "plad_id"]
rm(leader_year)

# Add tenure vars
setkey(plad, plad_id, year)
plad <- plad[, total_tenure := .N, by = "plad_id"]
plad[, plad_id := NULL]

# Subset to 1989 - 2023 time frame
plad <- plad[year %between% c(1989L, 2023L),]

# Denote gid columns
gid_cols <- paste0("GID_", 0:2)

# Rename regional identifiers
setnames(plad, paste0("gid_", 0:2), gid_cols)

# Keep only the leader with the longest total tenure per birth region-year
plad <- plad[plad[, .I[which.max(total_tenure)], by = c(gid_cols, "year")][["V1"]]]
plad[, total_tenure := NULL]

# Balance panel
plad[, is_birthregion := T]
regions <- plad[, ..gid_cols] |> 
  unique()
n_regions <- nrow(regions)
years <- 1989:2023
regions_years <- regions[rep.int(1:n_regions, length(years)),]
regions_years[, year := rep(years, each = n_regions)]
plad <- plad[regions_years, on = c(gid_cols, "year")]
rm(regions, years, regions_years, n_regions)
plad[is.na(is_birthregion), is_birthregion := F]

# Write to disk
fwrite(plad, "../data/interim/plad_adm_2.csv.gz")
