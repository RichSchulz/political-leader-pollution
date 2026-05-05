# Join interim data sets

# Clear environment
rm(list = ls())

# Load packages
require("data.table")

# Load country level data
qog <- fread("../data/interim/qog.csv.gz", na.strings = "")
frac <- fread("../data/interim/fractionalization.csv.gz", na.strings = "")
bl <- fread("../data/interim/barro_lee.csv.gz", na.strings = "")

# Load sub-national plad, light, population data
plad <- fread("../data/interim/plad_adm_2.csv.gz", na.strings = "")
lights <- fread("../data/interim/lights_adm_2.csv.gz", na.strings = "")
population <- fread("../data/interim/population_adm_2.csv.gz", na.strings = "")

# Denote gid columns
gid_cols <- paste0("GID_", 0:2)

# Convert light to wide format
lights <- dcast(lights, formula(paste0("year + ", paste0(gid_cols, collapse = " + "), " ~ author")), value.var = "light_mean")
authors <- c("chen", "chiovelli", "li", "nechaev", "ols")
setnames(lights, authors, paste0("light_mean_", authors))
rm(authors)

# Use ols as pre-2013 observations for nechaev
lights[is.na(light_mean_nechaev), light_mean_nechaev := light_mean_ols]

# Join data
comb <- qog[population, nomatch = NULL, on = c("GID_0", "year")]
comb <- lights[comb, on = c(gid_cols, "year")]
comb <- plad[comb, on = c(gid_cols, "year")]
comb[is.na(is_birthregion), is_birthregion := F]

# Subset to 1989 to 2023
comb <- comb[year %between% c(1989L, 2023L),]

# Print and drop countries that are never leader birth regions
cntry_had_leader <- comb[, .(any_leader = any(is_birthregion, na.rm = F)), by = "GID_0"]
paste0("countries without leaders: ", paste0(cntry_had_leader[any_leader == F, "GID_0"][["GID_0"]], collapse = ", ")) |> 
  print()
cntry_had_leader <- cntry_had_leader[any_leader == T, "GID_0"]
comb <- comb[cntry_had_leader, nomatch = NULL, on = "GID_0"]
rm(cntry_had_leader)

# Set birth region indicator for country years without birth region to NA
comb <- comb[, .(any_leader = any(is_birthregion, na.rm = F)), by = c("GID_0", "year")][comb, on = c("GID_0", "year")]
comb[any_leader == F, is_birthregion := NA]
comb[, any_leader := NULL]

# Add pre and post variables
setkeyv(comb, c(gid_cols, "year"))
comb[, c(paste0("f", 1:3), paste0("l", 1:3)) := c(shift(is_birthregion, (-1L):(-3L)), shift(is_birthregion, 1:3)), by = gid_cols]
comb[, c(paste0("pre", 1:3), paste0("post", 1:3)) := list(
  fifelse(is.na(f1), NA, f1 & (is.na(is_birthregion) | !is_birthregion)),
  fifelse(is.na(f2), NA, f2 & (is.na(is_birthregion) | !is_birthregion) & (is.na(f1) | !f1)),
  fifelse(is.na(f3), NA, f3 & (is.na(is_birthregion) | !is_birthregion) & (is.na(f1) | !f1) & (is.na(f2) | !f2)),
  fifelse(is.na(l1), NA, l1 & (is.na(is_birthregion) | !is_birthregion)),
  fifelse(is.na(l2), NA, l2 & (is.na(is_birthregion) | !is_birthregion) & (is.na(l1) | !l1)),
  fifelse(is.na(l3), NA, l3 & (is.na(is_birthregion) | !is_birthregion) & (is.na(l1) | !l1) & (is.na(l2) | !l2))
)]
comb[, c(paste0("f", 1:3), paste0("l", 1:3)) := NULL]

# Convert variables to logarithmic scale
comb[, c("lngdp", "lnpop") := list(log(wdi_gdpcappppcon2021), log(population / 1000L))]
comb[, c("wdi_gdpcappppcon2021", "population") := NULL]

# Reorder columns
setcolorder(comb, c(gid_cols, "year"))

# Rename columns
setnames(comb, gid_cols, tolower(gid_cols))

# Write to disk
fwrite(comb, "../data/analysis/adm_2.csv.gz")
  
rm(qog, frac, bl)
