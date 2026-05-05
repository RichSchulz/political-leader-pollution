# QoG pre-processing

# Clear environment
rm(list = ls())

# Load packages
require("data.table")
require("Rcpp")

# Load data
qog <- fread("../data/raw/qog/qog_std_ts_jan26.csv", select = c("year", "ccodealp", "p_polity2", "vdem_libdem", "wdi_gdpcappppcon2021", "bl_asymf",
  "al_ethnic2000"), key = c("ccodealp", "year"))[
  year %between% c(1989L, 2023L),]

# Interpolate schooling
sourceCpp("./dataprep/interpolate.cpp")
school <- qog[!is.na(bl_asymf), c("ccodealp", "year", "bl_asymf")]
school <- interpolate(school[["ccodealp"]], school[["year"]], school[["bl_asymf"]], uniqueN(school, by = "ccodealp") * (2024L - 1989L))
setDT(school)
qog[, bl_asymf := NULL]
qog <- school[qog, on = c("ccodealp", "year")]
rm(school)

# Extend static fractionalization variable to all years
frac <- qog[!is.na(al_ethnic2000), .SD[1L], by = "ccodealp", .SDcols = "al_ethnic2000"]
qog[, al_ethnic2000 := NULL]
qog <- frac[qog, on = "ccodealp"]
rm(frac)

# Change country column name
setnames(qog, "ccodealp", "GID_0")

# Remove duplicated rows
qog <- unique(qog, by = c("GID_0", "year"))

# Write to disk
fwrite(qog, "../data/interim/qog.csv.gz")
