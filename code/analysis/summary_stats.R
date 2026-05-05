# Generate summary statistics

# Clear environment
rm(list = ls())

# Load packages
require("data.table")
require("Rcpp")
require("xtable")

# Order in which to display lights variables
light_order <- c("OLS", "Nechaev", "Li", "Chen", "Chiovelli")

# Load data
d <- fread("../data/analysis/adm_2.csv.gz", select = c("gid_0", "gid_2", "year", paste0("light_mean_", tolower(light_order)), "is_birthregion", "lnpop",
  "al_ethnic2000", "p_polity2", "vdem_libdem", "schooling", "lngdp"), key = c("gid_2", "year"))

# Create log lights
light_cols <- names(d)[grepl("^light_mean_", names(d))]
d[, paste0("ln_", light_cols) := lapply(.SD, function(x) log(x + 0.01)), .SDcols = light_cols]
rm(light_cols)

# Set adm 2 variables
d2 <- d[, .SD, .SDcols = c("gid_2", "year", paste0("ln_light_mean_", tolower(light_order)), "is_birthregion", "lnpop")]
d2[, l_is_birthregion := shift(is_birthregion), by = "gid_2"]
d2[year < 1991L | year > 2022L, is_birthregion := NA]
d2 <- d2[is.na(l_is_birthregion), (paste0("ln_light_mean_", tolower(light_order))) := NA_real_]
d2[is.na(l_is_birthregion) | year < 1992L | is.infinite(lnpop), lnpop := NA_real_]
d2[, c("l_is_birthregion", "gid_2", "year") := NULL]

# Set adm 0 variables
d0 <- d[year %between% c(1991L, 2022L), .SD[1L], by = c("gid_0", "year"), .SDcols = c("al_ethnic2000", "p_polity2", "vdem_libdem", "schooling", "lngdp")]
d0[, c("gid_0", "year") := NULL]

# Create table 1
data.table(
  Variable = c(paste0("\\multicolumn{5}{l}{\\textbf{Panel A: Summary statistics for region-level variables}} \\\\\n$\\ln Light_{ict}(", light_order[1L],
    ")$"), paste0("$\\ln Light_{ict}(", light_order[2:length(light_order)], ")$"), "$Leader_{ict}$", "$\\ln Pop_{ict}$",
    "\\midrule\n\\multicolumn{5}{l}{\\textbf{Panel B: Summary statistics for country-level variables}} \\\\\n$Fractionalization_c$", "$Polity_{ct}$",
    "$LibDemocracy_{ct}$", "$Schooling_{ct}$", "$\\ln GDP_{ct}$"),
  Observations = lapply(list(d2, d0), function(dx) vapply(dx, function(x) (!is.na(x)) |> sum(), integer(1L), USE.NAMES = F)) |> do.call(c, args = _),
  Mean = lapply(list(d2, d0), function(dx) vapply(dx, mean, numeric(1L), na.rm = T, USE.NAMES = F)) |> do.call(c, args = _),
  S.D. = lapply(list(d2, d0), function(dx) vapply(dx, sd, numeric(1L), na.rm = T, USE.NAMES = F)) |> do.call(c, args = _),
  Min. = lapply(list(d2, d0), function(dx) vapply(dx, min, numeric(1L), na.rm = T, USE.NAMES = F)) |> do.call(c, args = _),
  Max. = lapply(list(d2, d0), function(dx) vapply(dx, max, numeric(1L), na.rm = T, USE.NAMES = F)) |> do.call(c, args = _)
) |> 
  xtable(caption = "Summary statistics", label = "tab:table1", align = "lXccccc", digits = 3L) |> 
  print(file = "../paper/results/tables/table_1.tex", table.placement = "!ht", caption.placement = "top", tabular.environment = "tabularx",
    include.rownames = F, booktabs = T, width = "\\textwidth", comment = F, format.args = list(big.mark = ","), sanitize.text.function = function(x) x,
    hline.after = c(-1L, 0L), add.to.row = list(pos = list(12L),
    command = "\\bottomrule\n\\multicolumn{5}{l}{\\footnotesize Notes: \\autoref{sec:data} describes all variables and provides their sources.}\n"))
rm(d, d2, d0, light_order)
