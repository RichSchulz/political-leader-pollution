# Run baseline regressions

# Clear environment
rm(list = ls())

# Load packages
require("fixest")
require("data.table")

# Load data
d <- fread("../data/analysis/adm_2.csv.gz")

# Order in which to display lights regressions
light_order <- c("OLS", "Nechaev", "Li", "Chen", "Chiovelli")

# Create log lights
light_cols <- names(d)[grepl("^light_mean_", names(d))]
d[, paste0("ln_01_", light_cols) := lapply(.SD, function(x) log(x + 0.01)), .SDcols = light_cols]
d[, paste0("ln_00_", light_cols) := lapply(.SD, log), .SDcols = light_cols]
d[, (light_cols) := NULL]
rm(light_cols)

# Standardize variables
std_vars <- c("al_ethnic2000", "p_polity2", "vdem_libdem", "schooling", "lngdp")
z_vars <- paste0("z_", std_vars)
z <- d[year %between% c(1991L, 2022L), .SD[1L], by = c("gid_0", "year"), .SDcols = c(std_vars)][,
  (z_vars) := lapply(.SD, scale), .SDcols = std_vars]
d[, (std_vars) := NULL]
z[, (std_vars) := NULL]
d <- z[d, on = c("gid_0", "year")]
rm(std_vars, z_vars, z)

# Define panel structure
d <- panel(d, ~gid_2 + year)

# Set dictionary
setFixest_dict(paste0("l(is_birthregion)TRUE: $Leader_{ict-1}$\n ",
  "lnpop: $\\ln Pop_{ict}$\n ",
  paste0("l(pre", 1:3, ")TRUE: $FutureLeader", 1:3, "_{ict-1}$", collapse = "\n "), "\n ",
  paste0("l(post", 1:3, ")TRUE: $PastLeader", 1:3, "_{ict-1}$", collapse = "\n "), "\n ",
  "l(is_birthregion*z_al_ethnic2000): $Leader_{ict-1} \\times Fractionalization_{c}$\n ",
  "l(is_birthregion*z_p_polity2): $Leader_{ict-1} \\times Polity_{ct-1}$\n ",
  "l(is_birthregion*z_vdem_libdem): $Leader_{ict-1} \\times LibDemocracy_{ct-1}$\n ",
  "l(is_birthregion*z_schooling): $Leader_{ict-1} \\times Schooling_{ct-1}$\n ",
  "l(is_birthregion*z_lngdp): $Leader_{ict-1} \\times \\ln GDP_{ct-1}$\n ",
  "gid_2: Region-fixed effects\n ",
  "gid_0^year: Country-year-fixed effects\n"
))

# Define table headers and table "footers"
table_header <- function(caption, label) {
  return(paste0(
    "\\begin{table}[!htbp]\n",
    "\\centering\n",
    "\\caption{", caption, "}\n",
    "\\label{tab:", label, "}\n",
    "\\small\n",
    "\\begin{tabularx}{\\textwidth}{Xccccc}\n",
    "\\toprule\n",
    "& (1) & (2) & (3) & (4) & (5) \\\\\n",
    "\\midrule\n",
    "Dependent variable: & \\multicolumn{5}{c}{$\\ln Light_{ict}(\\psi)$} \\\\\n",
    "Data source $\\psi$: & ", paste0(light_order, collapse = " & "), " \\\\"
  ))
}
table_footer <- function() {
  return(paste0(
    "\\end{tabularx}\n",
    "\\end{table}\n"
  ))
}

# Estimate table 2
table_2_reg <- list(
  feols(ln_01_light_mean_.[tolower(light_order)] ~ l(is_birthregion) | gid_2 + gid_0^year, d, vcov = ~gid_0),
  feols(ln_00_light_mean_.[tolower(light_order)] ~ l(is_birthregion) | gid_2 + gid_0^year, d, vcov = ~gid_0),
  feols(ln_01_light_mean_.[tolower(light_order)] ~ l(is_birthregion) + lnpop | gid_2 + gid_0^year, d, vcov = ~gid_0),
  feols(ln_01_light_mean_.[tolower(light_order)] ~ l(post1) + l(post2) + l(post3) + l(is_birthregion) + l(pre1) + l(pre2) + l(pre3) | gid_2 + gid_0^year, d,
    vcov = ~gid_0)
)
# Write results to disk
table_2 <- lapply(1:4, function(pos) {
  panel_title <- c(
    "A: Main results",
    "B: Extensive margin",
    "C: Controlling for population",
    "D: Dynamics"
  )[pos]
  esttex(table_2_reg[[pos]],
    digits = 3L,
    fitstat = "n",
    fontsize = "small",
    postprocess.tex = function(tb, pos) {
      if(pos == 4L) {
        return(tb[min(which(grepl("\\midrule", tb, fixed = T))):which(grepl("Notes", tb))])
      }
      return(tb[min(which(grepl("\\midrule", tb, fixed = T))):which(grepl("^Observations", tb))])
    },
    pos = pos,
    style.tex = style.tex(
      line.top = "\\toprule",
      model.title = "",
      tablefoot = F,
      line.bottom = paste0("\\bottomrule\n",
      "\\multicolumn{6}{p{0.98\\textwidth}}{\\footnotesize ",
      "Notes: The dependent variables, $\\ln Lights_{ict}(\\psi)$, are the log of average nighttime light intensity, computed based on the different ",
      "nighttime lights data sources $\\psi$ indicated at the top of each column and discussed in Section 2.1. We add a small constant (0.01) before taking ",
      "logs in panels A, C, and D, but not in panel B. The main explanatory variable, $Leader_{ict-1}$, indicates birth regions of the country's current ",
      "political leader. All specifications include region- and country-year-fixed effects, and those in panel C also log population ($\\ln Pop_{ict}$). ",
      "Panel D includes dummy variables $FutureLeaderX_{ict-1}$, indicating regions that are not currently a leader birth region, but will become one in ",
      "exactly $X$ years; and dummy variables  $PastLeaderX_{ict-1}$ indicating regions that are not currently a leader birth region, but were one until ",
      "exactly $X$ years ago. Standard errors are clustered at the country level and reported in parentheses. ***/**/* indicates statistical significance ",
      "at the 1/5/10 percent level.}"),
      stats.title = "",
      var.title = paste0("\\midrule\n\\multicolumn{6}{l}{\\textbf{Panel ", panel_title, "}}"),
      fixef.title = "\\midrule",
      fixef.where = "stats"
    )
  )
}) |> 
  do.call(c, args = _)
c(table_header("Evidence for regional favoritism", "table2"), table_2, table_footer()) |> 
  writeLines("../paper/results/tables/table_2.tex")
rm(table_2_reg, table_2)

# Estimate table 3
table_3_reg <- list(
  feols(ln_01_light_mean_.[tolower(light_order)] ~ l(is_birthregion) + l(is_birthregion*z_al_ethnic2000) | gid_2 + gid_0^year, d, vcov = ~gid_0),
  feols(ln_01_light_mean_.[tolower(light_order)] ~ l(is_birthregion) + l(is_birthregion*z_p_polity2) | gid_2 + gid_0^year, d, vcov = ~gid_0),
  feols(ln_01_light_mean_.[tolower(light_order)] ~ l(is_birthregion) + l(is_birthregion*z_vdem_libdem) | gid_2 + gid_0^year, d, vcov = ~gid_0),
  feols(ln_01_light_mean_.[tolower(light_order)] ~ l(is_birthregion) + l(is_birthregion*z_schooling) | gid_2 + gid_0^year, d, vcov = ~gid_0),
  feols(ln_01_light_mean_.[tolower(light_order)] ~ l(is_birthregion) + l(is_birthregion*z_lngdp) | gid_2 + gid_0^year, d, vcov = ~gid_0)
)
# Write results to disk
table_3 <- lapply(1:5, function(pos) {
  panel_title <- c(
    "A: Ethnic fractionalization",
    "B: Polity score",
    "C: Liberal democracy index",
    "D: Years of schooling",
    "E: GDP per capita"
  )[pos]
  esttex(table_3_reg[[pos]],
    digits = 3L,
    fitstat = "n",
    fontsize = "small",
    postprocess.tex = function(tb, pos) {
      if(pos == 5L) {
        return(tb[min(which(grepl("\\midrule", tb, fixed = T))):which(grepl("Notes", tb))])
      }
      return(tb[min(which(grepl("\\midrule", tb, fixed = T))):which(grepl("^Observations", tb))])
    },
    pos = pos,
    style.tex = style.tex(
      line.top = "\\toprule",
      model.title = "",
      tablefoot = F,
      line.bottom = paste0("\\bottomrule\n",
      "\\multicolumn{6}{p{0.98\\textwidth}}{\\footnotesize ",
      "Notes: The dependent variables, $\\ln Lights_{ict}(\\psi)$, are the log of average nighttime light intensity (plus 0.01), computed based on the ",
      "different nighttime lights data sources $\\psi$ indicated at the top of each column and discussed in Section 2.1. The main explanatory variable, ",
      "$Leader_{ict-1}$, indicates birth regions of the country's current political leader. Each panel includes interaction terms between ",
      "$Leader_{ict-1}$ and one of the following z-standarized country-level variables: Ethnic fractionalization by \\cite{alesina2003fractionalization} ",
      "in panel A, the Polity2 score from the Polity V Project in panel B, the Liberal Democracy index from the V-Dem Project in panel C, the years of ",
      "schooling by \\cite{barro2013new} in panel D, and log GDP per capita (PPP) by the World Development Indicators in panel E. All specifications ",
      "include region- and country-year-fixed effects. Standard errors are clustered at the country level and reported in parentheses. ***/**/* indicates ",
      "statistical significance at the 1/5/10 percent level.}"),
      stats.title = "",
      var.title = paste0("\\midrule\n\\multicolumn{6}{l}{\\textbf{Panel ", panel_title, "}}"),
      fixef.title = "\\midrule",
      fixef.where = "stats"
    )
  )
}) |> 
  do.call(c, args = _)
c(table_header("Heterogeneity in regional favoritism", "table3"), table_3, table_footer()) |> 
  writeLines("../paper/results/tables/table_3.tex")
rm(table_3_reg, table_3)
