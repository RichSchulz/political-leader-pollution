// This code is called from barro_lee.R
// It interpolates years of schooling per country

// The code relies on the assumption that the panel is sorted by country and year

// [[Rcpp::plugins(cpp20)]]
#include <Rcpp.h>
#include <vector>
#include <string>

// [[Rcpp::export]]
Rcpp::DataFrame interpolate(Rcpp::CharacterVector& gid_0, Rcpp::IntegerVector& year_in, Rcpp::NumericVector& yr_sch, const int n_rows_out) {
  // Insert into stl vectors to keep size flexible
  struct {
    std::vector<std::string> country;
    std::vector<double> schooling;
    std::vector<int> year;
  } out;
  out.country.reserve(n_rows_out);
  out.schooling.reserve(n_rows_out);
  out.year.reserve(n_rows_out);
  
  const int n_rows_in = year_in.size();
  
  // Insert given years
  for(int i = 0; i != n_rows_in; ++i) {
    out.year.push_back(year_in[i]);
    out.country.push_back(Rcpp::as<std::string>(gid_0[i]));
    out.schooling.push_back(yr_sch[i]);
  }
  
  double step {};
  bool is_last = true;
  
  // Insert interpolated years
  for(int i = 0; i != n_rows_in; ++i) {
    const bool is_first = is_last;
    // Check if this is the last observation of the country
    is_last = i == n_rows_in - 1 || gid_0[i + 1] != gid_0[i];
    // Handle countries with only one value
    if(is_first && is_last) {
      continue;
    }
    // Update step if it is not the last observation
    if(!is_last) {
      step = (yr_sch[i + 1] - yr_sch[i]) / (year_in[i + 1] - year_in[i]);
    }
    
    // Add years between 1989 and the country's first observation
    if(is_first && year_in[i] > 1989) {
      for(int y = year_in[i] - 1; y != 1988; --y) {
        out.country.push_back(Rcpp::as<std::string>(gid_0[i]));
        out.schooling.push_back(yr_sch[i] - step * (year_in[i] - y));
        out.year.push_back(y);
      }
    }
    
    // Add other years
    const int upper_bound = is_last ? 2024 : year_in[i + 1];
    for(int y = year_in[i] + 1; y != upper_bound; ++y) {
      out.country.push_back(Rcpp::as<std::string>(gid_0[i]));
      out.schooling.push_back(yr_sch[i] + step * (y - year_in[i]));
      out.year.push_back(y);
    }
  }
  
  // Create data frame
  Rcpp::CharacterVector r_country = Rcpp::wrap(out.country);
  Rcpp::NumericVector r_schooling = Rcpp::wrap(out.schooling);
  Rcpp::IntegerVector r_year = Rcpp::wrap(out.year);
  Rcpp::DataFrame all_years = Rcpp::DataFrame::create(Rcpp::Named("ccodealp") = r_country, Rcpp::Named("year") = r_year,
    Rcpp::Named("schooling") = r_schooling);
  return all_years;
}
