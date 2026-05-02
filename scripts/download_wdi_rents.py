import pandas as pd
import requests

indicators = {
    'NY.GDP.TOTL.RT.ZS': 'total_resource_rents_gdp',
    'NY.GDP.PETR.RT.ZS': 'oil_rents_gdp',
    'NY.GDP.MINR.RT.ZS': 'mineral_rents_gdp'
}

def fetch_wdi_data(indicator):
    url = f"http://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&per_page=20000"
    response = requests.get(url)
    data = response.json()
    if len(data) > 1:
        records = data[1]
        # Some records might not have 'countryiso3code' if it's an old API response format,
        # but modern WDI JSON does.
        processed = []
        for r in records:
            iso3 = r.get('countryiso3code')
            if not iso3:
                iso3 = r.get('country', {}).get('id') # Fallback
            processed.append({
                'country_code': iso3,
                'year': int(r['date']),
                indicators[indicator]: r['value']
            })
        return pd.DataFrame(processed)
    return pd.DataFrame()

dfs = []
for ind in indicators:
    print(f"Fetching {ind}...")
    df = fetch_wdi_data(ind)
    dfs.append(df)

final_df = dfs[0]
for df in dfs[1:]:
    final_df = final_df.merge(df, on=['country_code', 'year'], how='outer')

# Drop rows without a valid 3-letter country code (e.g. aggregates)
final_df = final_df[final_df['country_code'].str.len() == 3]
final_df = final_df.dropna(subset=['country_code'])

# Sort and save
final_df = final_df.sort_values(['country_code', 'year'])
final_df.to_csv('data/wdi_resource_rents.csv', index=False)
print("Saved to data/wdi_resource_rents.csv")
