import wbgapi as wb
import xlsxwriter
import pandas as pd

# Function to mine data from the API


def get_values_1970_2019(indicator, country="KEN"):
    data = wb.data.DataFrame(indicator, country)
    data_T = data.T
    clean_data = data_T.dropna()
    data_1980_2019 = clean_data.loc["YR1970":"YR2019"]
    return data_1980_2019


# Indicator variables
inflation_rate_indicator = ["FP.CPI.TOTL.ZG"]
real_interest_indicator = ["FR.INR.RINR"]
official_exchange_rate_indicator = ["PA.NUS.FCRF"]
pop_growth_rate_indicator = ["SP.POP.GROW"]
real_gdp_indicator = ["NY.GDP.MKTP.CD"]
broad_money_pc_gdp_indicator = ["FM.LBL.BMNY.GD.ZS"]
population_indicator = ["SP.POP.TOTL"]
per_capita_USD_indicator = ["NY.GDP.PCAP.CD"]
gdp_growth_indicator = ["NY.GDP.MKTP.KD.ZG"]
lending_interest_indicator = ["FR.INR.LEND"]
deposit_interest_rate_indicator = ["FR.INR.DPST"]
current_exports_indicator = ["NE.EXP.GNFS.CD"]
unemp_modeled_indicator = ["SL.UEM.TOTL.ZS"]
imports_USD_indicator = ["NE.IMP.GNFS.CD"]
cpi_indicator = ["FP.CPI.TOTL"]
millitary_expenditure_indicator = ["MS.MIL.XPND.CD"]
gvt_exp_on_education_indicator = ["SE.XPD.TOTL.GD.ZS"]
life_expc_years_indicator = ["SP.DYN.LE00.IN"]
co2_emissions_per_capita_indicator = ["EN.ATM.CO2E.PC"]
health_exp_per_capita_indicator = ["SH.XPD.CHEX.PC.CD"]
health_expe_pc_GDP_indicator = ["SH.XPD.CHEX.GD.ZS"]
risk_premium_indicator = ["FR.INR.RISK"]

# Output from the api
real_interest = get_values_1970_2019(real_interest_indicator)
inflation = get_values_1970_2019(inflation_rate_indicator)
ex_rate = get_values_1970_2019(official_exchange_rate_indicator)
pop_growth_rate = get_values_1970_2019(pop_growth_rate_indicator)
real_gdp = get_values_1970_2019(real_gdp_indicator)
broad_money = get_values_1970_2019(broad_money_pc_gdp_indicator)
pop = get_values_1970_2019(population_indicator)
per_capita = get_values_1970_2019(per_capita_USD_indicator)
gdp_growth = get_values_1970_2019(gdp_growth_indicator)
lending_interest = get_values_1970_2019(lending_interest_indicator)
deposit_rate = get_values_1970_2019(deposit_interest_rate_indicator)
current_exports = get_values_1970_2019(current_exports_indicator)
modeled_unemp = get_values_1970_2019(unemp_modeled_indicator)
imports = get_values_1970_2019(imports_USD_indicator)
consumer_price_index = get_values_1970_2019(cpi_indicator)
millitary_expenditure = get_values_1970_2019(millitary_expenditure_indicator)
education_exp_pec_of_gdp = get_values_1970_2019(gvt_exp_on_education_indicator)
life_expc_years = get_values_1970_2019(life_expc_years_indicator)
co2_emmisions = get_values_1970_2019(co2_emissions_per_capita_indicator)
health_exp_per_capita_USD = get_values_1970_2019(health_exp_per_capita_indicator)
health_expe_pc_GDP = get_values_1970_2019(health_expe_pc_GDP_indicator)
risk_premium = get_values_1970_2019(risk_premium_indicator)

# Create a dataframe

df = pd.DataFrame(pop)
df = df.rename(columns={"KEN": "population"})
df["broad_money"] = broad_money
df["real_gdp"] = real_gdp
df["pop_growth"] = pop_growth_rate
df["exc_rate"] = ex_rate
df["inflation"] = inflation
df["real_interest"] = real_interest
df["per_capita_USD"] = per_capita
df["gdp_growth_rate"] = gdp_growth
df["lending_interest_rate"] = lending_interest
df["deposit_interest_rate"] = deposit_rate
df["current_exports"] = current_exports
df["modeled_unemp"] = modeled_unemp
df["imports_in_USD"] = imports
df["cpi"] = consumer_price_index
df["millitary_expenditure_USD"] = millitary_expenditure
df["edu_exp_pc_of_GDP"] = education_exp_pec_of_gdp
df["life_exp_years"] = life_expc_years
df["co2_emmisions_per_capita"] = co2_emmisions
df["health_exp_percapita_USD"] = health_exp_per_capita_USD
df["health_exp_pc_of_GDP"] = health_expe_pc_GDP
df["risk_premium"] = risk_premium

df["treasury_rate"] = df["lending_interest_rate"] - df["risk_premium"]

print(df)

# Create a pandas excel writer

writer = pd.ExcelWriter("project_data.xlsx", engine="xlsxwriter")

# Convert df to xlsxwriter Excel object

df.to_excel(writer, sheet_name="first")

# Close Pandas Excel Write and output the Excel file

writer.save()
