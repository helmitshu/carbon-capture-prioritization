"""Phase 1 cleaning: rename bilingual headers, Alberta filter, types, missing values.
Writes data/Capstone_Dataset_clean.csv. Raw file untouched."""
import pandas as pd

df = pd.read_csv("data/Capstone_Dataset.csv")

RENAME = {
    "GHGRP ID No. / No d'identification du PDGES": "facility_id",
    "Reference Year / Ann\u00e9e de r\u00e9f\u00e9rence": "year",
    "Facility Name / Nom de l'installation": "facility_name",
    "Facility City or District or Municipality / Ville ou District ou Municipalit\u00e9 de l'installation": "city",
    "Facility Province or Territory / Province ou territoire de l'installation": "province",
    "Facility NPRI ID / Num\u00e9ro d'identification de l'INRP": "npri_id",
    "Facility NAICS Code / Code SCIAN de l'installation": "naics_code",
    "English Facility NAICS Code Description / Description du code SCIAN de l'installation en anglais": "sector",
    "French Facility NAICS Code Description / Description du code SCIAN de l'installation en fran\u00e7ais": "sector_fr",
    "Reporting Company Legal Name / D\u00e9nomination sociale de la soci\u00e9t\u00e9 d\u00e9clarante": "company_legal",
    "Reporting Company Trade Name / Nom commercial de la soci\u00e9t\u00e9 d\u00e9clarante": "company_trade",
    "CO2 (tonnes)": "co2",
    "CH4 (tonnes)": "ch4_t",
    "CH4 (tonnes CO2e / tonnes \u00e9q. CO2)": "ch4_co2e",
    "N2O (tonnes)": "n2o_t",
    "N2O (tonnes CO2e / tonnes \u00e9q. CO2)": "n2o_co2e",
    "Total Emissions (tonnes CO2e) / \u00c9missions totales (tonnes \u00e9q. CO2)": "total_emissions",
}
df = df.rename(columns=RENAME)
assert list(df.columns) == list(RENAME.values()), "rename mismatch"

# Alberta only (drop the single row with missing province)
df = df[df["province"].str.contains("Alberta", na=False)].copy()
print("Alberta rows:", len(df))

# numeric types for emissions
num_cols = ["co2", "ch4_t", "ch4_co2e", "n2o_t", "n2o_co2e", "total_emissions", "year", "naics_code"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# missing values: gas tonnes -> 0 (not reported = none reported); city -> Unknown
for c in ["ch4_t", "ch4_co2e", "n2o_t", "n2o_co2e"]:
    n = int(df[c].isna().sum())
    df[c] = df[c].fillna(0.0)
    print(f"filled {c}: {n} -> 0")
df["city"] = df["city"].fillna("Unknown")
df["company_trade"] = df["company_trade"].fillna(df["company_legal"])

# dedupe (none found, but enforce) and drop zero-emission rows (none found)
before = len(df)
df = df.drop_duplicates()
df = df[df["total_emissions"] > 0]
print(f"dedupe+zero filter: {before} -> {len(df)}")

# sanity: no missing in model-critical columns
crit = ["facility_name", "year", "naics_code", "sector", "co2", "total_emissions"]
print("missing in critical cols:", int(df[crit].isna().sum().sum()))

df.to_csv("data/Capstone_Dataset_clean.csv", index=False)
print("wrote data/Capstone_Dataset_clean.csv:", df.shape)
print("facilities:", df["facility_name"].nunique(), "| years:", df["year"].min(), "-", df["year"].max())
