"""Phase 1 profiling: GHGRP Capstone Dataset. Read-only, prints a data quality report."""
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

df = pd.read_csv("data/Capstone_Dataset.csv")
print("=" * 70)
print("SHAPE:", df.shape)
print("=" * 70)

print("\n--- COLUMNS ---")
for i, c in enumerate(df.columns):
    print(f"{i:2d} {c}")

print("\n--- MISSING VALUES (top) ---")
miss = df.isna().sum()
print(miss[miss > 0].sort_values(ascending=False).head(10))
print("total cells missing:", int(miss.sum()))

print("\n--- DUPLICATED ROWS:", int(df.duplicated().sum()))

# key columns by position (bilingual headers, match by keyword)
def col(kw):
    hits = [c for c in df.columns if kw.lower() in c.lower()]
    return hits[0] if hits else None

c_prov = col("province")
c_year = col("reference year")
c_fac = col("facility name")
c_naics = col("naics code /")
c_naics_desc = col("description du code scian en anglais") or col("english facility naics")
c_co2 = [c for c in df.columns if c.strip().lower() == "co2 (tonnes)"][0]
c_tot = col("total emissions")

print("\n--- YEAR RANGE ---")
print("min:", df[c_year].min(), "max:", df[c_year].max())
print(df[c_year].value_counts().sort_index().tail(5))

print("\n--- PROVINCE COUNTS ---")
print(df[c_prov].value_counts().head(8))

print("\n--- ZERO TOTAL-EMISSION ROWS:", int((df[c_tot] == 0).sum()))
print("--- NEGATIVE TOTAL-EMISSION ROWS:", int((df[c_tot] < 0).sum()))

print("\n--- ALBERTA SLICE ---")
ab = df[df[c_prov].str.contains("Alberta", na=False)].copy()
print("rows:", len(ab), "| facilities:", ab[c_fac].nunique())
# NOTE (instructor review Oct 3): facility NAME count (1503) overstates the
# facility population. 232 GHGRP IDs map to multiple names across years
# (renames/rebrands), so the lesson-correct grain is facility_id: 1199.
print("unique GHGRP ids:", ab["GHGRP ID No. / No d'identification du PDGES"].nunique(),
      "(lesson groups by id, not name)")
print("top NAICS sectors in Alberta:")
print(ab[c_naics_desc].value_counts().head(10))
print("\nAlberta total-emissions describe (tCO2e):")
print(ab[c_tot].describe()[["count", "mean", "std", "min", "50%", "max"]])
print("\nAlberta facilities above 100kt in latest year:")
latest = ab[c_year].max()
ab_latest = ab[ab[c_year] == latest]
n_big = int((ab_latest[c_tot] >= 100_000).sum())
print(f"year {latest}: {n_big} facility-rows >= 100,000 tCO2e")
print("\nCO2 vs total emissions (Alberta, latest year) sample:")
print(ab_latest[[c_fac, c_co2, c_tot]].sort_values(c_tot, ascending=False).head(5).to_string())
