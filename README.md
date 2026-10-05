# Carbon Capture Prioritization

Machine learning screening of Alberta industrial facilities for carbon capture
and storage (CCS) or carbon utilization (CU) deployment, built on public
emissions data.

## Live deployment (v2, current)

This branch is the official production line. The original v1 app is retired.

- Screening app: https://carbon-capture-app-v2-production.up.railway.app
- Prediction API: https://carbon-capture-api-production.up.railway.app
  (`/health`, `/predict`, `/model/structure`)

The app talks to the API over JSON and falls back to the bundled model
file if the API is unreachable, so it never goes down because the API does.
Model versions are tracked in `model/registry/`; only versions that pass
the validation gate (`scripts/06_validate_model.py`) may be promoted to
production (`scripts/07_promote_model.py`).

## The problem

Alberta emits roughly 270 megatonnes of CO2e per year. Carbon capture is
expensive, so it cannot go everywhere. The question is where each dollar
removes the most CO2. Case-by-case engineering reviews are thorough but
resource-intensive, so attention naturally concentrates on the largest
emitters. Public emissions data can do the first screening pass for every
reporting facility at once.

## Data

Environment and Climate Change Canada, Greenhouse Gas Reporting Program
(GHGRP), public dataset 2004 to 2023. 18,771 facility-year records;
6,999 rows across 1,199 Alberta facilities after cleaning.

## Method

- Aggregate to one row per facility (mean emissions across reported years).
- Keep facilities averaging at least 100,000 tCO2e per year, the population
  covered by Alberta's TIER regulation.
- Label by CO2 share of total emissions: at or above 85% screens as a
  CCS Candidate (concentrated stream), below as a Potential CU Candidate
  (mixed stream, better for utilization). The 85% line is a stipulated
  threshold, not an industry standard.
- Decision Tree classifier (max depth 3, balanced class weights) on three
  features: log-scaled average emissions, encoded industry sector, years
  reported. Gas shares are excluded from features to avoid label leakage.
- Evaluated with 5-fold stratified cross validation.

## Results

150 above-threshold facilities: 130 CCS candidates, 20 CU candidates.
Accuracy 0.75, macro F1 0.62. CCS recall 0.76, CU recall 0.65. CU precision
is only 0.30, so most CU predictions need human review. The tree
splits almost entirely on emissions scale; sector refines the boundary.
Oil sands, oil and gas extraction, and fossil-fuel power generation lead
the priority list.

## Run it

```bash
pip install -r requirements.txt
python scripts/02_clean.py          # clean the GHGRP extract
python scripts/03_features_model.py # features, labels, cross validation
python scripts/04_save_model.py     # save deployment artifacts
streamlit run app.py                # screening app
```

## Structure

- `app.py`: screening app (pick a facility, see its priority and the reasons)
- `features.py`: shared feature engineering, constants, and model config. Single
  source of truth imported by `scripts/03`, `scripts/04`, and `app.py`
- `model/`: trained tree, sector encoder, facility lookup table
- `scripts/`: profiling, cleaning, feature engineering, modeling
- `data/`: raw and cleaned GHGRP extracts
