# Retrain policy (backlog item 5)

What triggers a retrain of the capstone model, how each trigger is
detected, and what happens next. This is a policy document, not
automation: no retraining happens on its own, and no new version reaches
production without the validation gate (06) and the promotion script
(07). Every retrain produces a new registry version that starts in
staging.

## Triggers

### 1. New GHGRP data release (primary trigger)
The Greenhouse Gas Reporting Program publishes a new annual dataset,
usually in spring for the prior reporting year. A new year of facility
data is the strongest reason to retrain: more training rows, and the
model sees the latest emissions reality.

- Detection: check the open.canada.ca GHGRP dataset page each spring, or
  subscribe to its update notifications.
- Action: download the new release, run the pipeline in order
  (01 fetch, 02 clean, 03 engineer, 04 train), register the result as a
  new version in staging, run the validation gate, exercise it on the
  staging API, then promote only if everything passes.

### 2. Prediction drift
If the mix of predictions coming out of the deployed API drifts away
from the baseline, the world the model sees has changed and the model
may be going stale.

- Baseline (model v2, Oct 4 2026): CCS Candidate 70.7%, Potential CU
  Candidate 29.3% across the 150 Alberta facilities.
- Detection: read the prediction log (logs/predictions.jsonl, backlog
  item 4). Over any rolling 30-day window, if the CCS share moves more
  than 10 percentage points from baseline, or median log_emissions of
  incoming requests shifts more than 15% from the training median, flag
  it for review.
- Action: review first, retrain second. Drift can mean the data changed
  (retrain) or the incoming requests changed (a client sending different
  facilities, not a model problem). Do not retrain on drift alone
  without understanding which it is.

### 3. Scheduled cadence
Once a year, every spring after the GHGRP release check, review whether
a retrain is warranted even if no drift was flagged. This catches slow
decay that never trips a threshold.

### 4. Label-rule change
The labels are rule-derived (85% CO2 share cutoff). If the cutoff or the
labeling logic ever changes, for example a policy update redefines what
counts as CCS-suitable, all labels must be re-derived and the model
retrained and revalidated from scratch. Treat the old versions as
archived history, not as fallbacks.

## What a retrain always includes

1. New registry version, stage = staging. Never overwrite an existing
   version's artifact or card.
2. Validation gate (06) must PASS before the version is exercised.
3. Exercise on the staging API, including a prediction-parity check
   against the current production version: large unexplained flips get
   investigated, not waved through.
4. Promotion (07) moves the production pointer. The deploy step stays
   manual: set MODEL_VERSION on the production API service and redeploy.

## Explicit non-goals

- No automatic retraining on any trigger. A human reviews the trigger
  and decides.
- No automatic promotion. The gate passing is necessary but not
  sufficient; the promotion script is run by hand.
- No silent label changes. A label-rule change is a new modeling
  generation, documented as such.
