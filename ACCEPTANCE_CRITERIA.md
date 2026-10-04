# Acceptance Criteria — carbon-capture screening model (v2)

Defined from the AMII capstone deployment lesson's grading bars. A model
version is accepted for production only if it clears all three on
5-fold stratified cross-validation.

| Criterion | Bar | v1 (2026-10-04) | v2 (2026-10-04) |
|---|---|---|---|
| Overall accuracy | >= 0.70 | 0.7467 | 0.7333 |
| CCS Candidate F1 | >= 0.80 | 0.8390 | 0.8291 |
| Potential CU Candidate recall | >= 0.50 | 0.65 | 0.65 |

**Verdict:** v1 and v2 both clear all three bars. v2 is accepted as the
staging candidate.

**Notes**
- v2 was retrained on recleaned data (course-order cleaning: missing
  values before type coercion, duplicate checks, trade-name forward
  fill). Zero prediction flips across all 150 facilities vs v1; the
  small CV metric deltas come from row-order-dependent fold assignment,
  not from changed model behavior.
- Re-verify these bars on every retrain before promoting staging to
  production (backlog item 6, validation gate, will automate this).
