# Leakage checklist (SCRUM-171)

A feature is safe only if every item below is true.

- It uses records dated on or before the cutoff day.
- It does not use final_result, date_unregistration or any field set after the cutoff.
- Aggregates are computed per student without looking at the validation set.
- Encoders and scalers are fitted on training data only.
- `tests/test_leakage.py` passes for the feature.

## How the test works
Each feature builder runs twice: on the data, and on the same data with every post-cutoff row changed. The two outputs must be identical. The output must also contain no final_result, date_unregistration or at_risk column.

## Builders covered
- engagement_metrics and consecutive_inactive_days (SCRUM-166, 167)
- assessment_features (SCRUM-173)
- registration_features is covered in `tests/test_registration.py`

## Whole-course fields that must not be used
- total clicks over the whole course
- average score over the whole course
- submission counts over the whole course
