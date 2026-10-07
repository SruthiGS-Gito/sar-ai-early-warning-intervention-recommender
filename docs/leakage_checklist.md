# Leakage checklist (ticket 110)

A feature is safe only if every item below is true.

- It uses records dated on or before the cutoff day.
- It does not use the final result, withdrawal date or any field set after the cutoff.
- Aggregates are computed per student without looking at the validation set.
- Encoders and scalers are fitted on training data only.
- `tests/test_leakage.py` passes for the feature.
