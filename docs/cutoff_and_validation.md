# Cutoff and validation (SCRUM-170)

## Cutoff day
Day 30 of the module (`sar.config.CUTOFF_DAY`). A prediction is made using records dated on or before Day 30.

## Target
At Risk = Fail or Withdrawn. On Track = Pass or Distinction.

## Students who leave before the cutoff
Students with date_unregistration on or before Day 30 are excluded from modelling and kept for the retention funnel.

Verified counts:
- all registrations: 32,593
- unregistered on or before Day 30: 5,127 (every one is At Risk)
- model population: 27,466
- At Risk share in the model population: 44.0% (52.8% in all registrations)
- unregistered after Day 30 and still in the population: 4,945
- registered after Day 30: 16
- Withdrawn students who leave after Day 30: 4,944 (49.1% of Withdrawn with a date)
- no VLE activity by Day 30: 856 students (82.2% At Risk); with activity: 26,610 (42.8% At Risk)

## Validation split (src/sar/data/split.py)
- Train on earlier presentations (2013B, 2013J, 2014B). Validate on 2014J.
- Reason: a new presentation is scored with a model trained on past ones, so validation follows the same direction in time.
- 1,605 of 10,670 students in 2014J (15.0%) also registered in an earlier presentation.
- Students who appear in both sets are removed from the training set. `overlap_report()` gives the counts.
- In the Day 30 population, 1,258 students are in both sets. Removing them takes 1,348 rows out of training (18,263 to 16,915). Validation has 9,203 rows.
- Encoders and scalers are fitted on the training set only.

## Metrics
Recall and PR-AUC lead. The At Risk share of the validation set is the PR-AUC reference level.
