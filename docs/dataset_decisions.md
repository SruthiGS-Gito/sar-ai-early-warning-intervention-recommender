# Dataset decisions (SCRUM-162)

## Dataset
Open University Learning Analytics Dataset (OULAD), 7 CSV tables, 7 modules (AAA to GGG), 4 presentations (2013B, 2013J, 2014B, 2014J).

## Why this dataset
- It records both engagement (VLE clicks) and outcomes (final result) for the same students.
- It has dated records, so a prediction day can be fixed and checked.
- It is public and anonymised.

## Verified facts (data-understanding notebook, Part 1)
- 32,593 registrations. One row per (code_module, code_presentation, id_student) in studentInfo and studentRegistration.
- 28,785 unique students, so some students register more than once.
- studentVle has 10,655,280 rows. All keys link to studentInfo and vle.
- 18 assessments have no student submission.
- Final result: Pass 12,361, Withdrawn 10,156, Fail 7,052, Distinction 3,024.
- At Risk (Fail or Withdrawn) is 52.8% of all registrations.

## Proxy decisions
- Department: code_module.
- Attendance and absence: days with at least one VLE click, and days without one.
- Engagement: VLE clicks. Only days on or before the cutoff are used.

## Cleaning rules (src/sar/data/clean.py)
- imd_band label "10-20" is written "10-20%". Missing imd_band (1,111 rows, 3.41%) becomes "Unknown".
- studentVle exact duplicate rows (787,170 rows, 7.39%) are kept. After removing them, 2,587,864 rows still share the same student, site and day with different click counts, so several records per day exist and exact repeats are expected. Dropping them would remove 1,262,036 clicks. `drop_duplicates=True` switches the rule on.
- Missing values are not filled elsewhere. Imputation happens inside the model pipeline.

## Missing values
- assessments.date: 11 rows (5.34%).
- vle.week_from and week_to: 5,243 rows (82.39%). These columns are not used.
- studentRegistration.date_registration: 45 rows (0.14%). date_unregistration: 22,521 rows (69.10%), where an empty value means the student did not unregister.
- studentAssessment.score: 173 rows (0.10%).

## Data quality decisions (closed)
These items were open at Review 1. Each is closed with the decision below; the records are kept as published.
- 93 Withdrawn students without an unregistration date and 9 Fail students with one (102 records, all At Risk). Kept. The target uses `final_result` only, and the Day 30 population excludes a student only when an unregistration date on or before Day 30 is recorded, so a missing date keeps the student in the population. No last-activity correction is applied.
- Dates beyond course length: 1 unregistration (FFF 2013J, day 444) and 85 submissions (CMA 70, Exam 12, TMA 3), mostly FFF. All fall after Day 30, so the Day 30 filter removes them before any feature is built.
- 2,057 submissions before Day 0. Kept: they are known by Day 30. Banked results (`is_banked` = 1) are excluded from the submission delay features (`src/sar/features/assessment.py`).
- studied_credits up to 655 and 23 click rows of 1,000 or more. Kept as published, with no capping.

## Protected characteristics
The six demographic columns (gender, region, highest_education, imd_band, age_band, disability) are excluded from the final model's inputs, as recorded in `excluded_protected_features` in `models/final/config.json`. They are read only for the fairness audit. The final model has 24 inputs; see [final_model.md](final_model.md) for the configuration and the fairness results.

## Known limitations
- Single institution, distance learning. Results may not transfer to a classroom setting.
