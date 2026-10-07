# Review 1 scope (Sprint 1, 5 to 9 Oct 2026)

Goal: a clean, leakage-safe data foundation and a first baseline model.

## Mon 5 Oct
- 113 Project kickoff and brief analysis (Ancy)

## Tue 6 Oct
- 114 Download OULAD and first inspection (Abhijith)

## Wed 7 Oct
- 102 Development environment and GitHub repository (Sruthi)
- 115 Backlog and sprint planning in Jira (Sruthi)

## Thu 8 Oct
- 103 Loader for the seven OULAD tables (Abhijith), `src/sar/data/loader.py`
- 104 Clean student data and check join keys (Abhijith), `src/sar/data/clean.py`
- 107 Engagement metrics up to Day 30 (Gouri), `src/sar/features/engagement.py`
- 109 Confirm Day 30 cutoff and validation split (Sruthi), `src/sar/data/split.py`
- 110 Leakage checklist and automated cutoff test (Sruthi), `tests/test_leakage.py`
- 111 Target definition and Day 30 population (Midhula), `src/sar/data/target.py`

## Fri 9 Oct
- 105 Dataset choice and proxy decisions (Abhijith), `docs/dataset_decisions.md`
- 202 Processed tables for dashboard and funnel (Abhijith), `data/processed/`
- 106 Grade and outcome distribution charts (Ancy), `notebooks/`
- 205 Attendance and assignment patterns across departments (Ancy), `notebooks/`
- 108 Consecutive inactive days (Gouri), `src/sar/features/engagement.py`
- 116 Engagement-by-outcome charts (Gouri), `notebooks/`
- 204 Withdrawal timing analysis (Gouri), `notebooks/`
- 213 Engagement and registration features (Sruthi), `src/sar/features/registration.py`
- 211 Assessment features (Midhula), `src/sar/features/assessment.py`
- 112 Baseline Logistic Regression model (Midhula), `src/sar/models/baseline.py`
- 101 Sprint ceremonies and Friday review (Sruthi)

## Order of dependency
- Everyone waits for the loader and cleaned tables (103, 104).
- 108 and 116 wait for 107.
- 112 waits for 109 and 110.
- 106 and 205 wait for the cleaned tables.
