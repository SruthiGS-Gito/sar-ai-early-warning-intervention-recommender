# Student Academic Risk Early Warning and Intervention Recommender

Flags students at risk of failing or withdrawing from their first 30 days of activity, explains each flag, and suggests support options.

## Overview

**Problem.** Of the 32,593 registrations in the dataset, 52.8% end in Fail or Withdrawn. Early activity already separates the groups: students with no activity by Day 30 end At Risk 82% of the time, against 28% for those active on 21 or more days. And 49% of withdrawals happen after Day 30, when contact can still be made (`notebooks/164_SGS_day30_insights.ipynb`).

**Who it helps.** Student support staff and module teams who need a prioritised list of students to contact in the first weeks of a module.

**What the system does.**
1. Scores every student on Day 30 of a module, using only data recorded up to that day.
2. Explains each score with SHAP: which activity, assessment and study-load inputs raise or lower it, in a chart and a plain-language sentence.
3. Suggests up to three support types (tutoring, peer tutoring, mentoring, weekly check-ins, financial aid referral), with context from similar students in earlier presentations.
4. Presents all of this in a local Streamlit app.

TrackGenesis internship project 8. Current state: Sprint 2 (Review 2).

## Dataset

[Open University Learning Analytics Dataset (OULAD)](https://research.stem.open.ac.uk/ouanalyse/dataset/): 7 tables, 7 modules (AAA to GGG), 4 presentations (2013B, 2013J, 2014B, 2014J), 32,593 registrations.

- **Target.** At Risk means a final result of Fail or Withdrawn; On Track means Pass or Distinction.
- **Cutoff.** Day 30 of the module (`sar.config.CUTOFF_DAY`). Students who unregistered on or before Day 30 are left out, leaving a model population of 27,466 registrations, of which 44.0% are At Risk.
- **Split.** By time. Train on 2013B, 2013J and 2014B; test on 2014J. The 1,258 students who appear in both are removed from training (1,348 rows), leaving 16,915 training rows and 9,203 test rows.
- **Data is not included in this repository.** Download the dataset and place the seven CSV files in `data/raw/`:
  `assessments.csv`, `courses.csv`, `studentAssessment.csv`, `studentInfo.csv`, `studentRegistration.csv`, `studentVle.csv`, `vle.csv`.

Citation: Kuzilek, J., Hlosta, M. and Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data* 4, 170171. https://doi.org/10.1038/sdata.2017.171. The dataset is released under a CC BY 4.0 licence.

Details: [docs/dataset_decisions.md](docs/dataset_decisions.md), [docs/target_spec.md](docs/target_spec.md), [docs/cutoff_and_validation.md](docs/cutoff_and_validation.md).

## Approach

| Step | What it does | Where |
|---|---|---|
| Cutoff and target | Day 30 population and the Fail or Withdrawn target | `src/sar/data/target.py` |
| Features | Activity, inactivity streaks, assessments and registration timing, all up to Day 30; enrolment details | `src/sar/features/`, `scripts/run_baseline.py` |
| Leakage controls | Post-cutoff rows are dropped before any feature is built. A test changes every post-cutoff row and checks the features do not move. Encoders are fitted on training data only | [docs/leakage_checklist.md](docs/leakage_checklist.md), `tests/test_leakage.py` |
| Baseline | Logistic regression with balanced class weights | `src/sar/models/baseline.py` |
| Provisional Random Forest | Retrains the tuned settings from the team's model notebook on the training presentations. Threshold from out-of-fold training scores at 80% recall (GroupKFold by presentation); 2014J is never used to choose it | `src/sar/models/forest.py`, `scripts/train_forest.py` |
| SHAP explanations | Per-student contributions, a bar chart with background factors in a separate panel, and a plain-language sentence | `src/sar/explain/`, [docs/explanations.md](docs/explanations.md) |
| Similar-student recommender | Maps the inputs that raise a student's score to support types, and describes the 50 most similar training-presentation students from the same module | `src/sar/recommend/`, [docs/recommendations.md](docs/recommendations.md) |

## Current status (Sprint 2)

**On main**
- Sprint 1 foundation: data loading and cleaning, target and Day 30 population, time-based split, leakage checklist and test, engagement, assessment and registration features, and the baseline model.
- SCRUM-187: per-student SHAP explanations.
- SCRUM-188: intervention recommender.
- SCRUM-189: Streamlit app skeleton with cached loading.
- 91 automated tests (`pytest -q`).

**Not yet in the repository**
- Saved team model file and decision threshold (SCRUM-184, SCRUM-185, SCRUM-186). SCRUM-186 is the decision threshold ticket. Until these are in the repository, the app and explanations use the provisional Random Forest and its provisional threshold.
- Department risk table (SCRUM-178).
- Retention funnel table (SCRUM-177).
- Engagement profile table (SCRUM-183).

The app shows "not available yet" for the three tables until they are added.

## Results

Test presentation 2014J: 9,203 students, 40.7% At Risk. Figures are from [reports/metrics_baseline.json](reports/metrics_baseline.json) and [reports/model_forest_provisional.json](reports/model_forest_provisional.json).

| Model | Threshold | Recall | Precision | PR-AUC | ROC-AUC | Flagged |
|---|---|---|---|---|---|---|
| Baseline logistic regression | 0.50 | 0.626 | 0.632 | 0.625 | 0.733 | 40.3% |
| Random Forest (provisional) | 0.50 | 0.633 | 0.661 | 0.733 | 0.775 | 39.0% |
| Random Forest (provisional) | 0.404 | 0.781 | 0.566 | 0.733 | 0.775 | 56.2% |

The 0.404 threshold is provisional. It was chosen on the training presentations to reach 80% recall (out-of-fold recall 0.800, precision 0.626), and it flags 56% of 2014J students. Flagged shares at 0.50 are computed from the confusion matrices in the same files.

## Project structure

```
app/                  Streamlit app (streamlit_app.py)
.streamlit/           App settings: serve on localhost only
src/sar/              Installable package
  config.py           Paths, Day 30 cutoff, random seed
  data/               Loading, cleaning, target, split
  features/           Engagement, assessment, registration features
  models/             Baseline and provisional Random Forest
  explain/            SHAP explanations, phrases, chart
  recommend/          Support-type mapping, similar students, suggestions
  service/            App data loading and batch views
  viz/                Shared chart helpers
scripts/              Pipeline entry points
notebooks/            Analysis notebooks
tests/                Automated tests, all on synthetic data
docs/                 Decisions, checklists and method notes
reports/              Metrics JSON and exported figures
data/                 Local data only, ignored by git (raw/, interim/, processed/)
models/               Trained model files, created by scripts/train_forest.py, ignored by git
```

## Getting started

Requires Python 3.10 or later (`pyproject.toml`); tested with Python 3.12.

```
git clone https://github.com/SruthiGS-Gito/sar-ai-early-warning-intervention-recommender.git
cd sar-ai-early-warning-intervention-recommender
python -m venv .sar-env
.sar-env\Scripts\activate          # Windows
# source .sar-env/bin/activate     # macOS or Linux
pip install -r requirements.txt
pip install -e .
```

Place the seven OULAD CSV files in `data/raw/`, then run in this order:

| Step | Command | Output |
|---|---|---|
| 1 | `python scripts/make_interim.py` | Join-key checks; cleaned tables in `data/interim/` |
| 2 | `python scripts/run_baseline.py` | Day 30 feature matrix in `data/processed/`; `reports/metrics_baseline.json` |
| 3 | `python scripts/train_forest.py` | Provisional model in `models/`; `reports/model_forest_provisional.json` |
| 4 | `pytest -q` | Runs the test suite |

## Running the app

```
streamlit run app/streamlit_app.py
```

The app opens at http://localhost:8501 and is served on this machine only (`.streamlit/config.toml`), because it shows individual student records. Hosted demo: not available.

| Tab | Needs |
|---|---|
| Student Lookup | Feature matrix (step 2) and provisional model (step 3) |
| Department Dashboard | `data/processed/department_risk.csv` and `data/processed/student_profiles.csv` (SCRUM-178, SCRUM-183) |
| Retention Funnel | `data/processed/retention_funnel.csv` (SCRUM-177) |
| Batch Report | Feature matrix and provisional model |

Expected table columns and display rules: [docs/app.md](docs/app.md).

## Responsible use and limitations

- **Patterns, not effects.** OULAD records no interventions. Suggested support and the figures for similar students describe patterns among past students; they do not show that any support works.
- **Background factors.** Gender, region, highest education, deprivation band, age band and disability are never used as reasons or to drive a suggestion. The provisional model still uses them as inputs, so they still affect the score: about 18% of score movement on a sample of 2014J students ([docs/explanations.md](docs/explanations.md)).
- **A prioritised list, not a verdict.** The provisional threshold flags about 56% of students. Use the scores to decide whom to contact first.
- **Provisional model.** The Random Forest and its threshold are stand-ins until the team model and decision threshold (SCRUM-184, SCRUM-185, SCRUM-186) are in the repository.
- **Hand-written mapping.** Which risk drivers point to which support type is an assumption that needs review by student support staff.
- **Scores are not probabilities.** They rank students by risk.

## Team

| Member | Workstreams |
|---|---|
| Abhijith | Data loading, cleaning and dataset decisions; department risk dashboard; retention funnel |
| Ancy | Outcome and attendance analysis; at-risk indicator combinations; semester retention analysis |
| Gouri | Engagement metrics and inactivity streaks; engagement clustering and profiles |
| Midhula | Target definition, assessment features and baseline model; Random Forest and XGBoost training, tuning and decision threshold |
| Sruthi | Repository, environment and sprint planning; cutoff, split and leakage checks; SHAP explanations, intervention recommender and Streamlit app |

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before your first commit. Work is tracked in Jira (project SAR-AI), and every branch, commit and pull request carries its ticket key.

## Acknowledgements

- The Open University and the OULAD authors, Kuzilek, Hlosta and Zdrahal, for publishing the dataset.
- TrackGenesis, for the internship project.

## Licence

This repository does not include a licence file. The OULAD data is licensed separately, under CC BY 4.0, and is not included in the repository.
