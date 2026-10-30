# Student Academic Risk Early Warning and Intervention Recommender

TrackGenesis internship project 8 (team 8). The system scores every enrolled student on Day 30 of a module, using only records dated on or before that day, and ranks students by calibrated risk of failing or withdrawing. Each flagged student receives a risk band, a plain-language explanation and a rule-based support suggestion, and a local Streamlit application presents the results for a support team with a fixed contact capacity.

The full write-up is in [docs/final_report.md](docs/final_report.md).

## Dataset

Open University Learning Analytics Dataset (OULAD): seven tables, seven modules (AAA to GGG) and four presentations (2013B, 2013J, 2014B, 2014J).

Licence and citation, as stated under "Rights statement and License" on the dataset's page (https://research.stem.open.ac.uk/ouanalyse/open-dataset-more/; the address https://analyse.kmi.open.ac.uk/open_dataset redirects to the Open University's analytics site, which links to it):
- License: "This dataset is released under CC-BY 4.0 license." The licence link on that page is https://creativecommons.org/licenses/by/4.0/.
- Citing the dataset: "When citing the dataset please use the following reference:" Kuzilek J., Hlosta M., Zdrahal Z. Open University Learning Analytics dataset Sci. Data 4:170171 doi: 10.1038/sdata.2017.171 (2017).

The data is not stored in this repository; only aggregated figures derived from it are. Download it from the Open University analytics site and place the seven CSV files in `data/raw/`: `assessments.csv`, `courses.csv`, `studentAssessment.csv`, `studentInfo.csv`, `studentRegistration.csv`, `studentVle.csv`, `vle.csv`.

## Method in brief

- **Population and target.** One row per module registration. Students who unregistered on or before Day 30 are excluded. The target is 1 for Fail or Withdrawn and 0 for Pass or Distinction.
- **Split.** Trained on 2013B, 2013J and 2014B; validated on 2014J. Students who appear in both sets are removed from training.
- **Model.** A Random Forest on 24 Day 30 inputs, with isotonic calibration fitted on out-of-fold training scores. The six protected characteristics (gender, region, highest education, deprivation band, age band, disability) are excluded from the inputs.
- **Flagging.** Students are ranked by calibrated risk and the highest-risk share that the team can contact is flagged. Risk bands (Low, Medium, High) start at cutoffs chosen on out-of-fold training scores.
- **Explanations and support.** SHAP contributions give each student's main reasons; a rule-based mapping turns them into one of five support types.

## Validation results (2014J)

Validation on 2014J, 9,203 students with an at-risk prevalence of 0.407, from `models/final/metrics.json`. Intervals are 95% intervals from 1,000 bootstrap resamples (seed 42).

| Measure | Value |
|---|---|
| ROC-AUC | 0.766 (0.756 to 0.776) |
| PR-AUC | 0.721 (0.707 to 0.735) |
| Brier score, calibrated (raw) | 0.188 (0.191) |
| Expected calibration error, calibrated (raw), 10 bins | 0.026 (0.058) |
| Highest-risk 20% flagged | 1,841 students |
| Precision at 20% capacity | 0.800 |
| Recall at 20% capacity | 0.393 |
| Lift at 20% capacity | 1.96 |

2014J was used to compare model configurations and is a validation set rather than an untouched test set. Leave-one-presentation-out results, the full capacity table and the fairness audit are in [docs/final_model.md](docs/final_model.md).

## The application

`streamlit run app/streamlit_app.py` opens the application at http://localhost:8501. It is served on the local machine only (`.streamlit/config.toml`) because it shows individual student records. It opens on the validation presentation, 2014J, and labels training presentations as in-sample.

| Page | Shows |
|---|---|
| Overview | Summary cards, the risk-band distribution, the main reasons behind flags and the ranked list of flagged students |
| Student | One student's band and rank, the observed outcome of that band in the module, Day 30 indicators, the reasons behind the score and the support suggestions |
| Modules | Risk and band shares by module, at-risk rates by intake and activity, student profiles and the retention funnel |
| Support | Flagged students by suggested support type, overall and by module |
| About | Model, validation design, risk bands, limitations, data licence and rebuild commands |

Details: [docs/app.md](docs/app.md).

## How to run

Requires Python 3.12 or later (`pyproject.toml`), because shap 0.54.1 and numpy 2.5.3 do; the committed results were produced with Python 3.12.10. `requirements.txt` pins scikit-learn, pandas and numpy to the versions recorded in `models/final/config.json`, so that the saved model loads with the libraries it was trained with.

```
git clone https://github.com/SruthiGS-Gito/sar-ai-early-warning-intervention-recommender.git
cd sar-ai-early-warning-intervention-recommender
python -m venv .sar-env
.sar-env\Scripts\activate          # Windows
# source .sar-env/bin/activate     # macOS or Linux
pip install -r requirements.txt
pip install -e .
```

Place the seven OULAD CSV files in `data/raw/`, then run from the repository root, in this order:

| Step | Command | Output |
|---|---|---|
| 1 | `python scripts/make_interim.py` | Join-key checks; cleaned tables in `data/interim/` (used by the notebooks) |
| 2 | `python scripts/run_baseline.py` | Day 30 feature matrix in `data/processed/`; `reports/metrics_baseline.json` |
| 3 | `python scripts/train_final.py` | Final model and calibrator in `models/final/` (binaries git-ignored); rewrites `config.json` and `metrics.json` there |
| 4 | `python scripts/write_model_summary.py` | Regenerates `docs/final_model.md` from the two JSON files |
| 5 | `python scripts/precompute_shap.py` | Optional. SHAP explanations for 2014J in `models/final/shap/` (git-ignored). `--presentation 2013J` or `--all` for other presentations; `--jobs` sets the number of worker processes |
| 6 | `python scripts/build_team_tables.py` | Department, funnel and profile tables in `data/processed/`; rewrites `reports/team_tables.json` and the figures block in `docs/app.md` |
| 7 | `pytest` | Runs the test suite |
| 8 | `streamlit run app/streamlit_app.py` | Starts the application |

Steps 3, 4 and 6 rewrite tracked files; `git diff` shows any difference from the committed results. `SAR_DATA_DIR` points the scripts and the application at another data folder, and `SAR_MODELS_DIR` points the application at another models folder.

The tests run on synthetic data, so `pytest` works on a fresh clone without the dataset. Two tests in `tests/test_final.py` check the built model against the committed figures and are skipped until steps 2 and 3 have run.

## Repository structure

```
app/streamlit_app.py      Streamlit application (five pages)
.streamlit/config.toml    Local-only serving and theme
src/sar/                  Installable package
  config.py               Paths, Day 30 cutoff, seed, protected features, risk bands
  data/                   Loading, cleaning, target and Day 30 population, temporal split
  features/               Engagement, assessment and registration features
  models/                 Baseline, Random Forest, final model, model summary
  explain/                SHAP explanations, factor phrases, explanation chart
  recommend/              Support types and mapping, similar-student pool, recommender
  service/                Data loading, scoring, flagging, priority list and figures for the app
  tables/                 Department risk, retention funnel and student profile tables
  viz/                    Chart helpers for the exported figures
scripts/                  Command-line entry points (see How to run)
tests/                    Automated tests on synthetic data
notebooks/                Analysis and modelling notebooks (see notebooks/README.md)
docs/                     Decisions, method notes, generated summaries and the final report
reports/                  Baseline metrics, team-table summary and exported figures
models/final/             config.json and metrics.json (tracked); model binaries and SHAP values (ignored)
data/                     raw/, interim/ and processed/; local only, ignored by git
```

## Verifying the results

| Claim | Supported by |
|---|---|
| Model configuration, inputs, excluded features, cutoffs, file hashes | `models/final/config.json` |
| Validation, calibration, capacity, bootstrap, leave-one-presentation-out and fairness figures | `models/final/metrics.json` |
| Department risk, retention funnel and student profile figures | `reports/team_tables.json` |
| Readable summaries | `docs/final_model.md` (generated from the two JSON files) and the figures block in `docs/app.md` (generated from `reports/team_tables.json`) |
| Summaries match their sources; licence and citation quoted exactly | `tests/test_docs.py` |
| Application ranking reproduces the committed capacity table | `tests/test_final.py` |
| No feature uses data dated after Day 30 | `tests/test_leakage.py` and [docs/leakage_checklist.md](docs/leakage_checklist.md) |
| Complete method, results and limitations | [docs/final_report.md](docs/final_report.md) |

Notebooks, by author:
- Sruthi G S: `164_SGS_day30_insights.ipynb`, `215_SGS_shap_explanations.ipynb`, `216_SGS_intervention_recommender.ipynb`
- Ancy Michael: `206_AM_indicator_combinations.ipynb`
- Abhijith M S and Gouri M: `student_academic_risk.ipynb`
- Gouri M: `Engagement profile charts and export.ipynb`
- Midhula M S: `xgboost-random-forest (2).ipynb`, `xgboost-random-forest (4).ipynb`

## Limitations

2014J served to compare model configurations, so its figures may be slightly optimistic; the leave-one-presentation-out results show the spread across presentations. OULAD records no interventions, so the support suggestions are rule-based and their effect is unknown. At the same cutoff, recall is lower for women than for men and for students aged 35 to 55 than for younger students. In EEE and GGG no assessment is due by Day 30, so most students there have no early score. The calibrator is fitted across all modules, and OULAD holds no score release dates, so a score counted on Day 30 may not yet have been known to staff. All results come from one distance-learning institution.

## Working together

Read [CONTRIBUTING.md](CONTRIBUTING.md) before your first commit. Work is tracked in Jira (project SAR-AI), and each branch and pull request carries its ticket number. The Review 1 scope and owners are in [docs/review1_scope.md](docs/review1_scope.md).
