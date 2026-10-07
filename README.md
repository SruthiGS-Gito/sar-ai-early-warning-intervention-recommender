# Student Academic Risk Early Warning and Intervention Recommender

TrackGenesis internship project 8 (team 8). The aim is to flag students at risk of failing or withdrawing using only their first 30 days of activity, explain each flag, and recommend an intervention.

## Status

Review 1 (Sprint 1, 5 to 9 Oct 2026): data foundation, Day 30 cutoff, leakage control, first engagement and assessment features, and a baseline Logistic Regression model. Later sprints add the main models, SHAP explanations, the intervention recommender and the Streamlit console. Their folders are added when the work starts.

## Dataset

Open University Learning Analytics Dataset (OULAD). Reference: Kuzilek, Hlosta and Zdrahal (2017), Scientific Data 4:170171. The data is not stored in this repository. Download it from the Open University analytics site and place the seven CSV files in `data/raw/`:

- assessments.csv
- courses.csv
- studentAssessment.csv
- studentInfo.csv
- studentRegistration.csv
- studentVle.csv
- vle.csv

## Setup

```
git clone https://github.com/SruthiGS-Gito/sar-ai-early-warning-intervention-recommender.git
cd sar-ai-early-warning-intervention-recommender
python -m venv .sar-env
.sar-env\Scripts\activate          # Windows
# source .sar-env/bin/activate     # macOS or Linux
pip install -r requirements.txt
pip install -e .
pytest
```

## Structure

- `src/sar/`: project code as an installable package
  - `config.py`: paths, table names, Day 30 cutoff, random seed
  - `data/`: loading, cleaning, target, split
  - `features/`: engagement, assessment, registration features
  - `models/`: baseline model
  - `viz/`: shared chart helpers
- `notebooks/`: exploration and charts (see `notebooks/README.md`)
- `tests/`: automated checks, including the leakage test
- `docs/`: decisions and checklists for review
- `data/`: local data only, ignored by git
- `reports/figures/`: exported charts

## Working together

Read `CONTRIBUTING.md` before your first commit. Work is tracked in Jira (project SAR-AI). Each branch and pull request carries its ticket number.

## Review 1 scope and owners

See `docs/review1_scope.md`.
