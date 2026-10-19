# Per-student explanations (SCRUM-187, ticket 215)

For any student in the Day 30 population, `sar.explain` gives:
- a SHAP bar chart, with background factors in a separate panel
- a plain-language sentence

## How to run
```
python scripts/run_baseline.py     # builds data/processed/day30_feature_matrix.csv (once)
python scripts/train_forest.py     # trains the provisional model into models/ (git-ignored)
pytest -q
```
```python
from sar.config import MODELS_DIR
from sar.models.forest import load_model
from sar.explain.shap_explain import explain_student
from sar.explain.chart import plot_explanation

model, meta = load_model(MODELS_DIR / "forest_provisional.json")
expl = explain_student(model, one_row_of_features, meta["threshold"]["threshold"])
expl.sentence              # plain-language summary
plot_explanation(expl)     # matplotlib Figure
```
Worked example: `notebooks/215_SGS_shap_explanations.ipynb`. It shows aggregates and a synthetic student only.

## Model: PROVISIONAL
The model is a Random Forest with the tuned settings from tickets 212 and 217 (SCRUM-184, SCRUM-185; cell 69 of `notebooks/xgboost-random-forest (2).ipynb`, commit c4c3ba6). It is retrained on the training presentations (2013B, 2013J, 2014B) by `scripts/train_forest.py`. It is a stand-in until the team model (212, 217) and the decision threshold (214, SCRUM-186) are in the repo, and every chart and document says so.

- **Threshold:** the highest out-of-fold threshold that reaches 80% recall, using GroupKFold with one fold per training presentation. 2014J is never used to choose it. Current value 0.404.
- **2014J at the threshold:** recall 0.781, precision 0.566, 56% of students flagged.
- **2014J overall:** PR-AUC 0.733, ROC-AUC 0.775. The baseline scores 0.625 and 0.733.
- **Files:** model files stay in `models/`, which is git-ignored, and are hash-checked on load. `reports/model_forest_provisional.json` holds aggregate metrics only.

## Input groups
| Group | Inputs | In the sentence? |
|---|---|---|
| Behaviour | site activity, inactivity streaks, submissions, scores, delays, missed work, registration timing | Yes, as reasons |
| Study | previous attempts, credits studied | Yes, as reasons |
| Course | module, assessments due by Day 30 | Only as "course-level patterns, not about the student" |
| Background | gender, region, highest education, deprivation band, age band, disability | Never. Shown in a separate grey panel |

Sentence rules:
- A sentence names at most 3 inputs that raise the score and 2 that lower it.
- It names one input per idea; for example, average score and weighted score count once.
- Each named input must move the score by at least 0.005.

## Method notes
- **SHAP mode:** `shap.TreeExplainer` in tree_path_dependent mode, so the expected score comes from the training data stored in the trees. No validation data is used. Contributions are checked to add up to the score (gap < 1e-6).
- **Categories:** one-hot columns are summed back into their input, so "Module" is one contribution.
- **Speed:** about 0.3 s per student. A batch of 300 takes about 40 s, so batch views should explain only the students who are opened.

## Assumptions and limits
- The score ranks risk; it is not a probability. With balanced class weights, the starting score is 0.50, not the at-risk rate.
- Background factors never appear as reasons, but the model still uses them. On a sample of 500 2014J students they make up about 18% of all score movement, and highest education is the second largest input. Whether to keep them as model inputs is for the team model (tickets 212 and 217) to decide.
- Contributions describe how the model reaches its score. They do not say what caused an outcome or what would change it.
- SHAP splits credit between related inputs, so a single input can look smaller than the idea it shares.
- The threshold folds are not strictly time-ordered: 2013B is scored by a model trained on later training presentations.
