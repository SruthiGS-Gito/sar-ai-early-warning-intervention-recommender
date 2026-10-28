# Per-student explanations (SCRUM-187, ticket 215)

For any student in the Day 30 population, `sar.explain` gives:
- a SHAP bar chart of the inputs that move the student's score
- a plain-language sentence that names the student's risk band and the main inputs behind it

## How to run
```
python scripts/run_baseline.py       # builds data/processed/day30_feature_matrix.csv (once)
python scripts/train_final.py        # trains the final model into models/final/ (binaries git-ignored)
python scripts/precompute_shap.py    # optional: batch explanations for the validation presentation
pytest -q
```
```python
from sar.models.final import load_final, predict
from sar.service.views import assign_risk_band
from sar.explain.shap_explain import explain_student
from sar.explain.chart import plot_explanation

fitted, config = load_final()                               # hash-checked
raw, calibrated = predict(fitted, one_row_of_features)
band = assign_risk_band(calibrated, fitted["cutoffs"])[0]   # Low, Medium or High
expl = explain_student(fitted["model"], one_row_of_features[fitted["features"]], band=band)
expl.sentence              # plain-language summary
plot_explanation(expl)     # matplotlib Figure
```
For many students, `explain_batch(model, X, n_jobs=-1)` computes the same contributions in parallel, and `explanation_from_contributions` rebuilds an Explanation from stored values without recomputing them.

Worked example: `notebooks/215_SGS_shap_explanations.ipynb`. It shows aggregates and a synthetic student only, and was produced with the earlier 30-input model; the method is unchanged.

## Model
Explanations describe the final model: a Random Forest on 24 Day 30 inputs with isotonic calibration. Its configuration and validation results are in [final_model.md](final_model.md).

- **What is explained:** SHAP contributions add up to the model's raw score. The calibrator is a monotonic map from the raw score to the calibrated risk score, so it changes the scale but not which inputs raise or lower risk.
- **What is shown:** the risk band from the calibrated score (`sar.config.RISK_BANDS`), not a number. Sentences and chart titles never state a raw score or percentage.

## Input groups
| Group | Inputs | In the sentence? |
|---|---|---|
| Behaviour | site activity, inactivity streaks, submissions, scores, delays, missed work, registration timing | Yes, as reasons |
| Study | previous attempts, credits studied | Yes, as reasons |
| Course | module, assessments due by Day 30 | Only as "course-level patterns, not about the student" |
| Background | gender, region, highest education, deprivation band, age band, disability | Never. They are not inputs of the final model, so they have no contribution |

Sentence rules:
- A sentence names at most 3 inputs that raise the score and 2 that lower it.
- It names one input per idea; for example, average score and weighted score count once.
- Each named input must move the score by at least 0.005.

## Method notes
- **SHAP mode:** `shap.TreeExplainer` in tree_path_dependent mode, so the expected score comes from the training data stored in the trees. No validation data is used. Contributions are checked to add up to the score (gap < 1e-6).
- **Categories:** one-hot columns are summed back into their input, so "Module" is one contribution.
- **Speed:** about 0.25 s per student on one core. The app therefore reads batch values precomputed by `scripts/precompute_shap.py` and caches the explainer and each student's explanation (see [app.md](app.md#caching-and-speed)).
- **Protected attributes:** `tests/test_protected_outputs.py` checks, for a sample of students, that no protected attribute appears as an input, a driver, a phrase, a chart label or a word in the text.

## Assumptions and limits
- Contributions describe how the model reaches its score. They do not say what caused an outcome or what would change it.
- SHAP splits credit between related inputs, so a single input can look smaller than the idea it shares.
- Removing protected attributes as inputs does not remove all association with them: other inputs can correlate with them. Recall still differs by gender and age band; see the fairness section of [final_model.md](final_model.md).
- The out-of-fold folds are not strictly time-ordered: 2013B is scored by a model trained on later training presentations.
