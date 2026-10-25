# Streamlit app (SCRUM-189, ticket 302)

## How to run
```
python scripts/run_baseline.py       # Day 30 feature matrix (once)
python scripts/train_final.py        # final model into models/final/ (once, about a minute)
python scripts/precompute_shap.py    # optional: explanations for the validation presentation
streamlit run app/streamlit_app.py
```
- The app opens at http://localhost:8501.
- `.streamlit/config.toml` serves it on this machine only, because the app shows individual student records. Do not change the address to share it.
- `SAR_DATA_DIR` and `SAR_MODELS_DIR` point the app at other folders.

## Model
The app uses the final model in `models/final/`: a Random Forest on 24 Day 30 inputs, without protected attributes, with isotonic calibration. It is loaded through the hash-checked loader in `sar.models.final`. Its configuration, cutoffs and validation results are listed in [final_model.md](final_model.md), which is generated from `models/final/config.json` and `metrics.json`.

Each student gets:
- a **calibrated risk score**, used for ranking and banding and not shown as a number;
- a **risk band**: Low, Medium or High (see below);
- a **rank** within the presentation, 1 being the highest risk.

### Risk bands
Defined once in `sar.config.RISK_BANDS`. Each band starts at a cutoff from `models/final/config.json`; the cutoffs are chosen on out-of-fold training scores, never on the validation presentation.

| Band | Calibrated score |
|---|---|
| Low | below the `wide_net` cutoff (80% out-of-fold recall) |
| Medium | from the `wide_net` cutoff to below the `top_20` cutoff |
| High | from the `top_20` cutoff |

Wherever a student's risk is shown, the app shows the band, not a score. The current cutoff values are in [final_model.md](final_model.md#risk-bands).

### Flagging rules
The Batch Report offers two rules.
- **Capacity ranking (default):** rank the cohort by calibrated risk and flag the top share chosen on the slider (5% to 50%). This matches how a support team with a fixed number of contacts works. The precision and recall shown for each share are the validation figures from the capacity table in `metrics.json`.
- **Wide net (fixed cutoff):** flag everyone at or above the `wide_net` cutoff. Because at-risk rates differ between intakes, a cutoff fixed in advance does not flag a stable share of a new cohort; see the limitations in [final_model.md](final_model.md#limitations).

## Tabs
| Tab | Shows | Source |
|---|---|---|
| Student Lookup | Pick a presentation, module and student to see the risk band, the rank, the SHAP chart, the plain-language sentence, and up to three support types with similar-student context | SCRUM-187, SCRUM-188 |
| Department Dashboard | A status card for the department risk table and one for the student profiles table, with the table when present; a pointer to Batch Report for model-based counts | tickets 203 and 210 |
| Retention Funnel | A status card for the retention funnel table, with the table when present | ticket 207 |
| Batch Report | For a presentation: the flagging rule, students, flagged count and share with the validation precision and recall for that rule, a by-module table with students per risk band, and how calibrated risk spreads across the bands. The flagged list (closed expander, CSV download with band and rank) and an optional 3D view of average risk by activity and score bands (bands under 20 students hidden) are opt-in | final model |

Display rules:
- Aggregated by default. No student is shown until one is chosen, and the flagged list stays closed.
- A student's risk is shown as a band and a rank, not as a number.
- Students from training presentations are labelled, because their scores are optimistic.
- Protected attributes are not model inputs and never appear in explanations or suggestions.

## Tables from other tickets
The app does not build these tables and does not import their code. It reads a CSV from `data/processed/` if one exists. The format below is a proposal for the owners to confirm or change.

| File | Ticket | Required columns |
|---|---|---|
| `department_risk.csv` | 203 (SCRUM-178) | `code_module`, plus any aggregate columns |
| `retention_funnel.csv` | 207 (SCRUM-177) | `stage`, `students` |
| `student_profiles.csv` | 210 (SCRUM-183) | `profile`, plus any aggregate columns |

Each table has a status card showing what it is, the owner ticket, the expected file and required columns, and its current status. How each case is handled:
- **Missing file:** "not available yet", with the ticket number.
- **Missing columns:** a message naming them.
- **File with an `id_student` column:** refused, because the app shows aggregated tables only.
- **None of these cases** crashes the app.

## Caching and speed
- **Cached once:** the matrix, the model and calibrator, the SHAP explainer, the similar-student pool and any precomputed explanations. Scores are cached per presentation; an explanation is cached per student.
- **Precomputed explanations:** SHAP takes about 0.25 s per student on one core, so a whole presentation is too slow to explain while the app runs. `scripts/precompute_shap.py` computes a presentation's SHAP values once (in parallel) into `models/final/shap/`. Student Lookup reads them when they exist and were computed for the same model file (checked by hash); otherwise it computes the one student's explanation and caches it.

## Tests
`tests/test_app.py` runs the app headless with `streamlit.testing.v1.AppTest` on synthetic data:
- with nothing available
- with a synthetic matrix and final model: capacity ranking by default, a student lookup with a risk band, the wide-net rule and the 3D view
- with precomputed explanations
- with a training presentation
- with valid and student-level team tables

## Limits
- The team tables use a proposed format until their owners confirm it.
- The flagged-list download writes student IDs to the user's machine, and `models/final/shap/` holds per-student values. Both are for staff follow-up on this machine only and should be handled under the project's data rules; neither is committed.
- Validation figures come from one presentation (2014J). Their uncertainty and the spread across presentations are in [final_model.md](final_model.md).
