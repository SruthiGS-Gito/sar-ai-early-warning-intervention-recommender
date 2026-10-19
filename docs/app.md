# Streamlit app (SCRUM-189, ticket 302)

## How to run
```
python scripts/run_baseline.py     # Day 30 feature matrix (once)
python scripts/train_forest.py     # provisional model (once)
streamlit run app/streamlit_app.py
```
- The app opens at http://localhost:8501.
- `.streamlit/config.toml` serves it on this machine only, because the app shows individual student records. Do not change the address to share it.
- `SAR_DATA_DIR` and `SAR_MODELS_DIR` point the app at other folders.

## Tabs
| Tab | Shows | Source |
|---|---|---|
| Student Lookup | Pick a presentation, module and student to see the risk score, the SHAP chart (background factors in their own panel), the plain-language sentence, and up to three support types with similar-student context | SCRUM-187, SCRUM-188 |
| Department Dashboard | Department risk table and student profiles table, if present | tickets 203 and 210 |
| Retention Funnel | Retention funnel table, if present | ticket 207 |
| Batch Report | For a presentation: students, flagged count and share, by-module table, how scores spread around the threshold. The flagged list (closed expander, CSV download) and an optional 3D view of average score by activity and score bands (bands under 20 students hidden) are opt-in | provisional model |

Display rules:
- Aggregated by default. No student is shown until one is chosen, and the flagged list stays closed.
- Every view says the model is PROVISIONAL.
- Students from training presentations are labelled, because their scores are optimistic.

## Tables from other tickets
The app does not build these tables and does not import their code. It reads a CSV from `data/processed/` if one exists. The format below is a proposal for the owners to confirm or change.

| File | Ticket | Required columns |
|---|---|---|
| `department_risk.csv` | 203 (SCRUM-178) | `code_module`, plus any aggregate columns |
| `retention_funnel.csv` | 207 (SCRUM-177) | `stage`, `students` |
| `student_profiles.csv` | 210 (SCRUM-183) | `profile`, plus any aggregate columns |

How each case is handled:
- **Missing file:** "not available yet", with the ticket number.
- **Missing columns:** a message naming them.
- **File with an `id_student` column:** refused, because the app shows aggregated tables only.
- **None of these cases** crashes the app.

## Caching and speed
- **Cached:** the matrix and model load once, and so do the SHAP explainer and the similar-student pool. Scores are cached per presentation.
- **On real data:** first load is about 9 s; later reruns take about 0.1 s and a student lookup about 1.5 s.
- **Not in the Batch Report:** per-student SHAP for every flagged student (about 0.25 s each), so reasons open in Student Lookup.

## Tests
`tests/test_app.py` runs the app headless with `streamlit.testing.v1.AppTest` on synthetic data:
- with nothing available
- with a synthetic matrix and model, looking up a student and switching on the 3D view
- with a training presentation
- with valid and student-level team tables

## Limits
- The model is provisional until ticket 214.
- The team tables use a proposed format until their owners confirm it.
- The flagged-list download writes student IDs to the user's machine. It is meant for staff follow-up and should be handled under the project's data rules.
