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
| Department Dashboard | Department risk for a chosen presentation (table and stacked band-share chart) and the student profiles (table, descriptions and at-risk rate chart); a pointer to Batch Report for flagged counts | tickets 203 and 210 |
| Retention Funnel | The six-stage funnel for a chosen presentation and module (chart and table), with a note that it is retrospective | ticket 207 |
| Batch Report | For a presentation: the flagging rule, students, flagged count and share with the validation precision and recall for that rule, a by-module table with students per risk band, and how calibrated risk spreads across the bands. The flagged list (closed expander, CSV download with band and rank) and an optional 3D view of average risk by activity and score bands (bands under 20 students hidden) are opt-in | final model |

Display rules:
- Aggregated by default. No student is shown until one is chosen, and the flagged list stays closed.
- A student's risk is shown as a band and a rank, not as a number.
- Students from training presentations are labelled, because their scores are optimistic.
- Protected attributes are not model inputs and never appear in explanations or suggestions.

## Team tables
Three aggregated tables, built by one command after the feature matrix and the final model:
```
python scripts/build_team_tables.py
```
The code is in `sar.tables`. The script writes the tables to `data/processed/` and the summary used below to `reports/team_tables.json`, and regenerates the figures in this section from that summary.

| File | Ticket | One row per | Columns |
|---|---|---|---|
| `department_risk.csv` | 203 (SCRUM-178) | module and presentation, plus an "All modules" row per presentation | `code_module`, `code_presentation`, `score_source`, `students`, `at_risk_rate`, `share_low`, `share_medium`, `share_high`, `suppressed` |
| `retention_funnel.csv` | 207 (SCRUM-177) | stage, module and presentation, plus "All modules" rows | `code_module`, `code_presentation`, `stage_order`, `stage_key`, `stage`, `students`, `share_of_registered`, `share_of_previous`, `applicable`, `suppressed` |
| `student_profiles.csv` | 210 (SCRUM-183) | profile and scope (validation or training) | `scope`, `presentations`, `profile`, `cluster`, `description`, `students`, `share_of_scope`, `at_risk_rate`, `mean_<feature>` for each profile feature, `suppressed` |

### Definitions
**Department risk.** A department is one module in one presentation. `students` is the number of students in the Day 30 population (not unregistered by Day 30). `at_risk_rate` is the observed share whose final outcome was Fail or Withdrawn. The band shares are the shares of students in each risk band (Low, Medium, High; `sar.config.RISK_BANDS`) of the final model's calibrated score. For training presentations (`score_source` = training) the scores are in-sample: the model learned from those students' outcomes, so their band shares are better separated than they would be for a new intake.

**Retention funnel.** A retrospective, descriptive view. The stages are nested: each counts the students of the previous stage who also meet its condition.
1. Registered: every enrolment in `studentInfo`.
2. Still registered after Day 30: not unregistered on or before Day 30.
3. Active in the VLE by Day 30: at least one day of VLE activity by Day 30.
4. Submitted an assessment by Day 30: at least one non-exam assessment submitted by Day 30.
5. Did not withdraw: final outcome is not Withdrawn.
6. Passed: final outcome is Pass or Distinction.

Stages 1 to 4 use data available on Day 30. Stages 5 and 6 use final outcomes; they describe what happened after Day 30 and are not model inputs. Where a module presentation has no assessment due by Day 30, stage 4 would measure the course schedule rather than the student, so it is marked not applicable (`applicable` = False) and the stage 3 count is carried forward into stages 4 to 6.

**Student profiles.** KMeans clusters of five Day 30 engagement and submission features: `total_clicks` (log-transformed before scaling), `active_days`, `trailing_inactive_days`, `n_submitted` and `n_missed`. None of the six protected attributes, no assessment score and no outcome is a clustering input. The scaler and the clusters are fitted on the training presentations (after removing students who also appear in the validation presentation); validation students are then assigned to the nearest cluster.
- **Number of clusters:** the k from 3 to 5 with the highest silhouette score on the training rows (a fixed sample of 10,000 rows, seed 42). The chosen solution was then checked for interpretability: each cluster has a distinct centroid pattern and a distinct at-risk rate.
- **Names:** written after inspecting the centroids and at-risk rates, and assigned by rules on the centroids (`sar.tables.profiles`): the cluster with the fewest active days is *Little VLE activity*; of the rest, the one with the most missed assessments is *Missed early assessment*; then the one with the most clicks is *Highly engaged*; the remaining cluster is *Moderately engaged*.
- **Descriptions:** one sentence per cluster, filled in from its mean values and observed at-risk rate.
- **At-risk rate:** from final outcomes, attached after clustering. It is not a clustering input.

Credit: the feature choice and the range of cluster counts draw on Gouri's engagement profile notebook (`notebooks/Engagement profile charts and export.ipynb`), which clusters Day 30 clicks, active days, inactivity streaks, scores and submissions with KMeans and compares cluster counts by silhouette and inertia. This implementation differs from it in four ways: it fits on the training presentations only, leaves out assessment scores, tests k from 3 to 5, and uses its own profile names, because names such as "Improving" describe a change over time that a single Day 30 snapshot cannot show (a caution the notebook also gives).

### Privacy and display rules
- Every row is a group of students. No file holds a student identifier, and the app refuses any team table with an `id_student` column.
- Small cells: a group with fewer than `MIN_CELL` (20) students, or a funnel count from 1 to 19, is kept with its figures blank and `suppressed` set. In the funnel, enough further modules are hidden in the same presentation and stage that the "All modules" total cannot reveal a hidden count. Differences between consecutive stages of one module are not suppressed.
- The tables are generated data and stay in `data/processed/`, which git ignores. `reports/team_tables.json` holds only the aggregate figures quoted below and is tracked, so the documentation can be checked against it.
- The OULAD data are published by the Open University under the Creative Commons Attribution 4.0 licence (CC BY 4.0); see the dataset reference in the README.

How the app handles each table:
- **Missing file:** a status card with the ticket and the build command.
- **Missing columns:** a message naming them.
- **File with an `id_student` column:** refused.
- **None of these cases** crashes the app.

### Current figures
<!-- team-tables:start. Generated by scripts/build_team_tables.py from reports/team_tables.json; do not edit by hand. -->

Figures from the last run of `scripts/build_team_tables.py`. Groups and counts under 20 students are hidden. Source: OULAD (Kuzilek, Hlosta and Zdrahal 2017), CC BY 4.0.

#### Department risk, 2014J (validation)

| Module | Students | At-risk rate | Low | Medium | High |
|---|---:|---:|---:|---:|---:|
| AAA | 348 | 27% | 57% | 31% | 12% |
| BBB | 1,786 | 36% | 37% | 55% | 8% |
| CCC | 2,014 | 50% | 41% | 34% | 25% |
| DDD | 1,463 | 46% | 38% | 36% | 26% |
| EEE | 1,020 | 33% | 50% | 36% | 14% |
| FFF | 1,857 | 40% | 45% | 38% | 17% |
| GGG | 715 | 38% | 49% | 44% | 8% |
| All modules | 9,203 | 41% | 43% | 40% | 17% |

Rows in the table (all presentations): 26; hidden: 0.

#### Retention funnel, all modules

Students at each stage, with the share of registered students in brackets.

| Stage | 2013B | 2013J | 2014B | 2014J |
|---|---:|---:|---:|---:|
| Registered | 4,684 (100%) | 8,845 (100%) | 7,804 (100%) | 11,260 (100%) |
| Still registered after Day 30 | 4,111 (88%) | 7,655 (87%) | 6,497 (83%) | 9,203 (82%) |
| Active in the VLE by Day 30 | 3,984 (85%) | 7,372 (83%) | 6,272 (80%) | 8,982 (80%) |
| Submitted an assessment by Day 30 | 3,660 (78%) | 6,898 (78%) | 5,764 (74%) | 8,466 (75%) |
| Did not withdraw | 3,048 (65%) | 5,925 (67%) | 4,711 (60%) | 7,015 (62%) |
| Passed | 2,077 (44%) | 4,346 (49%) | 3,286 (42%) | 5,351 (48%) |

Submission stage not applicable (no assessment due by Day 30; previous count carried forward, also in the totals above): EEE 2013J, EEE 2014B, EEE 2014J, GGG 2013J, GGG 2014B, GGG 2014J. Rows in the table: 156; hidden counts: 0.

#### Student profiles

| k | Silhouette | Smallest cluster (share of training rows) |
|---:|---:|---:|
| 3 | 0.335 | 14% |
| 4 | 0.365 | 7% |
| 5 | 0.346 | 7% |

Chosen: k = 4.

| Profile | 2014J students | Share | At-risk rate | Mean clicks | Mean active days | Mean days without activity before Day 30 | Mean submitted | Mean missed | At-risk rate (training) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Moderately engaged | 5,008 | 54% | 43% | 194 | 10.3 | 3.7 | 0.85 | 0.00 | 46% |
| Missed early assessment | 472 | 5% | 86% | 142 | 7.7 | 8.9 | 0.03 | 1.00 | 91% |
| Highly engaged | 3,186 | 35% | 25% | 749 | 27.1 | 0.7 | 1.30 | 0.00 | 30% |
| Little VLE activity | 537 | 6% | 73% | 12 | 1.3 | 26.7 | 0.59 | 0.28 | 73% |

- **Moderately engaged**: On average 194 clicks on 10.3 active days by Day 30, 3.7 days without activity just before it, 0.85 assessments submitted and 0.00 missed; 43% went on to fail or withdraw.
- **Missed early assessment**: On average 142 clicks on 7.7 active days by Day 30, 8.9 days without activity just before it, 0.03 assessments submitted and 1.00 missed; 86% went on to fail or withdraw.
- **Highly engaged**: On average 749 clicks on 27.1 active days by Day 30, 0.7 days without activity just before it, 1.30 assessments submitted and 0.00 missed; 25% went on to fail or withdraw.
- **Little VLE activity**: On average 12 clicks on 1.3 active days by Day 30, 26.7 days without activity just before it, 0.59 assessments submitted and 0.28 missed; 73% went on to fail or withdraw.

<!-- team-tables:end -->

## Caching and speed
- **Cached once:** the matrix, the model and calibrator, the SHAP explainer, the similar-student pool and any precomputed explanations. Scores are cached per presentation; an explanation is cached per student.
- **Precomputed explanations:** SHAP takes about 0.25 s per student on one core, so a whole presentation is too slow to explain while the app runs. `scripts/precompute_shap.py` computes a presentation's SHAP values once (in parallel) into `models/final/shap/`. Student Lookup reads them when they exist and were computed for the same model file (checked by hash); otherwise it computes the one student's explanation and caches it.

## Tests
`tests/test_app.py` runs the app headless with `streamlit.testing.v1.AppTest` on synthetic data:
- with nothing available
- with a synthetic matrix and final model: capacity ranking by default, a student lookup with a risk band, the wide-net rule and the 3D view
- with precomputed explanations
- with a training presentation
- with team tables built from synthetic data, a student-level table (refused) and a table with missing columns

## Limits
- The team tables describe past presentations. The department and profile at-risk rates and the funnel's last two stages come from final outcomes, which are known only after a presentation ends.
- The flagged-list download writes student IDs to the user's machine, and `models/final/shap/` holds per-student values. Both are for staff follow-up on this machine only and should be handled under the project's data rules; neither is committed.
- Validation figures come from one presentation (2014J). Their uncertainty and the spread across presentations are in [final_model.md](final_model.md).
