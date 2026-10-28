# Streamlit app (SCRUM-189, ticket 302)

## How to run
```
python scripts/run_baseline.py       # Day 30 feature matrix (once)
python scripts/train_final.py        # final model into models/final/ (once, about a minute)
python scripts/precompute_shap.py    # optional: explanations for the validation presentation
streamlit run app/streamlit_app.py
```
- The app opens at http://localhost:8501.
- `.streamlit/config.toml` serves it on this machine only, because the app shows individual student records. Do not change the address to share it. It also sets a dark theme (deep navy background, indigo accent) and a minimal toolbar, which hides developer items such as the Deploy button.
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
The flagging rule is chosen in the sidebar, under Advanced; capacity is the default.
- **Capacity ranking (default):** rank the cohort by calibrated risk and flag the top share chosen on the slider (1% to 50%). This matches how a support team with a fixed number of contacts works. Students are ordered by calibrated score; because isotonic calibration gives many tied scores, ties are broken by the raw forest score and then by row order, which reproduces the capacity table in `metrics.json`. With "Capacity applies to: each module separately", the same share is flagged inside every module, with the same tie-breaking. Precision, recall and the numbers of at-risk students caught and missed are computed from the cohort's own outcomes for the chosen share, and are labelled in-sample for a training presentation. Capacity flags can include Medium-band students: High is a fixed cutoff set when the model was built, so a share larger than the High band's share reaches into Medium.
- **Wide net (fixed cutoff):** flag everyone at or above the `wide_net` cutoff. Because at-risk rates differ between intakes, a cutoff fixed in advance does not flag a stable share of a new cohort; see the limitations in [final_model.md](final_model.md#limitations).

## Layout
The app is organised around a tutor's workflow: see who to contact first, understand one student, then look at the cohort. It uses a dark, card-based layout: the CSS in the app styles bordered containers and metrics as cards, risk bands as coloured pills that always carry their name (High #ff5d73, Medium #ffb454, Low #35d0ba), and every chart uses one Plotly template with transparent backgrounds and light text. Each page opens with its title, a one-line subtitle and a pill naming the presentation (validation, or training and in-sample).

**Sidebar**: the app name and purpose; the page navigation (a radio styled as a full-width list: Overview, Student, Modules, Support, About; only the selected page runs, and the chosen student and page selections are kept when moving between pages); then the controls, each only where it applies. Presentation (2014J, the validation presentation, by default) shows on every page. Module (All modules by default) and Contact capacity (1% to 50%, default 20%) show on Overview and Support; elsewhere the pages use all modules and the settings keep their last values. A collapsed Advanced section, on Overview only, holds "Show training presentations (in-sample scores)" (its in-sample note appears only when it is turned on), the flagging rule (capacity or wide net) and the capacity scope (the whole cohort or each module separately). Explanations are hover help on the controls. Only when a training presentation is chosen does the sidebar show the in-sample warning.

| Page | Shows |
|---|---|
| Overview (home) | Four KPI cards of equal height with the same slots (icon and label, value, one caption line, and a bar area that stays empty when the value is not a share): Students in view; Flagged for contact, with a bar for the flagged share; Precision and At-risk reached (recall), each with a bar. For a training presentation Precision and At-risk reached read "Not shown" (in-sample scores are optimistic and not informative). One caption names the rule and cohort, for example "Highest-risk 20% of 2014J, all modules." Then "Risk distribution", a donut of the Low, Medium and High bands among the students in view with the total in the centre, and "Main reasons behind flags", the flagged students by kind of main reason as a share of the flagged students (kinds under 5% are counted together as "Other"); it is drawn only when reasons are prepared for every flagged student, and otherwise says "Available for the validation presentation (2014J)." Then "Students requiring review": a student ID search and a band filter above the ranked list of flagged students (rank, student ID, module, risk band as a coloured pill, a Day 30 snapshot such as "8 active days, 1 missed, 0 submitted", main reason shortened to one line, and suggested support). Selecting a row opens that student on the Student page. The full list downloads as CSV with a score-source column. A collapsed "Flags by module and band" section holds the flagged students per band and the by-module table. |
| Student | Example buttons (a 2014J student in the High or Low band, drawn with a fixed seed from the chosen module when it has one) and the selectors "Module of the student" and "Student ID"; until a student is chosen, one instruction line. For a chosen student, a header card: a person icon, the student ID, module and presentation, the band pill and the rank within the module. For the validation presentation it adds a ring, "Observed outcome for this band": the share of 2014J students in the student's band and module who went on to fail or withdraw, with n (for example 71% of the 42 High-band students in AAA). It is a group statistic, not a personal probability; where the band has fewer than 20 students in the module, the band's rate across all modules is shown and the caption says so. A training presentation shows the band and the in-sample note only. The card ends with "This is a model prediction. It supports, and does not replace, professional judgment." Four tabs follow. Overview: the verdict, which names only factors the chart shows (or says that no input lowers the score noticeably), and the primary suggestion. Day 30 indicators: six cards of equal height (total clicks, active days, days inactive just before Day 30, assessments submitted, assessments missed, average score), each with its value ("Not available yet" when missing) and one line comparing it with the module median. Why this score: up to four raising and two lowering activity, assessment and study-load inputs, in the raw forest score (probability points, before calibration), the calibrated score once ("calibrated across all modules"), and an expander with the sentence and every input to three decimals, grouped as Behaviour, Study and Course. Support: the primary suggestion tagged Primary; other matched types tagged Also consider; and an "About these suggestions" expander with the definition of the similar-student group (the 50 students of the same module in the training presentations closest on Day 30 activity, assessment and study-load inputs, matched on inputs and not on the score) and the note that suggestions are rule-based and descriptive. A Low-band student gets a primary suggestion only when it comes from a missed assessment or a spell without activity; otherwise the page says "No specific support suggested at Day 30." and the rule-based matches are in an expander. |
| Modules | An "All modules" card, then one card per module, four per row and of equal height: the observed at-risk rate, "n = ... students" and a thin stacked bar of the Low, Medium and High shares (groups under 20 students show n < 20). Then two rows of charts: the stacked band shares by module and the observed at-risk rate by intake (2013B to 2014J, from final outcomes, with n; validation in violet and training in blue); the observed at-risk rate by days active by Day 30 (0, 1-5, 6-10, 11-15, 16-20, 21+, with n; active days include days before the module starts) and the student profiles with "What each profile means". The retention funnel has a full-width row: short stage labels (the full wording is in the data table), values inside the bars, its module selector and one caption (retrospective; students carried forward where no assessment was due by Day 30). Collapsed sections: "Data tables", "Model reliability" (the calibration chart for 2014J) and "Optional 3D view". OULAD has modules, not departments, so each department row is a module. |
| Support | "Suggested support across flagged students": one card per support type, ordered by count, with its definition from the recommender and the number of flagged students whose primary suggestion is that type. A type with no students explains, from the rule mapping, which inputs it is matched from and which types share them. When one type is the primary suggestion for more than half of the flagged students, one sentence says so (on 2014J at 20% capacity, 1,211 of 1,841, 66%, get weekly check-ins first). A bar chart of the counts and a stacked bar by module use violets and blues only, so no colour reads as a risk band. A boxed note says that suggestions are rule-based and that OULAD has no intervention records, so no outcome or effectiveness figures are shown. Where suggestions are not prepared for every flagged student, the page says "Available for the validation presentation (2014J)." |
| About | Each section in its own card: the model; reading the Student page (the observed-outcome ring, the score scales, similar students and the Low-band rule); design and validation (Day 30 cutoff, temporal split, removal of students in both sets, cutoffs from out-of-fold training scores, 2014J as a validation rather than an untouched test set); the risk bands, their cutoffs in plain language and how to read them (observed at-risk rate and lift on 2014J); how to read the Modules page; how capacity flagging works; limitations (including the two lowest module recalls at 20% capacity and the group recall gaps from `metrics.json`); the OULAD licence and citation; and a Data details expander with the model build date, whether each table is loaded and the rebuild commands. |

Main reason and suggested support for the whole list come from `sar.service.priority`, which applies the explanation and recommender rules to a table of SHAP contributions. With precomputed explanations (`scripts/precompute_shap.py`, done for 2014J) every row is filled; without them, the top 50 students of the list are explained when the list is shown, and the rest say "Open the student to see".

Display rules:
- No screen shows ticket numbers, file paths or required-column lists. Build commands and the model build date appear only under About, Data details.
- Aggregated by default. Student IDs and individual rows appear only in the Overview list and the Student page. Student IDs are the pseudonymised identifiers published with OULAD.
- A student's risk is shown as a band and a rank, not as a number.
- The validation presentation (2014J) is shown by default. A training presentation is offered only after the user turns on "Show training presentations (in-sample scores)" under Advanced; it is then labelled in the presentation list, in the sidebar warning, on the cards and in every exported table.
- Explanations that used to sit under the charts are in the section headings' hover help and in About, "Reading the Modules page"; the in-sample warning and the funnel's retrospective and carry-forward note stay visible.
- Protected attributes are not model inputs and never appear in explanations or suggestions.

## Team tables
Three aggregated tables, built by one command after the feature matrix and the final model:
```
python scripts/build_team_tables.py
```
The code is in `sar.tables`. The script writes the tables to `data/processed/` and the summary used below to `reports/team_tables.json`, and regenerates the figures in this section from that summary.

| File | Ticket | One row per | Columns |
|---|---|---|---|
| `department_risk.csv` | 203 (SCRUM-178) | module and presentation, plus an "All modules" row per presentation | `code_module`, `code_presentation`, `score_source`, `students`, `at_risk_rate`, `share_low`, `share_medium`, `share_high`, `rate_low`, `rate_medium`, `rate_high`, `suppressed` |
| `retention_funnel.csv` | 207 (SCRUM-177) | stage, module and presentation, plus "All modules" rows | `code_module`, `code_presentation`, `stage_order`, `stage_key`, `stage`, `students`, `share_of_registered`, `share_of_previous`, `applicable`, `suppressed` |
| `student_profiles.csv` | 210 (SCRUM-183) | profile and scope (validation or training) | `scope`, `presentations`, `profile`, `cluster`, `description`, `students`, `share_of_scope`, `at_risk_rate`, `mean_<feature>` for each profile feature, `suppressed` |

### Definitions
**Department risk.** A department is one module in one presentation. `students` is the number of students in the Day 30 population (not unregistered by Day 30). `at_risk_rate` is the observed share whose final outcome was Fail or Withdrawn. The band shares are the shares of students in each risk band (Low, Medium, High; `sar.config.RISK_BANDS`) of the final model's calibrated score. `rate_low`, `rate_medium` and `rate_high` are the observed at-risk rates inside each band; the app divides each by the overall at-risk rate of the same row (the module's own rate for a module, the presentation's rate for All modules) to show the band's lift. A band rate is blank when the band holds fewer than 20 students in that group. For training presentations (`score_source` = training) the scores are in-sample: the model learned from those students' outcomes, so their band shares are better separated than they would be for a new intake.

**Retention funnel.** A retrospective, descriptive view. The stages are nested: each counts the students of the previous stage who also meet its condition.
1. Registered: every enrolment in `studentInfo`.
2. Still registered after Day 30: not unregistered on or before Day 30.
3. Active in the VLE by Day 30: at least one day of VLE activity by Day 30.
4. Met the Day 30 submission check (submitted, or none was due): at least one non-exam assessment submitted by Day 30, or no assessment was due by Day 30.
5. Did not withdraw: final outcome is not Withdrawn.
6. Passed: final outcome is Pass or Distinction.

Stages 1 to 4 use data available on Day 30. Stages 5 and 6 use final outcomes; they describe what happened after Day 30 and are not model inputs. Where a module presentation has no assessment due by Day 30, stage 4 would measure the course schedule rather than the student, so it is marked not applicable (`applicable` = False) and the stage 3 count is carried forward into stages 4 to 6.

**Student profiles.** KMeans clusters of five Day 30 engagement and submission features: `total_clicks` (log-transformed before scaling), `active_days`, `trailing_inactive_days`, `n_submitted` and `n_missed`. None of the six protected attributes, no assessment score and no outcome is a clustering input. The scaler and the clusters are fitted on the training presentations (after removing students who also appear in the validation presentation); validation students are then assigned to the nearest cluster. The profile features were selected to give interpretable engagement groups, after a trial in which a lateness-based feature (`n_late`) produced a single-purpose cluster defined by late submission alone.
- **Number of clusters:** the k from 3 to 5 with the highest silhouette score on the training rows (a fixed sample of 10,000 rows, seed 42). The chosen solution was then checked for interpretability: each cluster has a distinct centroid pattern and a distinct at-risk rate.
- **Names:** written after inspecting the centroids and at-risk rates, and assigned by rules on the centroids (`sar.tables.profiles`): the cluster with the fewest active days is *Little VLE activity*; of the rest, the one with the most missed assessments is *Missed early assessment*; then the one with the most clicks is *Highly engaged*; the remaining cluster is *Moderately engaged*.
- **Descriptions:** one sentence per cluster, filled in from its mean values and observed at-risk rate.
- **At-risk rate:** from final outcomes, attached after clustering. It is not a clustering input.

Credit: the feature choice and the range of cluster counts draw on Gouri's engagement profile notebook (`notebooks/Engagement profile charts and export.ipynb`), which clusters Day 30 clicks, active days, inactivity streaks, scores and submissions with KMeans and compares cluster counts by silhouette and inertia. This implementation differs from it in four ways: it fits on the training presentations only, leaves out assessment scores, tests k from 3 to 5, and uses its own profile names, because names such as "Improving" describe a change over time that a single Day 30 snapshot cannot show (a caution the notebook also gives).

### Privacy and display rules
- Every row is a group of students. No file holds a student identifier, and the app refuses any team table with an `id_student` column.
- Small cells: a group with fewer than `MIN_CELL` (20) students, or a funnel count from 1 to 19, is kept with its figures blank and `suppressed` set. In the funnel, enough further modules are hidden in the same presentation and stage that the "All modules" total cannot reveal a hidden count. Differences between consecutive stages of one module are not suppressed.
- The tables are generated data and stay in `data/processed/`, which git ignores. `reports/team_tables.json` holds only the aggregate figures quoted below and is tracked, so the documentation can be checked against it.
- Licence of the source data, as stated on the OULAD page (https://research.stem.open.ac.uk/ouanalyse/open-dataset-more/): "This dataset is released under CC-BY 4.0 license." Reference required when citing the dataset: Kuzilek J., Hlosta M., Zdrahal Z. Open University Learning Analytics dataset Sci. Data 4:170171 doi: 10.1038/sdata.2017.171 (2017). See the README.

How the app handles each table:
- **Missing file:** a card saying the table is not available yet; the ticket, the expected file and the build command are under About, Data details.
- **Missing columns:** a card saying the file does not have the expected columns; About, Data details names them.
- **File with an `id_student` column:** refused.
- **None of these cases** crashes the app.

### Current figures
<!-- team-tables:start. Generated by scripts/build_team_tables.py from reports/team_tables.json; do not edit by hand. -->

Figures from the last run of `scripts/build_team_tables.py`. Groups and counts under 20 students are hidden. Source: OULAD, released under CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/); Kuzilek J., Hlosta M., Zdrahal Z. Open University Learning Analytics dataset Sci. Data 4:170171 doi: 10.1038/sdata.2017.171 (2017).

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
| Met the Day 30 submission check (submitted, or none was due) | 3,660 (78%) | 6,898 (78%) | 5,764 (74%) | 8,466 (75%) |
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
- **Precomputed explanations:** SHAP takes about 0.25 s per student on one core, so a whole presentation is too slow to explain while the app runs. `scripts/precompute_shap.py` computes a presentation's SHAP values once (in parallel) into `models/final/shap/`. The priority list and the student view read them when they exist and were computed for the same model file (checked by hash); otherwise it computes the one student's explanation and caches it.

## Tests
`tests/test_app.py` runs the app headless with `streamlit.testing.v1.AppTest` on synthetic data:
- with nothing available
- with a synthetic matrix and final model: the priority list on first load (cards, ranked list, reasons and suggestions checked against the recommender), the student view (header, factors, support cards, snapshot), the module filter, the wide-net rule and the 3D view
- with precomputed explanations
- with a training presentation
- with team tables built from synthetic data, a student-level table (refused) and a table with missing columns

## Limits
- The team tables describe past presentations. The department and profile at-risk rates and the funnel's last two stages come from final outcomes, which are known only after a presentation ends.
- The flagged-list download writes student IDs to the user's machine, and `models/final/shap/` holds per-student values. Both are for staff follow-up on this machine only and should be handled under the project's data rules; neither is committed.
- Validation figures come from one presentation (2014J). Their uncertainty and the spread across presentations are in [final_model.md](final_model.md).
