# Suggested support (SCRUM-188, ticket 216)

For a flagged student, `sar.recommend` suggests up to three support types. Each comes with:
- the linked risk drivers, in plain language
- context from the most similar past students, with the sample size

The recommender is **rule-based**. It matches the model's risk drivers to support types through a fixed mapping that staff can review and edit. Nothing is learned from outcomes: **OULAD contains no record of interventions**, so the data cannot show whether any support works.

## How to run
The model must be trained first (see `docs/explanations.md`). Then:
```python
from sar.data.split import make_split
from sar.recommend.neighbours import NeighbourPool
from sar.recommend.recommender import recommend

train, val = make_split(matrix)
pool = NeighbourPool(train)                     # build once
expl = explain_student(fitted["model"], row[fitted["features"]], band=band)
print(recommend(expl, row, pool).text())
```
Worked example: `notebooks/216_SGS_intervention_recommender.ipynb`. It shows aggregates and a synthetic student only, and was produced with the earlier 30-input model; the rules are unchanged apart from the actionable-input list below.

## Actionable inputs
Only inputs that a support contact can respond to drive a suggestion (`ACTIONABLE` in `sar/recommend/interventions.py`):

| Category | Inputs |
|---|---|
| Engagement | site clicks, active days, clicks per active day, clicks in the last two weeks and their trend, first active day, inactivity streaks |
| Submissions | assessments submitted and missed, average and weighted score, score trend |
| Timing | late submissions, average submission delay, registration timing |
| Workload | credits studied |

Never used: protected attributes (gender, region, highest education, deprivation band, age band, disability), course-level inputs (module, assessments due) and fixed history (previous attempts). This is checked when the module is imported.

## How a suggestion is chosen
1. Take the actionable inputs that raise the student's risk score by at least 0.005.
2. Each input points to one or two support types with weights that sum to 1 (`sar/recommend/interventions.py`). For example, a low average score points to tutoring (0.6) and peer tutoring (0.4), and a long inactive streak points to weekly check-ins (0.6) and mentoring (0.4).
3. A support type's match score is the sum of weight × contribution. The top three are suggested. Each names up to two drivers, preferring drivers that a higher-ranked suggestion has not already named.
4. For a student in the Low risk band the suggestions are marked optional.

Financial aid referral is matched from workload and registration timing, never from the deprivation band.

## Similar students
- **Pool:** the training presentations only (2013B, 2013J, 2014B). The pool refuses 2014J, so a validation student's outcome can never reach a suggestion.
- **Distance:** activity, assessment and study-load inputs only. Click counts are log-scaled; missing scores are filled with the pool median and flagged. The imputer and scaler are fitted on the pool.
- **Which neighbours:** the 50 nearest from the same module. If a module has fewer than 50 earlier students, all modules are used and the text says so.
- **What is shown:** how many neighbours ended At Risk or On Track, and the median of each suggestion's main driver in both groups. A group with fewer than 10 students is not shown.

## Assumptions and limits
- **Descriptive context only.** Neighbour figures describe what similar past students looked like. They are not evidence of effectiveness and do not say what any support would change.
- **Mapping effects.** Weekly check-ins and peer tutoring come up often, because many engagement inputs map to them. Financial aid referral is rare.
- **Indirect association.** Protected attributes are not model inputs, but other inputs can correlate with them.
- **Speed.** Each suggestion needs the student's SHAP explanation; the app uses precomputed values where available (see `docs/app.md`).
