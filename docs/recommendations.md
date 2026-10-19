# Suggested support (SCRUM-188, ticket 216)

For a flagged student, `sar.recommend` suggests up to three support types. Each comes with:
- the linked risk drivers, in plain language
- context from the most similar past students, with the sample size

## How to run
The model must be trained first (see `docs/explanations.md`). Then:
```python
from sar.data.split import make_split
from sar.recommend.neighbours import NeighbourPool
from sar.recommend.recommender import recommend

train, val = make_split(matrix)
pool = NeighbourPool(train)                     # build once
expl = explain_student(model, row[features], threshold)
print(recommend(expl, row, pool).text())
```
Worked example: `notebooks/216_SGS_intervention_recommender.ipynb`. It shows aggregates and a synthetic student only.

## How a suggestion is chosen
1. Take the inputs that raise the student's risk score by at least 0.005: activity, assessment and study-load inputs only.
2. Each input points to one or two support types with weights that sum to 1 (`sar/recommend/interventions.py`). For example, a low average score points to tutoring (0.6) and peer tutoring (0.4), and a long inactive streak points to weekly check-ins (0.6) and mentoring (0.4).
3. A support type's match score is the sum of weight × contribution. The top three are suggested. Each names up to two drivers, preferring drivers that a higher-ranked suggestion has not already named.

Background factors (gender, region, highest education, deprivation band, age band, disability) and course-level inputs (module, assessments due) are never mapped; this is checked when the module is imported. Financial aid referral is matched from study load and registration timing, never from the deprivation band.

The mapping is an assumption for staff to review and edit. It is not learned from data, because OULAD records no interventions.

## Similar students
- **Pool:** the training presentations only (2013B, 2013J, 2014B). The pool refuses 2014J, so a validation student's outcome can never reach a suggestion.
- **Distance:** activity, assessment and study-load inputs only. Click counts are log-scaled; missing scores are filled with the pool median and flagged. The imputer and scaler are fitted on the pool.
- **Which neighbours:** the 50 nearest from the same module. If a module has fewer than 50 earlier students, all modules are used and the text says so. In OULAD the smallest module has 339, so this is a safeguard.
- **What is shown:** how many neighbours ended At Risk or On Track, and the median of each suggestion's main driver in both groups. A group with fewer than 10 students is not shown.
- **Sanity check:** on a sample of 300 2014J students, a median 52% of flagged students' neighbours ended At Risk, against 22% for students not flagged.

## Assumptions and limits
- **Associations only.** Neighbour figures describe what similar past students looked like. They do not say what any support would change.
- **Mapping effects.** Weekly check-ins and peer tutoring come up often, because many activity inputs map to them. Financial aid referral is rare.
- **Indirect background influence.** Background factors never drive a suggestion directly, but the model uses them, and in tree models an input can change how much the other inputs contribute.
- **Hidden medians.** About 15% of suggestions show no medians, because one outcome group is too small.
- **Provisional model.** Built on the provisional model until ticket 214. Each suggestion takes about 0.25 s per student, including the SHAP explanation.
