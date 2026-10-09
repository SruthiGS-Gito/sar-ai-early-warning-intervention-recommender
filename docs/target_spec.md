# Target and Day 30 population (SCRUM-172)

| Item | Decision |
|---|---|
| Unit of prediction | one module registration = (code_module, code_presentation, id_student) |
| Target `at_risk` | 1 = Fail or Withdrawn, 0 = Pass or Distinction |
| Prediction day | Day 30 (`CUTOFF_DAY` in `src/sar/config.py`) |
| Model population | students NOT unregistered on or before Day 30 |
| Early leavers | kept aside (`early_leavers()`) for the retention funnel, not modelled |
| Never features | `final_result`, `date_unregistration` |

Why: a student who left before Day 30 cannot be "warned early", and
`date_unregistration` would reveal the answer.

Verified counts:
- registrations total: 32,593
- early leavers (unregistered <= Day 30): 5,127
- model population: 27,466
- at-risk share in model population: 44.0 %
