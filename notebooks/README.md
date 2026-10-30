# Notebooks

Naming: `<ticket>_<initials>_<topic>.ipynb`, for example `106_AM_grade_distribution.ipynb`.

Rules:
- Import project code from `sar` instead of copying functions.
- Clear heavy outputs before committing.
- Save finished charts to `reports/figures/`.
- Read data from `sar.config.RAW_DIR` or `PROCESSED_DIR`. Do not use absolute paths from your own machine.

## XGBoost and Random Forest notebooks

Both notebooks are by Midhula M S. Their file names do not follow the `<ticket>_<initials>_<topic>` convention; they are kept unchanged because the documentation cites them.

- `xgboost-random-forest (2).ipynb`: model training, hyperparameter tuning and model comparison for the at-risk prediction task. It compares Random Forest and XGBoost with the logistic regression baseline on the Day 30 feature matrix, and its tuned Random Forest settings are the ones used by the final model.
- `xgboost-random-forest (4).ipynb`: a later version of the same notebook, which adds a threshold check, the data-leakage checklist and an evaluation of the selected model.
