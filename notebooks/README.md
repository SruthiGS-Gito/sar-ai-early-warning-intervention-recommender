# Notebooks

Naming: `<ticket>_<initials>_<topic>.ipynb`, for example `106_AM_grade_distribution.ipynb`.

Rules:
- Import project code from `sar` instead of copying functions.
- Clear heavy outputs before committing.
- Save finished charts to `reports/figures/`.
- Read data from `sar.config.RAW_DIR` or `PROCESSED_DIR`. Do not use absolute paths from your own machine.



Naming:  `xgboost_randomforest (2).ipynb`.

Overview
- The xgboost_randomforest.ipynb notebook implements model training,
- Hyperparameter tuning, and model comparison for the student at-risk
- Prediction task. It compares Random Forest and XGBoost with the existing
- Logistic Regression baseline using the prepared Day 30 feature matrix.
