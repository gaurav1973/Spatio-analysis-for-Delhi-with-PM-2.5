# Fast Streamlit deployment validation

## Problem identified

The earlier deployment package installed and imported `xgboost` and `scikit-learn`, loaded the model before the user selected a page, loaded every table at startup, and rendered all analysis tabs together. These choices increase Community Cloud build time and cold-start work.

## Performance changes

- Removed runtime `xgboost` dependency.
- Removed runtime `scikit-learn` dependency.
- Reduced `requirements.txt` from five direct packages to one (`streamlit`).
- Replaced the 111 KB runtime XGBoost model with a 19 KB compact tree representation.
- Loads the model only when the prediction page is used.
- Loads each table only on the page that needs it.
- Displays one analysis figure at a time.
- Replaced the full interactive daily chart with a pre-generated figure.
- Disabled the development file watcher in the deployed configuration.

## Code validation

- `streamlit_app.py`, `app_helpers.py` and `validate_deployment.py` compiled successfully.
- Four unit tests passed.
- All required deployment files were found.
- All five application pages executed through an offline Streamlit interface harness.
- No runtime `xgboost` or `scikit-learn` import remains.

## Model validation

During development, the compact predictor was checked against all 399 saved XGBoost test predictions from the executed dissertation notebook.

- Maximum absolute difference across the full test set: `0.0001395 µg/m³`.
- Packaged validation sample maximum difference: below `0.001 µg/m³`.
- The difference is only floating-point precision from the exported tree values.

## Measured helper performance in this environment

- Compact model load: approximately `0.0004` seconds.
- One prediction: approximately `0.00003` seconds.

## Notebook validation

The revised Streamlit-fast notebook was executed from top to bottom:

- 20 of 20 code cells executed.
- No stored execution errors.
- Compact-model export assertion passed.
- Required deployment artefacts were found.

## Environment limitation

A live Streamlit server could not be installed in this container because external package installation is disabled. The package was instead syntax-checked, unit-tested, prediction-validated and exercised through an offline Streamlit API harness.
