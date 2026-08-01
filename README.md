# Delhi PM2.5 Streamlit app — fast Community Cloud version

This is the optimised deployment package for the MA981 dissertation project.

## Why this version starts faster

The earlier app installed and imported `xgboost` and `scikit-learn`, loaded the model before the user selected a page, loaded every CSV at startup, and rendered all analysis tabs together.

This version:

- uses a compact pure-Python copy of the fitted XGBoost trees;
- requires only Streamlit at deployment time;
- loads the model only when the prediction page is used;
- loads each small table only on the page that needs it;
- displays one analysis figure at a time;
- uses a static daily-series figure instead of rebuilding a large interactive chart;
- caches metadata, tables and the compact model.

The compact predictor was checked against all 399 saved XGBoost test predictions. The maximum absolute difference was below `0.001 µg/m³`.


## Important: upload the folder structure

The GitHub repository root must contain the `model`, `data`, `assets` and `.streamlit` folders. Do not upload only `streamlit_app.py`.

Correct repository root:

```text
streamlit_app.py
app_helpers.py
requirements.txt
model/
data/
assets/
.streamlit/
```

This ZIP is GitHub-ready: its files and folders are stored directly at the archive root. Extract it, then upload every extracted item to the repository root.

## Deploy

1. Upload the contents of this folder to the root of a GitHub repository.
2. Open Streamlit Community Cloud.
3. Select the repository and the `main` branch.
4. Set the entry point to `streamlit_app.py`.
5. Select Python 3.12.
6. Deploy.

The first Community Cloud build can still take a few minutes because Streamlit creates a new Python environment. Later app reruns should be much faster.

## Local run

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Validate the package

```bash
python validate_deployment.py
python -m pytest -q
```

## Important limitation

This is an academic prototype trained on historical Delhi data from 2015–2020. It is not a live regulatory, medical or public-health forecasting service.
