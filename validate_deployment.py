"""Offline validation for the fast Streamlit deployment package."""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pandas as pd

from app_helpers import (
    EXPECTED_FEATURES,
    load_lite_model,
    predict_lite_model,
    predict_next_day,
    validate_project_files,
)

BASE_DIR = Path(__file__).resolve().parent


class FakeBlock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def metric(self, *args, **kwargs):
        return None


class FakeSidebar:
    def __init__(self):
        self.page = "Project overview"

    def radio(self, label, options):
        return self.page

    def caption(self, *args, **kwargs):
        return None


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.sidebar = FakeSidebar()
        self.submit_prediction = True

    def set_page_config(self, *args, **kwargs):
        return None

    def cache_data(self, *args, **kwargs):
        return lambda function: function

    def cache_resource(self, *args, **kwargs):
        return lambda function: function

    def columns(self, count):
        return [FakeBlock() for _ in range(count)]

    def form(self, *args, **kwargs):
        return FakeBlock()

    def selectbox(self, label, options):
        return options[0]

    def date_input(self, label, value, **kwargs):
        return value

    def number_input(self, label, value=0.0, **kwargs):
        return value

    def form_submit_button(self, *args, **kwargs):
        return self.submit_prediction

    def stop(self):
        raise RuntimeError("st.stop called during validation")

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def load_app_with_fake_streamlit():
    fake_streamlit = FakeStreamlit()
    sys.modules["streamlit"] = fake_streamlit
    specification = importlib.util.spec_from_file_location(
        "fast_streamlit_app", BASE_DIR / "streamlit_app.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module, fake_streamlit


def main() -> None:
    validate_project_files(BASE_DIR)

    start = time.perf_counter()
    model = load_lite_model(BASE_DIR / "model" / "xgboost_pm25_lite.json")
    model_load_seconds = time.perf_counter() - start

    history = pd.read_csv(BASE_DIR / "data" / "default_prediction_history.csv")
    values = history["PM2.5"].astype(float).tolist()
    prediction, features = predict_next_day(model, date(2020, 7, 2), values)

    assert model["feature_names"] == EXPECTED_FEATURES
    assert prediction > 0
    assert features["PM2.5_Lag1"] == values[-1]
    assert features["PM2.5_Lag7"] == values[0]

    sample = pd.read_csv(BASE_DIR / "data" / "model_validation_sample.csv")
    differences = []
    for _, row in sample.iterrows():
        feature_values = {name: float(row[name]) for name in EXPECTED_FEATURES}
        lite_prediction = predict_lite_model(model, feature_values)
        differences.append(abs(lite_prediction - float(row["XGBoost"])))
    maximum_difference = max(differences)
    assert maximum_difference < 0.001

    app, fake_streamlit = load_app_with_fake_streamlit()
    for page in [
        "Project overview",
        "Spatio-temporal analysis",
        "Next-day prediction",
        "Model comparison",
        "SHAP interpretation",
    ]:
        fake_streamlit.sidebar.page = page
        app.main()

    app_text = (BASE_DIR / "streamlit_app.py").read_text(encoding="utf-8")
    helper_text = (BASE_DIR / "app_helpers.py").read_text(encoding="utf-8")
    assert "import xgboost" not in app_text + helper_text
    assert "sklearn" not in app_text + helper_text

    print("Fast deployment validation passed.")
    print(f"Lite model load time: {model_load_seconds:.6f} seconds")
    print(f"Default example prediction: {prediction:.6f} µg/m³")
    print(f"Maximum difference from saved XGBoost predictions: {maximum_difference:.12f}")
    print("All five application pages executed through the offline Streamlit harness.")
    print("No runtime XGBoost or scikit-learn dependency is used.")


if __name__ == "__main__":
    main()
