"""Fast, testable helpers for the Delhi PM2.5 Streamlit app.

The deployed app uses a compact pure-Python representation of the fitted
XGBoost trees. This avoids importing the large XGBoost and scikit-learn
packages in Streamlit Community Cloud while preserving the fitted model's
predictions for the five validated input features.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

EXPECTED_FEATURES = [
    "Month",
    "DayOfWeek",
    "PM2.5_Lag1",
    "PM2.5_Lag7",
    "PM2.5_Rolling7",
]


def load_json(path: str | Path) -> dict | list:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_lite_model(path: str | Path) -> dict:
    model = load_json(path)
    if not isinstance(model, dict):
        raise ValueError("The lite model file must contain a JSON object.")
    if model.get("format") != "xgboost_tree_ensemble_lite_v1":
        raise ValueError("Unsupported lite model format.")
    validate_feature_contract(model.get("feature_names", []))
    if not isinstance(model.get("trees"), list) or not model["trees"]:
        raise ValueError("The lite model contains no trees.")
    return model


def validate_feature_contract(feature_order: Sequence[str]) -> None:
    received = list(feature_order)
    if received != EXPECTED_FEATURES:
        raise ValueError(
            "Feature order does not match the dissertation model. "
            f"Expected {EXPECTED_FEATURES}; received {received}."
        )


def validate_history(previous_seven_days: Iterable[float]) -> list[float]:
    values = [float(value) for value in previous_seven_days]
    if len(values) != 7:
        raise ValueError("Exactly seven previous daily PM2.5 values are required.")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("All PM2.5 inputs must be finite numbers.")
    if any(value < 0 for value in values):
        raise ValueError("PM2.5 inputs cannot be negative.")
    return values


def build_feature_values(
    prediction_date: date | datetime,
    previous_seven_days: Iterable[float],
) -> dict[str, float]:
    values = validate_history(previous_seven_days)
    return {
        "Month": float(prediction_date.month),
        "DayOfWeek": float(prediction_date.weekday()),
        "PM2.5_Lag1": values[-1],
        "PM2.5_Lag7": values[0],
        "PM2.5_Rolling7": sum(values) / 7.0,
    }


def _predict_tree(node: float | list, feature_vector: Sequence[float]) -> float:
    while isinstance(node, list):
        feature_index, threshold, left_child, right_child = node
        node = left_child if feature_vector[int(feature_index)] < float(threshold) else right_child
    return float(node)


def predict_lite_model(model: dict, feature_values: dict[str, float]) -> float:
    feature_names = list(model["feature_names"])
    validate_feature_contract(feature_names)
    feature_vector = [float(feature_values[name]) for name in feature_names]
    prediction = float(model["base_score"])
    prediction += sum(_predict_tree(tree, feature_vector) for tree in model["trees"])
    return prediction


def predict_next_day(
    model: dict,
    prediction_date: date | datetime,
    previous_seven_days: Iterable[float],
) -> tuple[float, dict[str, float]]:
    features = build_feature_values(prediction_date, previous_seven_days)
    prediction = predict_lite_model(model, features)
    return prediction, features


def required_files(base_directory: str | Path) -> list[Path]:
    base = Path(base_directory)
    return [
        base / "streamlit_app.py",
        base / "app_helpers.py",
        base / "requirements.txt",
        base / "model" / "xgboost_pm25_lite.json",
        base / "data" / "project_metadata.json",
        base / "data" / "model_results.csv",
        base / "data" / "default_prediction_history.csv",
        base / "assets" / "xgboost_shap_summary.png",
    ]


def validate_project_files(base_directory: str | Path) -> None:
    missing = [str(path) for path in required_files(base_directory) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing deployment files: " + ", ".join(missing))
