from datetime import date
from pathlib import Path
import json
import pandas as pd

from app_helpers import (
    EXPECTED_FEATURES,
    build_feature_values,
    load_lite_model,
    predict_lite_model,
    predict_next_day,
    validate_history,
)

BASE = Path(__file__).resolve().parent


def test_history_validation():
    assert validate_history([1, 2, 3, 4, 5, 6, 7]) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_feature_values():
    values = build_feature_values(date(2020, 7, 2), [10, 20, 30, 40, 50, 60, 70])
    assert values == {
        "Month": 7.0,
        "DayOfWeek": 3.0,
        "PM2.5_Lag1": 70.0,
        "PM2.5_Lag7": 10.0,
        "PM2.5_Rolling7": 40.0,
    }


def test_model_contract_and_prediction():
    model = load_lite_model(BASE / "model" / "xgboost_pm25_lite.json")
    assert model["feature_names"] == EXPECTED_FEATURES
    prediction, features = predict_next_day(
        model,
        date(2020, 7, 2),
        [40, 45, 50, 55, 60, 65, 70],
    )
    assert prediction > 0
    assert features["PM2.5_Rolling7"] == 55.0


def test_exact_saved_predictions():
    model = load_lite_model(BASE / "model" / "xgboost_pm25_lite.json")
    sample = pd.read_csv(BASE / "data" / "model_validation_sample.csv")

    differences = []
    for _, row in sample.iterrows():
        feature_values = {name: float(row[name]) for name in EXPECTED_FEATURES}
        actual = predict_lite_model(model, feature_values)
        differences.append(abs(actual - float(row["XGBoost"])))

    assert max(differences) < 0.001
