"""Optimised Streamlit Community Cloud app for the Delhi PM2.5 dissertation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app_helpers import load_json, load_lite_model, predict_next_day, validate_project_files

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
ASSET_DIR = BASE_DIR / "assets"

st.set_page_config(
    page_title="Delhi PM2.5 Analysis and Prediction",
    page_icon="🌫️",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_metadata() -> dict:
    return load_json(DATA_DIR / "project_metadata.json")


@st.cache_resource(show_spinner=False)
def get_lite_model() -> dict:
    return load_lite_model(MODEL_DIR / "xgboost_pm25_lite.json")


@st.cache_data(show_spinner=False)
def read_csv(filename: str, parse_dates: tuple[str, ...] = ()) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / filename, parse_dates=list(parse_dates) or None)


def show_image(filename: str, caption: str) -> None:
    path = ASSET_DIR / filename
    if not path.exists():
        st.warning(f"Figure not found: {filename}")
        return
    st.image(str(path), caption=caption, use_container_width=True)


def render_overview() -> None:
    metadata = get_metadata()
    st.title("Delhi PM2.5 Analysis and Prediction")
    st.write(
        "This academic application presents the dissertation's verified "
        "spatio-temporal analysis and next-day Delhi-wide PM2.5 model."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Study period", f"{metadata['calendar_start']} to {metadata['calendar_end']}")
    col2.metric("Calendar days", f"{metadata['calendar_days']:,}")
    col3.metric("Modelling rows", f"{metadata['modelling_rows']:,}")
    col4.metric("Missing Delhi-wide days", metadata["missing_daily_pm25_days"])

    st.subheader("Research focus")
    st.markdown(
        """
        1. How PM2.5 varies across years, months and seasons.
        2. How observed PM2.5 differs between Delhi monitoring stations.
        3. How well next-day Delhi-wide PM2.5 can be predicted using recent PM2.5 history.
        """
    )

    st.subheader("Final modelling design")
    st.write(
        "Persistence, Linear Regression, Random Forest and XGBoost were compared. "
        "A focused SHAP analysis was applied to XGBoost."
    )
    st.info(
        "Academic prototype only. The model was trained on historical Delhi data from "
        "2015–2020 and does not use live monitoring or meteorological information."
    )


def render_analysis() -> None:
    st.title("Spatio-temporal analysis")
    st.write("Choose one verified notebook output to display. Loading one figure at a time keeps the app responsive.")

    view = st.selectbox(
        "Analysis view",
        [
            "Yearly pattern",
            "Monthly pattern",
            "Seasonal pattern",
            "Station comparison",
            "Pollutant correlation",
            "Daily PM2.5 series",
        ],
    )

    if view == "Yearly pattern":
        show_image("yearly_pm25.png", "Yearly average Delhi-wide PM2.5")
    elif view == "Monthly pattern":
        show_image("monthly_pm25.png", "Monthly average Delhi-wide PM2.5")
    elif view == "Seasonal pattern":
        show_image("seasonal_pm25.png", "Seasonal average Delhi-wide PM2.5")
    elif view == "Station comparison":
        show_image("station_pm25_top10.png", "Stations with the highest observed mean PM2.5")
        stations = read_csv("station_means.csv").head(10).copy()
        stations["ObservedMeanPM25"] = stations["ObservedMeanPM25"].round(2)
        st.dataframe(stations, hide_index=True, use_container_width=True)
        st.caption("Station comparisons are descriptive because stations have different operating periods.")
    elif view == "Pollutant correlation":
        show_image("pollutant_correlation.png", "Correlation between observed pollutants")
        st.caption("Correlation describes association and does not establish causation.")
    else:
        show_image("daily_pm25.png", "Delhi-wide daily PM2.5")
        st.caption("The daily value is the mean across stations reporting on each date.")


def render_prediction() -> None:
    metadata = get_metadata()
    defaults = read_csv("default_prediction_history.csv", parse_dates=("Date",)).sort_values("Date")

    st.title("Next-day PM2.5 prediction")
    st.write(
        "Enter seven consecutive Delhi-wide PM2.5 values from seven days ago to yesterday. "
        "The app creates the same five features used by the dissertation model."
    )

    with st.form("prediction_form"):
        prediction_date = st.date_input(
            "Prediction date",
            value=pd.Timestamp(metadata["default_prediction_date"]).date(),
        )
        labels = ["7 days ago", "6 days ago", "5 days ago", "4 days ago", "3 days ago", "2 days ago", "Yesterday"]
        default_values = defaults["PM2.5"].astype(float).tolist()
        previous_seven_days = []
        columns = st.columns(4)
        for index, (label, default_value) in enumerate(zip(labels, default_values)):
            with columns[index % 4]:
                previous_seven_days.append(
                    st.number_input(
                        label,
                        min_value=0.0,
                        value=float(round(default_value, 2)),
                        step=1.0,
                        format="%.2f",
                        key=f"pm25_day_{index}",
                    )
                )
        submitted = st.form_submit_button("Predict next-day PM2.5")

    if not submitted:
        return

    try:
        prediction, features = predict_next_day(
            model=get_lite_model(),
            prediction_date=prediction_date,
            previous_seven_days=previous_seven_days,
        )
    except (FileNotFoundError, ValueError, TypeError) as error:
        st.error(str(error))
        return

    st.metric("Predicted Delhi-wide PM2.5", f"{prediction:.2f} µg/m³")
    feature_table = pd.DataFrame([features]).rename(columns={
        "DayOfWeek": "Day of week",
        "PM2.5_Lag1": "PM2.5 lag 1",
        "PM2.5_Lag7": "PM2.5 lag 7",
        "PM2.5_Rolling7": "Previous 7-day mean",
    })
    st.dataframe(feature_table.round(2), hide_index=True, use_container_width=True)
    st.warning("This is a dissertation research prediction, not a live regulatory or medical forecast.")


def render_model_comparison() -> None:
    st.title("Model comparison")
    results = read_csv("model_results.csv").copy()
    results[["MAE", "RMSE", "R2"]] = results[["MAE", "RMSE", "R2"]].round(3)
    st.dataframe(results, hide_index=True, use_container_width=True)
    show_image("model_error_comparison.png", "Test-set prediction errors")
    show_image("xgboost_observed_vs_predicted.png", "Observed and XGBoost-predicted Delhi-wide PM2.5")
    st.markdown(
        "**Interpretation:** Persistence had the lowest MAE, while XGBoost had the lowest RMSE and highest R². "
        "The simple baseline therefore remained competitive."
    )


def render_shap() -> None:
    st.title("Focused SHAP interpretation")
    show_image("xgboost_shap_summary.png", "SHAP summary for the XGBoost test predictions")
    shap_table = read_csv("xgboost_shap_importance.csv").copy()
    shap_table["MeanAbsoluteSHAP"] = shap_table["MeanAbsoluteSHAP"].round(3)
    st.dataframe(shap_table, hide_index=True, use_container_width=True)
    st.write(
        "Lag 1 was by far the most influential feature, followed by the previous seven-day mean. "
        "SHAP explains model behaviour and does not prove causation."
    )


def main() -> None:
    try:
        validate_project_files(BASE_DIR)
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    page = st.sidebar.radio(
        "Navigation",
        ["Project overview", "Spatio-temporal analysis", "Next-day prediction", "Model comparison", "SHAP interpretation"],
    )
    st.sidebar.caption("MA981 MSc Dissertation · University of Essex")

    if page == "Project overview":
        render_overview()
    elif page == "Spatio-temporal analysis":
        render_analysis()
    elif page == "Next-day prediction":
        render_prediction()
    elif page == "Model comparison":
        render_model_comparison()
    else:
        render_shap()


if __name__ == "__main__":
    main()
