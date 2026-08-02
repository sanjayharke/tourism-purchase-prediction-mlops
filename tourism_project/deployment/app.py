"""Streamlit frontend for the Visit with Us purchase-prediction model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().with_name("model.joblib")


@st.cache_resource
def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.joblib is missing. Run the GitHub Actions training pipeline first."
        )
    bundle = joblib.load(MODEL_PATH)
    required_keys = {"model", "threshold", "feature_names", "test_metrics"}
    missing = required_keys - set(bundle)
    if missing:
        raise ValueError(f"Invalid model bundle; missing keys: {sorted(missing)}")
    return bundle


def build_input_form() -> pd.DataFrame:
    """Collect one customer's details and return a model-ready dataframe."""
    with st.form("customer_form"):
        st.subheader("Customer profile")
        left, middle, right = st.columns(3)

        with left:
            age = st.number_input("Age", min_value=18, max_value=100, value=36)
            contact = st.selectbox(
                "Type of contact", ["Self Enquiry", "Company Invited"]
            )
            city_tier = st.selectbox("City tier", [1, 2, 3])
            occupation = st.selectbox(
                "Occupation",
                ["Salaried", "Small Business", "Large Business", "Freelancer"],
            )
            gender = st.selectbox("Gender", ["Male", "Female"])
            designation = st.selectbox(
                "Designation",
                ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
            )

        with middle:
            persons = st.number_input(
                "Number of persons visiting", min_value=1, max_value=10, value=3
            )
            children = st.number_input(
                "Number of children visiting", min_value=0, max_value=6, value=1
            )
            trips = st.number_input(
                "Annual number of trips", min_value=0, max_value=30, value=3
            )
            passport = st.selectbox(
                "Has passport", options=[1, 0], format_func=lambda x: "Yes" if x else "No"
            )
            own_car = st.selectbox(
                "Owns a car", options=[1, 0], format_func=lambda x: "Yes" if x else "No"
            )
            marital_status = st.selectbox(
                "Marital status", ["Single", "Married", "Divorced"]
            )

        with right:
            duration = st.number_input(
                "Pitch duration (minutes)", min_value=1, max_value=180, value=15
            )
            followups = st.number_input(
                "Number of follow-ups", min_value=0, max_value=15, value=4
            )
            product = st.selectbox(
                "Product pitched",
                ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"],
            )
            property_star = st.selectbox("Preferred property stars", [3, 4, 5])
            satisfaction = st.selectbox("Pitch satisfaction score", [1, 2, 3, 4, 5])
            income = st.number_input(
                "Monthly income",
                min_value=1_000,
                max_value=500_000,
                value=23_000,
                step=500,
            )

        submitted = st.form_submit_button(
            "Predict purchase likelihood", type="primary", use_container_width=True
        )

    input_data = pd.DataFrame(
        [
            {
                "Age": float(age),
                "CityTier": int(city_tier),
                "DurationOfPitch": float(duration),
                "NumberOfPersonVisiting": int(persons),
                "NumberOfFollowups": float(followups),
                "PreferredPropertyStar": float(property_star),
                "NumberOfTrips": float(trips),
                "Passport": int(passport),
                "PitchSatisfactionScore": int(satisfaction),
                "OwnCar": int(own_car),
                "NumberOfChildrenVisiting": float(children),
                "MonthlyIncome": float(income),
                "TypeofContact": contact,
                "Occupation": occupation,
                "Gender": gender,
                "ProductPitched": product,
                "MaritalStatus": marital_status,
                "Designation": designation,
            }
        ]
    )
    input_data.attrs["submitted"] = submitted
    return input_data


def main() -> None:
    st.set_page_config(
        page_title="Wellness Tourism Purchase Predictor",
        page_icon="✈️",
        layout="wide",
    )
    st.title("Wellness Tourism Purchase Predictor")
    st.caption(
        "Visit with Us | Decision support for prioritizing prospective customers"
    )

    try:
        bundle = load_model_bundle()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    input_data = build_input_form()
    if input_data.attrs["submitted"]:
        ordered_input = input_data[bundle["feature_names"]]
        probability = float(bundle["model"].predict_proba(ordered_input)[0, 1])
        threshold = float(bundle["threshold"])
        predicted_buyer = probability >= threshold

        st.divider()
        metric_col, decision_col = st.columns([1, 2])
        with metric_col:
            st.metric("Purchase probability", f"{probability:.1%}")
            st.progress(min(max(probability, 0.0), 1.0))
        with decision_col:
            if predicted_buyer:
                st.success(
                    "High-priority prospect: include this customer in the targeted campaign."
                )
            else:
                st.info(
                    "Lower-priority prospect: consider a lower-cost nurture campaign."
                )
            st.write(
                f"The operating threshold is **{threshold:.2f}**. It was selected "
                "from out-of-fold training predictions by maximizing F2, which gives "
                "more weight to finding potential buyers than to avoiding an extra call."
            )

        with st.expander("View model-ready input"):
            st.dataframe(ordered_input, use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        "This model is a marketing decision-support tool. Monitor data drift and "
        "retrain when customer behavior or campaign conditions change."
    )


if __name__ == "__main__":
    main()

