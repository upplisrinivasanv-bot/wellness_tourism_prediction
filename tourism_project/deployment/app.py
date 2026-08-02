
import os
import joblib
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Wellness Tourism Purchase Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧳 Wellness Tourism Package Prediction")
st.markdown("Predict if a customer will purchase the newly introduced **Wellness Tourism Package** based on demographics and past behaviors.")

# 2. Path to Trained Model
# Assumes app.py is in root and the model is at 'tourism_project/deployment/model.pkl'
MODEL_PATH = os.path.join("tourism_project", "deployment", "best_model.pkl") # Changed to best_model.pkl

@st.cache_resource
def load_prediction_model(path):
    if not os.path.exists(path):
        # Fallback search if the pipeline places it with a different serialized extension
        for ext in [".pkl", ".joblib", ".pkl.gz"]:
            alt_path = path.replace(".pkl", ext)
            if os.path.exists(alt_path):
                return joblib.load(alt_path)
        raise FileNotFoundError(f"Model file not found at {path}. Please check your deployment path.")
    return joblib.load(path)

try:
    model = load_prediction_model(MODEL_PATH)
    st.sidebar.success("✅ Model loaded successfully from deployment pipeline!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. User Input Layout and Collection
st.subheader("📋 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    type_of_contact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3], help="1: Highest Tier, 3: Lowest Tier")
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    num_person = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2, step=1)
    preferred_stars = st.slider("Preferred Property Star Rating", min_value=3, max_value=5, value=3, step=1)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    num_trips = st.number_input("Average Number of Trips/Year", min_value=1, max_value=20, value=3, step=1)

with col3:
    passport = st.selectbox("Has Passport?", ["No", "Yes"])
    own_car = st.selectbox("Owns a Car?", ["No", "Yes"])
    num_children = st.number_input("Number of Children Visiting (< 5 years old)", min_value=0, max_value=5, value=0, step=1)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Gross Monthly Income", min_value=0, value=40000, step=1000)

# 4. Map Inputs to Dataframe Format (Matching Data Dictionary Structure)
# Converts UI values to expected model formats (e.g., binary labels to 0 or 1)
input_data = {
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_person,
    "PreferredPropertyStar": preferred_stars,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}

# Convert dictionary into a pandas dataframe
features_df = pd.DataFrame([input_data])

# 5. Prediction Engine
st.markdown("---")
if st.button("🔮 Predict Purchase Probability", type="primary"):
    try:
        # Check if the model has a prediction probability function
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(features_df)[0][1]
            prediction = model.predict(features_df)[0]
        else:
            prediction = model.predict(features_df)[0]
            prob = None

        # Display Results
        st.subheader("🎯 Prediction Result")

        if prediction == 1:
            st.success("**Target Customer:** This individual is highly likely to purchase the Wellness Tourism Package!")
            if prob is not None:
                st.metric(label="Purchase Probability", value=f"{prob * 100:.2f}%")
        else:
            st.warning("**Low Potential Customer:** This individual is unlikely to purchase the Wellness Tourism Package.")
            if prob is not None:
                st.metric(label="Purchase Probability", value=f"{prob * 100:.2f}%")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Ensure your preprocessing pipeline (e.g., ColumnTransformer/OneHotEncoder) is embedded directly within the saved `model.pkl` pipeline object.")
