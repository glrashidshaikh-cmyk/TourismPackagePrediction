import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load trained model
model_path = hf_hub_download(
    repo_id="RashidShaikhCFA/TourismPackagePrediction",
    filename="best_prediction_model_v1.joblib"
)
model = joblib.load(model_path)

st.set_page_config(page_title="Tourism Package Prediction", layout="wide")

st.title("🌍 Tourism Package Prediction")
st.write("Predict whether a customer is likely to purchase the tourism package.")

user_input = {}

col1, col2, col3 = st.columns(3)

with col1:
    user_input["Age"] = st.number_input("Age", min_value=18, max_value=100, value=35)
    user_input["NumberOfPersonVisiting"] = st.number_input("Number Of Person Visiting", min_value=1, max_value=10, value=2)
    user_input["NumberOfTrips"] = st.number_input("Number Of Trips", min_value=0, max_value=50, value=2)
    user_input["MonthlyIncome"] = st.number_input("Monthly Income", min_value=0, value=25000)

with col2:
    user_input["CityTier"] = st.number_input("City Tier", min_value=1, max_value=3, value=1)
    user_input["NumberOfFollowups"] = st.number_input("Number Of Followups", min_value=0, max_value=10, value=2)
    user_input["PitchSatisfactionScore"] = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    user_input["TypeofContact"] = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])

with col3:
    user_input["DurationOfPitch"] = st.number_input("Duration Of Pitch", min_value=0, max_value=100, value=20)
    user_input["PreferredPropertyStar"] = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
    user_input["NumberOfChildrenVisiting"] = st.number_input("Number Of Children Visiting", min_value=0, max_value=10, value=0)
    user_input["Occupation"] = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])

col4, col5, col6 = st.columns(3)

with col4:
    user_input["Gender"] = st.selectbox("Gender", ["Male", "Female"])
    user_input["Passport"] = st.selectbox("Passport", ["Yes", "No"])

with col5:
    user_input["ProductPitched"] = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    user_input["OwnCar"] = st.selectbox("Own Car", ["Yes", "No"])

with col6:
    user_input["MaritalStatus"] = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    user_input["Designation"] = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

input_df = pd.DataFrame([user_input])

input_df["Passport"] = input_df["Passport"].map({"Yes": 1, "No": 0})
input_df["OwnCar"] = input_df["OwnCar"].map({"Yes": 1, "No": 0})

st.subheader("Input Data")
st.dataframe(input_df, use_container_width=True)

if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("The customer is likely to buy the travel package.")
    else:
        st.warning("The customer is not likely to buy the travel package.")
