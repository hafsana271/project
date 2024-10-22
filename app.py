import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app
st.title("Employee Promotion Prediction")

# Create input fields for the features
employee_id = st.text_input("Employee ID")
department = st.selectbox("Department", [
    "Finance", "HR", "Legal", "Operations", "Procurement", 
    "R&D", "Sales & Marketing", "Technology"
])
region = st.selectbox("Region", [f"region_{i}" for i in range(1, 35)])  # Assuming there are 34 regions
education = st.selectbox("Education", ["Bachelor's", "Master's & above", "Below Secondary"])
gender = st.selectbox("Gender", ["m", "f"])
recruitment_channel = st.selectbox("Recruitment Channel", ["sourcing", "referred", "other"])
no_of_trainings = st.number_input("Number of Trainings", min_value=1, max_value=10, value=1)
age = st.number_input("Age", min_value=18, max_value=60, value=30)
previous_year_rating = st.number_input("Previous Year Rating", min_value=1.0, max_value=5.0, value=3.0)
length_of_service = st.number_input("Length of Service (years)", min_value=1, max_value=40, value=5)
KPIs_met = st.selectbox("KPIs met >80%", [0, 1])
awards_won = st.selectbox("Awards Won?", [0, 1])
avg_training_score = st.number_input("Average Training Score", min_value=0, max_value=100, value=70)

# Load the original feature names used during training
training_feature_names = model.feature_names_in_

# Mapping categorical features to one-hot encoding format
department_columns = [f"department_{dep}" for dep in [
    "Finance", "HR", "Legal", "Operations", 
    "Procurement", "R&D", "Sales & Marketing", "Technology"
]]
region_columns = [f"region_region_{i}" for i in range(1, 35)]
education_columns = ["education_Below Secondary", "education_Bachelor's", "education_Master's & above"]
recruitment_channel_columns = ["recruitment_channel_other", "recruitment_channel_referred", "recruitment_channel_sourcing"]

# Create the input DataFrame with all features
input_data = {
    'age': age,
    'no_of_trainings': no_of_trainings,
    'previous_year_rating': previous_year_rating,
    'length_of_service': length_of_service,
    'KPIs_met >80%': KPIs_met,
    'awards_won?': awards_won,
    'avg_training_score': avg_training_score,
    **{col: 0 for col in department_columns},
    **{col: 0 for col in region_columns},
    **{col: 0 for col in education_columns},
    **{col: 0 for col in recruitment_channel_columns},
    'employee_id': 0,  # Add 'employee_id' with a default value
    'gender_m': 1 if gender == 'm' else 0  # Binary encoding for gender
}

# Set the correct one-hot encoded values
input_data[f"department_{department}"] = 1
input_data[f"region_{region}"] = 1
input_data[f"education_{education}"] = 1
input_data[f"recruitment_channel_{recruitment_channel}"] = 1

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Add any missing columns from the training feature set and reorder the DataFrame
for feature in training_feature_names:
    if feature not in input_df.columns:
        input_df[feature] = 0  # Add missing features with a default value

input_df = input_df[training_feature_names]


# Prediction
if st.button("Predict Promotion Status"):
    prediction = model.predict(input_df)
    st.write(f"The predicted promotion status for employee {employee_id} is: {int(prediction[0])}")
