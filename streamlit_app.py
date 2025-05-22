import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# Set page configuration
st.set_page_config(page_title="Salary Prediction App", page_icon="💼", layout="centered")

# Check if all required files exist
required_files = [
    r"D:\SpaceCode_GraduationProject\best_model.pkl",
    r"D:\SpaceCode_GraduationProject\scaler.pkl",
    r"D:\SpaceCode_GraduationProject\gender_encoder.pkl",
    r"D:\SpaceCode_GraduationProject\job_title_encoder.pkl",
    r"D:\SpaceCode_GraduationProject\job_title_mapping.json",
    r"D:\SpaceCode_GraduationProject\X_train_LabelEncoded.csv"
]

for file in required_files:
    if not os.path.exists(file):
        st.error(f"Required file not found: {file}")
        st.stop()

# Load the model, scaler, and encoders
model = joblib.load(r"D:\SpaceCode_GraduationProject\best_model.pkl")
scaler = joblib.load(r"D:\SpaceCode_GraduationProject\scaler.pkl")
le_gender = joblib.load(r"D:\SpaceCode_GraduationProject\gender_encoder.pkl")
le_job_title = joblib.load(r"D:\SpaceCode_GraduationProject\job_title_encoder.pkl")

# Load job title mapping
with open(r"D:\SpaceCode_GraduationProject\job_title_mapping.json", 'r') as f:
    job_title_mapping = json.load(f)

# Define the expected columns (based on X_train_LabelEncoded.csv)
expected_columns = pd.read_csv(r"D:\SpaceCode_GraduationProject\X_train_LabelEncoded.csv").columns.tolist()

# Streamlit UI
st.title("💼 Salary Prediction App")
st.write("Enter your details to predict your salary.")

# Input fields
with st.form(key="prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    years_of_experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    education_level = st.selectbox("Education Level", options=[0, 1, 2, 3], format_func=lambda x: ["High School", "Bachelor’s", "Master’s", "PhD"][x])
    senior = st.selectbox("Senior Status", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    gender = st.selectbox("Gender", options=list(le_gender.classes_))
    job_title = st.selectbox("Job Title", options=list(job_title_mapping.keys()))

    # Optional fields for Country and Race (if they exist in expected_columns)
    country = None
    race = None
    country_columns = [col for col in expected_columns if col.startswith("Country_")]
    race_columns = [col for col in expected_columns if col.startswith("Race_")]
    
    if country_columns:
        country_options = [col.replace("Country_", "") for col in country_columns]
        country = st.selectbox("Country", options=country_options)
    if race_columns:
        race_options = [col.replace("Race_", "") for col in race_columns]
        race = st.selectbox("Race", options=race_options)

    # Submit button
    submit_button = st.form_submit_button(label="Predict Salary")

# Process input and predict
if submit_button:
    try:
        # Create input data dictionary
        input_data = {
            'Age': age,
            'Years of Experience': years_of_experience,
            'Education Level': education_level,
            'Senior': senior,
            'Experience_to_Age_Ratio': years_of_experience / age,
            'Gender_Encoded': le_gender.transform([gender])[0],
            'Job_Title_Encoded': le_job_title.transform([job_title])[0]
        }

        # Add Country and Race columns if they exist
        for col in expected_columns:
            if col.startswith('Country_') or col.startswith('Race_'):
                input_data[col] = 0  # Default to 0
            if country and col == f"Country_{country}":
                input_data[col] = 1
            if race and col == f"Race_{race}":
                input_data[col] = 1

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=expected_columns)

        # Scale numeric features
        numeric_cols = ['Age', 'Years of Experience', 'Education Level', 'Experience_to_Age_Ratio']
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Display result
        st.success(f"Predicted Salary: ${prediction:,.2f}")

    except Exception as e:
        st.error(f"Error: {str(e)}")

# Add some instructions
st.markdown("""
### Instructions
- **Age**: Enter your age (18-100).
- **Years of Experience**: Enter your years of work experience (0-50).
- **Education Level**: Select your education level (0: High School, 1: Bachelor’s, 2: Master’s, 3: PhD).
- **Senior Status**: Select whether you are a senior employee (Yes/No).
- **Gender**: Select your gender.
- **Job Title**: Select your job title from the list.
- **Country/Race**: Select if applicable.
""")
