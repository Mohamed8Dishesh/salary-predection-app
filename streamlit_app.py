import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# Set page configuration
st.set_page_config(page_title="Salary Prediction App", page_icon="📈", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #3b82f6;
    }
    .header {
        color: #1e3a8a;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .subheader {
        color: #4b5563;
        font-size: 18px;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-title {
        color: #1e3a8a;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Check if all required files exist
required_files = [
    "best_model.pkl",
    "scaler.pkl",
    "gender_encoder.pkl",
    "job_title_encoder.pkl",
    "job_title_mapping.json",
    "X_train_LabelEncoded.csv"
]

for file in required_files:
    if not os.path.exists(file):
        st.error(f"Required file not found: {file}")
        st.stop()

# Load the model, scaler, and encoders
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
le_gender = joblib.load("gender_encoder.pkl")
le_job_title = joblib.load("job_title_encoder.pkl")

# Load job title mapping
with open("job_title_mapping.json", 'r') as f:
    job_title_mapping = json.load(f)

# Define the expected columns (based on X_train_LabelEncoded.csv)
expected_columns = pd.read_csv("X_train_LabelEncoded.csv").columns.tolist()

# Streamlit UI
st.markdown('<div class="header">📈 Salary Prediction App</div>', unsafe_allow_html=True)
st.markdown("""
    <div class="subheader">
    Discover your potential salary with our advanced machine learning model! 
    Enter your details like age, experience, and job title, and let our XGBoost-powered app predict your earnings accurately.
    </div>
""", unsafe_allow_html=True)

# Optional: Add a name input for personalization
user_name = st.text_input("Your Name (Optional)", placeholder="Enter your name for a personalized result")

# Input fields
with st.form(key="prediction_form"):
    st.markdown('<div class="section-title">Personal Information</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    with col2:
        years_of_experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    
    st.markdown('<div class="section-title">Job Details</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        education_level = st.selectbox("Education Level", options=[0, 1, 2, 3], format_func=lambda x: ["High School", "Bachelor’s", “Master’s", "PhD"][x])
        gender = st.selectbox("Gender", options=list(le_gender.classes_))
    with col4:
        senior = st.selectbox("Senior Status", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        job_title = st.selectbox("Job Title", options=list(job_title_mapping.keys()))

    # Optional fields for Country and Race
    country = None
    race = None
    country_columns = [col for col in expected_columns if col.startswith("Country_")]
    race_columns = [col for col in expected_columns if col.startswith("Race_")]
    
    if country_columns or race_columns:
        st.markdown('<div class="section-title">Additional Information</div>', unsafe_allow_html=True)
        col5, col6 = st.columns(2)
        if country_columns:
            with col5:
                country_options = [col.replace("Country_", "") for col in country_columns]
                country = st.selectbox("Country", options=country_options)
        if race_columns:
            with col6:
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

        # Display personalized result
        if user_name:
            st.success(f"Hi {user_name}, based on your inputs, your predicted salary is: ${prediction:,.2f}")
        else:
            st.success(f"Based on your inputs, your predicted salary is: ${prediction:,.2f}")

        # Add a financial tip
        if years_of_experience < 5:
            st.info("💡 Tip: Gaining more years of experience could significantly boost your salary!")
        elif education_level < 2:
            st.info("💡 Tip: Consider pursuing a Master’s or PhD to increase your earning potential!")

    except Exception as e:
        st.error(f"Error: {str(e)}")

# Add instructions
st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
st.markdown("""
- **Ag**: Enter your age (18-100).
- **Years of Experience**: Enter your years of work experience (0-50).
- **Education Level**: Select your education level (0: High School, 1: Bachelor’s, 2: Master’s, 3: PhD).
- **Senior Status**: Select whether you are a senior employee (Yes/No).
- **Gender**: Select your gender.
- **Job Title**: Select your job title from the list.
- **Country/Race**: Select if applicable.

Our app uses a powerful XGBoost model trained on real-world data to provide accurate salary predictions. 
""")
