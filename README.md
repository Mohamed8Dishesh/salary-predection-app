# Salary Prediction App

This web application predicts salary based on user-provided features using machine learning models. The project involved data acquisition from Kaggle, followed by comprehensive data cleaning and Exploratory Data Analysis (EDA) using Python. Feature engineering techniques were applied to prepare the data for model training.

Three machine learning models were trained and evaluated: Linear Regression, Random Forest, and XGBoost. The XGBoost model achieved the highest accuracy of approximately 0.97 on the evaluation data, making it the selected best model for deployment.

The trained XGBoost model is deployed using Streamlit, providing an interactive web interface where users can input their data and receive a predicted salary.

## Demo App

[![Salary Prediction App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smartsalaryapp.streamlit.app/)

Click the badge above to open the Salary Prediction App in your browser.


## Data Cleaning and EDA

The data cleaning phase involved handling missing values, identifying and addressing outliers, and ensuring data consistency. Exploratory Data Analysis (EDA) was performed to gain insights into the data distribution, relationships between features, and potential patterns relevant for salary prediction. This included visualizations and statistical summaries to understand the dataset characteristics.

## Feature Engineering and Machine Learning Models

Feature engineering techniques were applied to create informative features for the models. This included [mention specific techniques if you remember, e.g., encoding categorical variables, scaling numerical features, creating interaction terms].

The following machine learning models were trained and evaluated:

* **Linear Regression:** A basic linear model used as a baseline.
* **Random Forest:** An ensemble learning method based on decision trees.
* **XGBoost (Extreme Gradient Boosting):** A powerful gradient boosting algorithm known for its high performance.

Model performance was evaluated using appropriate metrics [mention the specific metric if you remember, e.g., R-squared, Mean Squared Error, Accuracy]. The **XGBoost model demonstrated the best performance with an accuracy of approximately 0.97**, outperforming Linear Regression and Random Forest. This trained XGBoost model (`best_model.pkl`) was then used for the Streamlit application.

## Deployment

The best performing model (XGBoost) was deployed using the Streamlit library. This allows users to interact with the model through a user-friendly web interface. Users can input relevant features (e.g., job title, experience, location) through the Streamlit app, and the trained model predicts the corresponding salary. The `streamlit_app.py` file contains the code for the web application.

## Author

Mohamed Dehish ([GitHub Profile](https://github.com/Mohamed8Dishesh))

Connect with me on [LinkedIn](https://www.linkedin.com/in/mohamed-deshish/).
