import streamlit as st
import joblib

"""
# Employee Performance Prediction System

Edit employee's detail below to predict his performance:
"""

model = joblib.load('model.pkl')
scalers = joblib.load('scaler.pkl')

def predict_performance(features):
    features_to_scale = ['EmployeeType', 'PayZone', 'DepartmentType', 'GenderCode', 
                         'RaceDesc', 'MaritalDesc', 'CurrentEmployeeRating', 
                         'EngagementScore', 'SatisfactionScore', 'WorkLifeBalanceScore']
    scaled_features = []

    for i, feature in enumerate(features):
        feature_name = features_to_scale[i]
        scaled_feature = scalers[feature_name].transform([[feature]])[0][0]
        scaled_features.append(scaled_feature)

    prediction = model.predict([scaled_features])
    return prediction[0]

EmployeeType = st.selectbox("Employee Type", [0, 1, 2], format_func=lambda x: 'Contract' if x == 0 else 'Full Time' if x == 1 else 'Part Time')
PayZone = st.selectbox("Pay Zone", [0, 1, 2], format_func=lambda x: 'Zone A' if x == 0 else 'Zone B' if x == 1 else 'Zone C')
DepartmentType = st.selectbox("Department Type", [0, 1, 2, 3, 4, 5], format_func=lambda x: 'Admin Offices' if x == 0 else 'Executive Office' if x == 1 else 'IT / IS' if x == 2 else 'Production' if x == 3 else 'Sales' if x == 4 else 'Software Engineering')
GenderCode = st.selectbox("Gender", [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
RaceDesc = st.selectbox("Race", [0, 1, 2, 3, 4], format_func=lambda x: 'Asian' if x == 0 else 'Black' if x == 1 else 'Hispanic' if x == 2 else 'Other' if x == 3 else 'White')
MaritalDesc = st.selectbox("Marital Status", [0, 1, 2, 3], format_func=lambda x: 'Divorced' if x == 0 else 'Married' if x == 1 else 'Single' if x == 2 else 'Widowed')
CurrentEmployeeRating = st.slider("Current Employee Rating", 0, 5, 3)
EngagementScore = st.slider("Engagement Score", 0, 5, 3)
SatisfactionScore = st.slider("Satisfaction Score", 0, 5, 3)
WorkLifeBalanceScore = st.slider("Word-Life Balance Score", 0, 5, 3)

features = [EmployeeType, PayZone, DepartmentType, GenderCode, RaceDesc, MaritalDesc, CurrentEmployeeRating, EngagementScore, SatisfactionScore, WorkLifeBalanceScore]

if st.button("Predict"):
    result = predict_performance(features)
    st.write(f"Predicted performance: {'Exceeds' if result == 0 else 'Fully Meets' if result == 1 else 'Needs Improvement' if result == 2 else 'PIP'}")