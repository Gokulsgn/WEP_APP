import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the model from the pickle file
model_path = r'C:\Users\gokul\Documents\GitHub\WEP_APP\APP_1\SVC.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Center the title
st.markdown("<h1 style='text-align: center;'>Loan Prediction Application</h1>", unsafe_allow_html=True)


# Input fields
gender = st.selectbox('Gender', ['Male', 'Female'])
married = st.selectbox('Married', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
applicant_income = st.number_input('Applicant Income', min_value=0)
coapplicant_income = st.number_input('Coapplicant Income', min_value=0)
loan_amount = st.number_input('Loan Amount', min_value=0)
loan_amount_term = st.number_input('Loan Amount Term', min_value=0)
credit_history = st.selectbox('Credit History', ['0', '1'])
property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

# Convert inputs into a format suitable for the model
def preprocess_input():
    # Encode categorical variables
    gender_num = 1 if gender == 'Male' else 0
    married_num = 1 if married == 'Yes' else 0
    dependents_num = 3 if dependents == '3+' else int(dependents)
    education_num = 1 if education == 'Graduate' else 0
    self_employed_num = 1 if self_employed == 'Yes' else 0
    credit_history_num = int(credit_history)
    
    # Property area encoding
    if property_area == 'Urban':
        property_area_num = 0
    elif property_area == 'Semiurban':
        property_area_num = 1
    else:
        property_area_num = 2

    # Return as a numpy array
    return np.array([[gender_num, married_num, dependents_num, education_num, self_employed_num,
                      applicant_income, coapplicant_income, loan_amount, loan_amount_term,
                      credit_history_num, property_area_num]])


# Center the button using Streamlit layout
col1, col2, col3 = st.columns([1, 1, 1])  # Create three columns with a 1:2:1 ratio

with col2:  # Center the button in the middle column
    if st.button("Predict Loan Status"):
        # Preprocess input
        input_data = preprocess_input()
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Convert prediction to scalar
        prediction = prediction[0]  # Extract the scalar value from the array
        
        # Display prediction result
        if prediction == 1:
            st.success("Loan Approved!")
        else:
            st.error("Loan Rejected!")
    


