import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# App title
st.title("🏦 Loan Default Prediction App")
st.write("Enter applicant details to predict loan default risk.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.slider("Applicant Income", 0, 100000, 1000)
coapplicant_income = st.slider("Coapplicant Income", 0, 50000, 500)
loan_amount = st.slider("Loan Amount (in thousands)", 0, 1000, 10)
loan_amount_term = st.slider("Loan Amount Term (in months)", 0, 480, 12)
credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert inputs to numeric codes
gender_code = 1 if gender == "Male" else 0
married_code = 1 if married == "Yes" else 0
dependents_code = 3 if dependents == "3+" else int(dependents)
education_code = 0 if education == "Graduate" else 1
self_employed_code = 1 if self_employed == "Yes" else 0
credit_history_code = 1 if credit_history == "Good (1)" else 0
property_area_code = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# Create input array
input_data = np.array([[gender_code, married_code, dependents_code, education_code,
                        self_employed_code, applicant_income, coapplicant_income,
                        loan_amount, loan_amount_term, credit_history_code, property_area_code]])

# Scale numeric features
input_scaled = scaler.transform(input_data)

# Predict and display results
if st.button("Predict Loan Default"):
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error("❌ High Risk: Loan Likely to Default.")
    else:
        st.success("✅ Low Risk: Loan Likely to be Approved.")

   # --- Chart 1: Input Summary ---
st.subheader("📊 Input Summary")

# Separate numeric and categorical features
numeric_features = {
    "Applicant Income": applicant_income,
    "Coapplicant Income": coapplicant_income,
    "Loan Amount (in thousands)": loan_amount,
    "Loan Term (months)": loan_amount_term
}

categorical_features = {
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self Employed": self_employed,
    "Credit History": credit_history,
    "Property Area": property_area
}

# Display categorical inputs as table
st.markdown("#### Categorical Inputs")
st.table(pd.DataFrame(categorical_features.items(), columns=["Feature", "Value"]))

# Display numeric inputs as bar chart
st.markdown("#### Numeric Inputs")
num_df = pd.DataFrame(numeric_features.items(), columns=["Feature", "Value"])
fig_num, ax_num = plt.subplots(figsize=(8, 4))
sns.barplot(y="Feature", x="Value", data=num_df, ax=ax_num, palette="Blues_d")
ax_num.set_title("Numeric Input Feature Values")
st.pyplot(fig_num)
 
