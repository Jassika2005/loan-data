import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import base64

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

# Sliders for numeric input
applicant_income = st.slider("Applicant Income", min_value=0, max_value=100000, step=1000)
coapplicant_income = st.slider("Coapplicant Income", min_value=0, max_value=50000, step=500)
loan_amount = st.slider("Loan Amount (in thousands)", min_value=0, max_value=1000, step=10)
loan_amount_term = st.slider("Loan Amount Term (in months)", min_value=0, max_value=480, step=12)

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

# Predict
if st.button("Predict Loan Default"):
    prediction = model.predict(input_scaled)
    result = "❌ High Risk: Loan Likely to Default." if prediction[0] == 1 else "✅ Low Risk: Loan Likely to be Approved."
    if prediction[0] == 1:
        st.error(result)
    else:
        st.success(result)

    # --- Chart 1: User Input Summary ---
    st.subheader("📊 Your Input Summary")
    input_dict = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self Employed": self_employed,
        "Applicant Income": applicant_income,
        "Coapplicant Income": coapplicant_income,
        "Loan Amount": loan_amount,
        "Loan Term": loan_amount_term,
        "Credit History": credit_history,
        "Property Area": property_area
    }
    input_df = pd.DataFrame(input_dict.items(), columns=["Feature", "Value"])
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.barplot(y="Feature", x="Value", data=input_df, palette="Blues_d", ax=ax1)
    ax1.set_title("Input Feature Values")
    st.pyplot(fig1)

    # --- Chart 2: Feature Importance (if supported) ---
    if hasattr(model, "feature_importances_"):
        st.subheader("📈 Model Feature Importances")
        feature_names = ["Gender", "Married", "Dependents", "Education", "Self Employed",
                         "Applicant Income", "Coapplicant Income", "Loan Amount",
                         "Loan Term", "Credit History", "Property Area"]
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(y="Feature", x="Importance", data=importance_df, palette="viridis", ax=ax2)
        ax2.set_title("Model Feature Importance")
        st.pyplot(fig2)
    else:
        st.info("Feature importance chart not available for this model.")

   # --- PDF Report (Safe for Latin-1 Encoding) ---
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

pdf.cell(200, 10, txt="Loan Default Prediction Report", ln=True, align='C')
pdf.ln(10)

# Clean result string (remove emojis or non-latin1 chars)
plain_result = "High Risk: Loan Likely to Default." if prediction[0] == 1 else "Low Risk: Loan Likely to be Approved."
pdf.cell(200, 10, txt=f"Prediction Result: {plain_result}", ln=True)

pdf.ln(5)
for key, value in input_dict.items():
    key_str = str(key)
    value_str = str(value)
    # Avoid characters outside Latin-1
    pdf.cell(200, 10, txt=f"{key_str}: {value_str}", ln=True)

# Save PDF
pdf_output = "loan_prediction_report.pdf"
pdf.output(pdf_output)


