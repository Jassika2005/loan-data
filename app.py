import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏦 Loan Default Prediction App")
st.write("Enter applicant details to predict loan default risk.")

# Input widgets
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

# Encoding
enc = {
    "gender": 1 if gender=="Male" else 0,
    "married": 1 if married=="Yes" else 0,
    "dependents": 3 if dependents=="3+" else int(dependents),
    "education": 0 if education=="Graduate" else 1,
    "self_employed": 1 if self_employed=="Yes" else 0,
    "credit_history": 1 if credit_history=="Good (1)" else 0,
    "property_area": {"Urban":2,"Semiurban":1,"Rural":0}[property_area]
}

input_array = np.array([[enc["gender"], enc["married"], enc["dependents"], enc["education"],
                         enc["self_employed"], applicant_income, coapplicant_income,
                         loan_amount, loan_amount_term, enc["credit_history"], enc["property_area"]]])
input_scaled = scaler.transform(input_array)

if st.button("Predict Loan Default"):
    prediction = model.predict(input_scaled)[0]
    plain_result = "High Risk: Loan Likely to Default." if prediction == 1 else "Low Risk: Loan Likely to be Approved."
    st.error("❌ "+plain_result) if prediction == 1 else st.success("✅ "+plain_result)

    # Input Summary chart
    st.subheader("Input Summary")
    input_dict = {
        "Gender": gender, "Married": married, "Dependents": dependents,
        "Education": education, "Self Employed": self_employed,
        "Applicant Income": applicant_income, "Coapplicant Income": coapplicant_income,
        "Loan Amount": loan_amount, "Loan Term": loan_amount_term,
        "Credit History": credit_history, "Property Area": property_area
    }
    df = pd.DataFrame(input_dict.items(), columns=["Feature","Value"])
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(y="Feature", x="Value", data=df, palette="Blues_d", ax=ax)
    ax.set_title("Input Feature Values")
    st.pyplot(fig)

    # Coefficient Importance chart for Logistic Regression
    if hasattr(model, "coef_"):
        coeffs = model.coef_[0]
        features = ["Gender","Married","Dependents","Education","Self Employed",
                    "Applicant Income","Coapplicant Income","Loan Amount",
                    "Loan Term","Credit History","Property Area"]
        coef_df = pd.DataFrame({"Feature":features, "Coefficient":coeffs}).sort_values(by="Coefficient", ascending=False)
        st.subheader("Model Feature Coefficients (Importance)")
        fig2, ax2 = plt.subplots(figsize=(8,5))
        sns.barplot(y="Feature", x="Coefficient", data=coef_df, palette="coolwarm", ax=ax2)
        ax2.set_title("Coefficient-based Feature Importance")
        st.pyplot(fig2)
    else:
        st.info("Coefficient information not available for this model.")

    # Optional: Basic Partial Dependence Plot for Applicant Income
    st.subheader("Partial Dependence: Applicant Income")
    incomes = np.linspace(0, 100000, 50)
    probs = []
    for inc in incomes:
        arr = input_array.copy()
        arr[0,5] = inc
        arr_scaled = scaler.transform(arr)
        probs.append(model.predict_proba(arr_scaled)[0][1])
    fig3, ax3 = plt.subplots(figsize=(6,4))
    ax3.plot(incomes, probs, marker="o")
    ax3.set_xlabel("Applicant Income")
    ax3.set_ylabel("Probability of Default")
    ax3.set_title("PDP: Effect of Income on Default Risk")
    st.pyplot(fig3)

    # PDF Report download
    pdf = FPDF()
    pdf.add_page(); pdf.set_font("Arial", size=12)
    pdf.cell(200,10,txt="Loan Default Prediction Report",ln=True, align='C')
    pdf.ln(10); pdf.cell(200,10,txt=f"Result: {plain_result}", ln=True)
    pdf.ln(5)
    for k,v in input_dict.items():
        pdf.cell(200,8,txt=f"{k}: {v}", ln=True)
    buf = io.BytesIO(); pdf.output(buf)
    buf.seek(0)
    st.download_button("Download Report as PDF", buf, "loan_report.pdf", "application/pdf")
