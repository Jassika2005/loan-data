import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io

# Inject background image via CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://raw.githubusercontent.com/your-username/your-repo-name/main/background.jpg");
        background-attachment: fixed;
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏦 Loan Default Prediction App")
...

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
    result = "❌ High Risk: Loan Likely to Default." if prediction[0] == 1 else "✅ Low Risk: Loan Likely to be Approved."
    plain_result = "High Risk" if prediction[0] == 1 else "Low Risk"

    st.subheader("🔍 Prediction Result")
    st.write(result)

    # Input summaries
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

    # Combine for report
    input_dict = {**categorical_features, **numeric_features}

    # Show inputs
    st.subheader("📊 Input Summary")

    st.markdown("#### Categorical Inputs")
    st.table(pd.DataFrame(categorical_features.items(), columns=["Feature", "Value"]))

    st.markdown("#### Numeric Inputs")
    num_df = pd.DataFrame(numeric_features.items(), columns=["Feature", "Value"])
    fig_num, ax_num = plt.subplots(figsize=(8, 4))
    sns.barplot(y="Feature", x="Value", data=num_df, ax=ax_num, palette="Blues_d")
    ax_num.set_title("Numeric Input Feature Values")
    st.pyplot(fig_num)

    # --- Generate PDF Report ---
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Loan Default Prediction Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Prediction Result: {plain_result}", ln=True)
    pdf.ln(5)

    for key, value in input_dict.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    # Save PDF to buffer
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    buffer = io.BytesIO(pdf_bytes)

    # Download button
    st.download_button(
        label="📥 Download Prediction Report as PDF",
        data=buffer,
        file_name="loan_prediction_report.pdf",
        mime="application/pdf"
    )

