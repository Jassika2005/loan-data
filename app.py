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

    # -- Chart 1: Input Summary
    st.subheader("📊 Input Summary")
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
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.barplot(y="Feature", x="Value", data=input_df, ax=ax1, palette="Blues_d")
    ax1.set_title("Input Feature Values")
    st.pyplot(fig1)

    # -- Chart 2: Feature Importance (works for tree-based or logistic models)
    st.subheader("📈 Model Feature Importance")
    feature_names = ["Gender", "Married", "Dependents", "Education", "Self Employed",
                     "Applicant Income", "Coapplicant Income", "Loan Amount",
                     "Loan Term", "Credit History", "Property Area"]

    if hasattr(model, "feature_importances_"):
        # For RandomForest, DecisionTree, etc.
        importance = model.feature_importances_
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(y="Feature", x="Importance", data=importance_df, palette="viridis", ax=ax2)
        ax2.set_title("Feature Importance (Tree-based Model)")
        st.pyplot(fig2)

    elif hasattr(model, "coef_"):
        # For Logistic Regression
        importance = model.coef_[0]
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.barplot(y="Feature", x="Importance", data=importance_df, palette="coolwarm", ax=ax3)
        ax3.set_title("Feature Coefficients (Logistic Regression)")
        st.pyplot(fig3)

    else:
        st.info("❗ This model does not support feature importance visualization.")

    # -- Optional: PDP for Applicant Income
    st.subheader("📉 Income vs. Default Risk")
    incomes = np.linspace(0, 100000, 50)
    probs = []
    for inc in incomes:
        temp = input_data.copy()
        temp[0, 5] = inc  # change applicant income
        temp_scaled = scaler.transform(temp)
        try:
            prob = model.predict_proba(temp_scaled)[0][1]
        except:
            prob = model.predict(temp_scaled)[0]
        probs.append(prob)

    fig4, ax4 = plt.subplots()
    ax4.plot(incomes, probs, color="purple", marker="o")
    ax4.set_title("Partial Dependence: Applicant Income")
    ax4.set_xlabel("Applicant Income")
    ax4.set_ylabel("Predicted Risk")
    st.pyplot(fig4)
