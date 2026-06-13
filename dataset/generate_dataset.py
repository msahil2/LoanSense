"""
generate_dataset.py
--------------------
Generates a synthetic Loan Approval dataset (loan_data.csv) with realistic
distributions for the Loan Approval Prediction System.

Run:
    python dataset/generate_dataset.py
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 1200  # number of records

genders = np.random.choice(["Male", "Female"], size=N, p=[0.78, 0.22])
married = np.random.choice(["Yes", "No"], size=N, p=[0.65, 0.35])
dependents = np.random.choice(["0", "1", "2", "3+"], size=N, p=[0.55, 0.18, 0.17, 0.10])
education = np.random.choice(["Graduate", "Not Graduate"], size=N, p=[0.78, 0.22])
self_employed = np.random.choice(["Yes", "No"], size=N, p=[0.14, 0.86])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], size=N, p=[0.38, 0.38, 0.24])

applicant_income = np.random.gamma(shape=5.0, scale=1200, size=N).round(0) + 1500
coapplicant_income = np.where(
    married == "Yes",
    np.random.gamma(shape=3.0, scale=900, size=N).round(0),
    0
)

loan_amount = (np.random.gamma(shape=4.0, scale=35, size=N) + 50).round(0)
loan_amount_term = np.random.choice([360, 180, 120, 60, 300, 240, 84, 36, 12], size=N,
                                     p=[0.55, 0.12, 0.08, 0.05, 0.05, 0.05, 0.04, 0.03, 0.03])

# Credit history strongly correlated with approval
credit_history = np.random.choice([1.0, 0.0], size=N, p=[0.84, 0.16])

# Introduce some missing values to mimic real-world data
def add_missing(arr, frac=0.04):
    arr = arr.astype(object)
    idx = np.random.choice(len(arr), size=int(len(arr) * frac), replace=False)
    arr[idx] = np.nan
    return arr

gender_m = add_missing(genders, 0.02)
married_m = add_missing(married, 0.02)
dependents_m = add_missing(dependents, 0.03)
self_employed_m = add_missing(self_employed, 0.05)
loan_amount_m = add_missing(loan_amount.astype(float), 0.03)
loan_amount_term_m = add_missing(loan_amount_term.astype(float), 0.02)
credit_history_m = add_missing(credit_history.astype(float), 0.06)

# -----------------------------------------------------------------
# Generate target variable (Loan_Status) using a logical scoring rule
# so that the dataset has real, learnable signal.
# -----------------------------------------------------------------
total_income = applicant_income + coapplicant_income
income_to_loan = total_income / (loan_amount * 1000 + 1)

score = (
    (credit_history == 1.0).astype(float) * 3.0
    + (income_to_loan > 0.03).astype(float) * 1.5
    + (education == "Graduate").astype(float) * 0.6
    + (property_area == "Semiurban").astype(float) * 0.5
    + (married == "Yes").astype(float) * 0.3
    - (dependents == "3+").astype(float) * 0.4
    - (loan_amount > 200).astype(float) * 0.7
    + np.random.normal(0, 1.2, size=N)  # noise
)

loan_status = np.where(score > 3.0, "Y", "N")

df = pd.DataFrame({
    "Loan_ID": [f"LP{1000+i}" for i in range(N)],
    "Gender": gender_m,
    "Married": married_m,
    "Dependents": dependents_m,
    "Education": education,
    "Self_Employed": self_employed_m,
    "ApplicantIncome": applicant_income.astype(int),
    "CoapplicantIncome": coapplicant_income.astype(int),
    "LoanAmount": loan_amount_m,
    "Loan_Amount_Term": loan_amount_term_m,
    "Credit_History": credit_history_m,
    "Property_Area": property_area,
    "Loan_Status": loan_status,
})

out_path = os.path.join(os.path.dirname(__file__), "loan_data.csv")
df.to_csv(out_path, index=False)
print(f"Dataset generated: {out_path}")
print(df["Loan_Status"].value_counts())
