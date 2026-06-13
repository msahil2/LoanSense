"""
preprocessing.py
-----------------
Shared data-cleaning and feature-engineering utilities used by both the
training pipeline (train.py) and the Flask application (app.py).

Keeping this logic in one place guarantees that the exact same
transformations are applied at training time and at prediction time.
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Constant mappings used for encoding categorical fields.
# These mappings MUST stay consistent between training and inference.
# ----------------------------------------------------------------------
GENDER_MAP = {"Male": 1, "Female": 0}
MARRIED_MAP = {"Yes": 1, "No": 0}
EDUCATION_MAP = {"Graduate": 1, "Not Graduate": 0}
SELF_EMPLOYED_MAP = {"Yes": 1, "No": 0}
DEPENDENTS_MAP = {"0": 0, "1": 1, "2": 2, "3+": 3}
PROPERTY_AREA_MAP = {"Rural": 0, "Semiurban": 1, "Urban": 2}
LOAN_STATUS_MAP = {"Y": 1, "N": 0}

# Final feature order used by the model. Order matters!
FEATURE_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
    "TotalIncome",
    "IncomeLoanRatio",
]

# Human readable labels for the explainable-AI section
FEATURE_LABELS = {
    "Gender": "Gender",
    "Married": "Marital Status",
    "Dependents": "Number of Dependents",
    "Education": "Education Level",
    "Self_Employed": "Self-Employment Status",
    "ApplicantIncome": "Applicant Income",
    "CoapplicantIncome": "Co-applicant Income",
    "LoanAmount": "Loan Amount",
    "Loan_Amount_Term": "Loan Term",
    "Credit_History": "Credit History",
    "Property_Area": "Property Area",
    "TotalIncome": "Total Household Income",
    "IncomeLoanRatio": "Income-to-Loan Ratio",
}


def clean_and_engineer(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Clean raw loan-application data and create engineered features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe containing the original (string-valued) columns.
    is_training : bool
        If True, the dataframe is expected to contain the Loan_Status
        target column, which will also be encoded.

    Returns
    -------
    pd.DataFrame
        Fully cleaned & encoded dataframe ready for the ML pipeline.
    """
    data = df.copy()

    # ------------------------------------------------------------
    # 1. Missing value handling
    #    - Categorical columns -> fill with mode
    #    - Numerical columns   -> fill with median
    # ------------------------------------------------------------
    categorical_cols = ["Gender", "Married", "Dependents", "Self_Employed"]
    for col in categorical_cols:
        if col in data.columns and data[col].isnull().any():
            mode_val = data[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            data[col] = data[col].fillna(fill_val)

    numeric_fill_median = ["LoanAmount", "Loan_Amount_Term"]
    for col in numeric_fill_median:
        if col in data.columns and data[col].isnull().any():
            data[col] = data[col].astype(float)
            data[col] = data[col].fillna(data[col].median())

    if "Credit_History" in data.columns and data["Credit_History"].isnull().any():
        # Most common scenario: missing credit history defaults to "has history"
        data["Credit_History"] = data["Credit_History"].fillna(1.0)

    for col in ["ApplicantIncome", "CoapplicantIncome"]:
        if col in data.columns:
            data[col] = data[col].fillna(0)

    # ------------------------------------------------------------
    # 2. Label Encoding for categorical fields
    # ------------------------------------------------------------
    data["Gender"] = data["Gender"].map(GENDER_MAP).fillna(1).astype(int)
    data["Married"] = data["Married"].map(MARRIED_MAP).fillna(0).astype(int)
    data["Education"] = data["Education"].map(EDUCATION_MAP).fillna(1).astype(int)
    data["Self_Employed"] = data["Self_Employed"].map(SELF_EMPLOYED_MAP).fillna(0).astype(int)
    data["Dependents"] = (
        data["Dependents"].astype(str).map(DEPENDENTS_MAP).fillna(0).astype(int)
    )
    data["Property_Area"] = (
        data["Property_Area"].map(PROPERTY_AREA_MAP).fillna(2).astype(int)
    )

    data["Credit_History"] = data["Credit_History"].astype(float)
    data["ApplicantIncome"] = data["ApplicantIncome"].astype(float)
    data["CoapplicantIncome"] = data["CoapplicantIncome"].astype(float)
    data["LoanAmount"] = data["LoanAmount"].astype(float)
    data["Loan_Amount_Term"] = data["Loan_Amount_Term"].astype(float)

    # ------------------------------------------------------------
    # 3. Feature Engineering
    # ------------------------------------------------------------
    data["TotalIncome"] = data["ApplicantIncome"] + data["CoapplicantIncome"]

    # Avoid division by zero; LoanAmount is expressed in thousands in this dataset
    safe_loan_amount = data["LoanAmount"].replace(0, np.nan)
    data["IncomeLoanRatio"] = (data["TotalIncome"] / (safe_loan_amount * 1000))
    data["IncomeLoanRatio"] = data["IncomeLoanRatio"].fillna(0)
    # Clip extreme outliers so they don't dominate scaling
    data["IncomeLoanRatio"] = data["IncomeLoanRatio"].clip(upper=1.0)

    # ------------------------------------------------------------
    # 4. Target encoding (training only)
    # ------------------------------------------------------------
    if is_training and "Loan_Status" in data.columns:
        data["Loan_Status"] = data["Loan_Status"].map(LOAN_STATUS_MAP)

    return data


def build_feature_frame(input_dict: dict) -> pd.DataFrame:
    """
    Convert a dictionary of raw form inputs (as received from the web
    form / API) into a single-row, fully-engineered DataFrame ready for
    model.predict().

    Parameters
    ----------
    input_dict : dict
        Dictionary with raw keys matching the dataset columns
        (string values exactly as chosen in the HTML form).

    Returns
    -------
    pd.DataFrame
        Single row dataframe with columns ordered as FEATURE_COLUMNS.
    """
    raw = {
        "Gender": input_dict.get("Gender", "Male"),
        "Married": input_dict.get("Married", "No"),
        "Dependents": input_dict.get("Dependents", "0"),
        "Education": input_dict.get("Education", "Graduate"),
        "Self_Employed": input_dict.get("Self_Employed", "No"),
        "ApplicantIncome": float(input_dict.get("ApplicantIncome", 0) or 0),
        "CoapplicantIncome": float(input_dict.get("CoapplicantIncome", 0) or 0),
        "LoanAmount": float(input_dict.get("LoanAmount", 0) or 0),
        "Loan_Amount_Term": float(input_dict.get("Loan_Amount_Term", 360) or 360),
        "Credit_History": float(input_dict.get("Credit_History", 1) or 0),
        "Property_Area": input_dict.get("Property_Area", "Urban"),
    }

    df = pd.DataFrame([raw])
    df = clean_and_engineer(df, is_training=False)
    return df[FEATURE_COLUMNS]
