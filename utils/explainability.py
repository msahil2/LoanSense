"""
explainability.py
-------------------
A lightweight, dependency-free "SHAP-style" explanation module.

For tree-based and linear models we can compute a per-prediction
contribution score for each feature without requiring the heavy `shap`
library (which is often difficult to install in constrained deployment
environments). The approach:

  - For Logistic Regression: contribution = coefficient * (scaled feature value)
  - For tree-based models (Decision Tree / Random Forest): contribution is
    approximated using the global feature_importances_ weighted by how far
    the feature value deviates from the training-set mean (a simple but
    effective local-explanation proxy).

The output is a ranked list of the top contributing factors together with
a human-readable description of why each factor pushed the decision
toward approval or rejection.
"""

import numpy as np

from utils.preprocessing import FEATURE_LABELS, FEATURE_COLUMNS

# Approximate "reference" (average applicant) values used to determine
# whether a given applicant's value is above/below typical, for generating
# human-readable explanations. These mirror the synthetic dataset's
# central tendencies.
REFERENCE_VALUES = {
    "Gender": 1,
    "Married": 1,
    "Dependents": 0,
    "Education": 1,
    "Self_Employed": 0,
    "ApplicantIncome": 6000,
    "CoapplicantIncome": 1800,
    "LoanAmount": 140,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": 1,
    "TotalIncome": 7800,
    "IncomeLoanRatio": 0.05,
}


def _direction_text(feature: str, value: float, reference: float, contributes_positively: bool) -> str:
    """
    Build a short, human-friendly sentence describing how a feature value
    influenced the prediction.
    """
    label = FEATURE_LABELS.get(feature, feature)

    if feature == "Credit_History":
        if value >= 1:
            detail = "a clean credit history record"
        else:
            detail = "no verified credit history record"
    elif feature == "Education":
        detail = "a Graduate education level" if value >= 1 else "a Non-Graduate education level"
    elif feature == "Married":
        detail = "married applicant status" if value >= 1 else "single applicant status"
    elif feature == "Property_Area":
        mapping = {0: "a Rural property area", 1: "a Semiurban property area", 2: "an Urban property area"}
        detail = mapping.get(int(value), "the selected property area")
    elif feature in ("ApplicantIncome", "CoapplicantIncome", "TotalIncome"):
        cmp = "above" if value >= reference else "below"
        detail = f"a household income {cmp} the typical applicant average"
    elif feature == "LoanAmount":
        cmp = "higher than" if value >= reference else "lower than"
        detail = f"a requested loan amount {cmp} the typical applicant"
    elif feature == "IncomeLoanRatio":
        cmp = "healthy" if value >= reference else "tight"
        detail = f"a {cmp} income-to-loan ratio"
    elif feature == "Loan_Amount_Term":
        detail = f"a loan term of {int(value)} months"
    elif feature == "Dependents":
        detail = f"{int(value)} dependent(s)"
    else:
        detail = f"the provided {label.lower()}"

    if contributes_positively:
        return f"{label}: {detail}, which increased the chance of approval."
    else:
        return f"{label}: {detail}, which decreased the chance of approval."


def explain_prediction(model, model_name: str, scaler, feature_row, feature_importance: dict, top_n: int = 3):
    """
    Generate a top-N explanation for a single prediction.

    Parameters
    ----------
    model : trained sklearn estimator
    model_name : str
        Name of the model ("Logistic Regression", "Decision Tree", "Random Forest")
    scaler : fitted StandardScaler (used only for Logistic Regression)
    feature_row : pandas.DataFrame (single row) with columns == FEATURE_COLUMNS
    feature_importance : dict
        Global feature importance dictionary (feature_name -> importance weight)
    top_n : int
        Number of top factors to return.

    Returns
    -------
    list of dict
        Each dict has: feature, label, contribution (float), direction
        ("positive"/"negative"), explanation (str)
    """
    values = feature_row.iloc[0]
    contributions = {}

    if model_name == "Logistic Regression" and scaler is not None:
        scaled_values = scaler.transform(feature_row)[0]
        coefs = model.coef_[0]
        for i, feat in enumerate(FEATURE_COLUMNS):
            contributions[feat] = float(coefs[i] * scaled_values[i])
    else:
        # Local proxy explanation for tree-based models:
        # contribution = global_importance * sign(deviation from reference)
        for feat in FEATURE_COLUMNS:
            ref = REFERENCE_VALUES.get(feat, 0)
            val = float(values[feat])
            importance = feature_importance.get(feat, 0)
            # Normalize deviation to roughly [-1, 1] using a soft scale
            spread = max(abs(ref), 1.0)
            deviation = np.clip((val - ref) / spread, -1.5, 1.5)
            contributions[feat] = importance * deviation

    # Rank by absolute contribution magnitude
    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)

    explanations = []
    for feat, contrib in ranked[:top_n]:
        positive = contrib >= 0
        explanations.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "contribution": round(float(contrib), 4),
            "direction": "positive" if positive else "negative",
            "explanation": _direction_text(
                feat, float(values[feat]), REFERENCE_VALUES.get(feat, 0), positive
            ),
        })

    return explanations


def compute_risk_score(probability_approved: float) -> float:
    """
    Convert an approval probability (0-1) into a 0-100 'risk score' where
    a HIGHER score means HIGHER RISK (lower chance of approval).

    risk_score = (1 - probability_approved) * 100
    """
    risk = (1 - probability_approved) * 100
    return round(float(risk), 2)
