"""
train.py
---------
End-to-end Machine Learning training pipeline for the Loan Approval
Prediction System.

Pipeline steps:
    1. Load raw dataset (dataset/loan_data.csv)
    2. Clean data & handle missing values
    3. Label encode categorical features
    4. Feature engineering (TotalIncome, IncomeLoanRatio)
    5. Train/Test split
    6. Train multiple models:
        - Logistic Regression
        - Decision Tree
        - Random Forest
    7. Evaluate all models (accuracy, precision, recall, f1)
    8. Automatically select the best model
    9. Save the best model + scaler + metadata using Joblib
   10. Save evaluation metrics & feature importances for the dashboard

Run:
    python train.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from utils.preprocessing import clean_and_engineer, FEATURE_COLUMNS

# ----------------------------------------------------------------------
# Paths (relative - no hardcoded absolute paths)
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "loan_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance.json")
ALL_MODELS_PATH = os.path.join(MODEL_DIR, "all_models.pkl")


def load_data() -> pd.DataFrame:
    """Load the raw CSV dataset."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Run 'python dataset/generate_dataset.py' first."
        )
    return pd.read_csv(DATASET_PATH)


def train_models():
    print("=" * 60)
    print("LOAN APPROVAL PREDICTION SYSTEM - MODEL TRAINING")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. Load & clean data
    # ------------------------------------------------------------
    print("\n[1/6] Loading dataset...")
    raw_df = load_data()
    print(f"      Loaded {len(raw_df)} rows.")

    print("[2/6] Cleaning data, encoding labels, engineering features...")
    df = clean_and_engineer(raw_df, is_training=True)
    df = df.dropna(subset=["Loan_Status"])  # safety net

    X = df[FEATURE_COLUMNS]
    y = df["Loan_Status"].astype(int)

    # ------------------------------------------------------------
    # 2. Train/Test split
    # ------------------------------------------------------------
    print("[3/6] Splitting data into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numeric features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------
    # 3. Train multiple models
    # ------------------------------------------------------------
    print("[4/6] Training models: Logistic Regression, Decision Tree, Random Forest...")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42
        ),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds).tolist()

        results[name] = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "confusion_matrix": cm,
        }
        trained_models[name] = model

        print(f"      {name:<22} | Accuracy: {acc*100:6.2f}% | F1: {f1*100:6.2f}%")

    # ------------------------------------------------------------
    # 4. Select best model (by F1 score, tie-break on accuracy)
    # ------------------------------------------------------------
    print("[5/6] Selecting best model based on F1-score...")
    best_name = max(
        results, key=lambda k: (results[k]["f1_score"], results[k]["accuracy"])
    )
    best_model = trained_models[best_name]
    print(f"      >> Best model: {best_name} "
          f"(F1: {results[best_name]['f1_score']}%, "
          f"Accuracy: {results[best_name]['accuracy']}%)")

    # ------------------------------------------------------------
    # 5. Feature importance (Explainable AI section)
    # ------------------------------------------------------------
    print("[6/6] Computing feature importances & saving artifacts...")

    feature_importance = {}
    if best_name == "Logistic Regression":
        importances = np.abs(best_model.coef_[0])
    elif hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    else:
        importances = np.ones(len(FEATURE_COLUMNS)) / len(FEATURE_COLUMNS)

    # Normalize to sum to 1 for easy percentage display
    importances = np.array(importances, dtype=float)
    if importances.sum() > 0:
        importances = importances / importances.sum()

    for col, imp in zip(FEATURE_COLUMNS, importances):
        feature_importance[col] = round(float(imp), 4)

    # Also compute feature importance from Random Forest regardless of which
    # model wins, since it's useful for the "Model Comparison" /
    # "Feature Importance" visualizations.
    rf_model = trained_models["Random Forest"]
    rf_importance = {
        col: round(float(imp), 4)
        for col, imp in zip(FEATURE_COLUMNS, rf_model.feature_importances_)
    }

    # ------------------------------------------------------------
    # 6. Save artifacts with Joblib
    # ------------------------------------------------------------
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(trained_models, ALL_MODELS_PATH)

    metadata = {
        "best_model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "results": results,
        "uses_scaler": best_name == "Logistic Regression",
        "trained_rows": len(df),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(
            {
                "best_model_importance": feature_importance,
                "random_forest_importance": rf_importance,
                "best_model_name": best_name,
            },
            f,
            indent=2,
        )

    print("\nTraining complete. Artifacts saved to /model:")
    print(f"  - {os.path.basename(MODEL_PATH)}")
    print(f"  - {os.path.basename(SCALER_PATH)}")
    print(f"  - {os.path.basename(ALL_MODELS_PATH)}")
    print(f"  - {os.path.basename(METRICS_PATH)}")
    print(f"  - {os.path.basename(FEATURE_IMPORTANCE_PATH)}")
    print("=" * 60)


if __name__ == "__main__":
    train_models()
