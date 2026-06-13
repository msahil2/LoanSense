"""
app.py
-------
Main Flask application for the Loan Approval Prediction System.

Routes:
    GET  /                  -> Home page (loan application form)
    POST /predict           -> Run prediction, store result, show result page
    GET  /history           -> Prediction history page
    GET  /admin             -> Admin dashboard (stats + charts)
    GET  /analytics          -> Data analytics dashboard
    GET  /model-comparison   -> Model comparison page
    GET  /report/<id>        -> Download PDF report for a prediction
    POST /admin/refresh-charts -> Regenerate all chart images
    GET  /api/health          -> Health check endpoint (for deployment platforms)

Environment variables used:
    PORT            - port to bind to (Render/Railway provide this)
    DATABASE_URL    - if set and starts with 'postgres', use PostgreSQL
    SECRET_KEY      - Flask secret key
    MODEL_DIR       - override path to the model directory (optional)
"""

import os
import json
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_file, abort
)
import joblib
import pandas as pd

from utils.preprocessing import build_feature_frame, FEATURE_COLUMNS
from utils.explainability import explain_prediction, compute_risk_score
from utils.pdf_report import generate_pdf_report
from utils import visualizations
from database import db

# ----------------------------------------------------------------------
# App configuration
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "model"))

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "loan-approval-secret-key-change-in-production")

# ----------------------------------------------------------------------
# Load trained model artifacts (dynamically, no hardcoded absolute paths)
# ----------------------------------------------------------------------
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance.json")

_model = None
_scaler = None
_metadata = None
_feature_importance = None


def load_model_artifacts():
    """Load model, scaler and metadata into module-level globals.

    Called once at startup. If artifacts are missing (e.g. first deploy
    before training has run), raises a clear error instructing the user
    to run train.py.
    """
    global _model, _scaler, _metadata, _feature_importance

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Please run 'python train.py' before starting the server."
        )

    _model = joblib.load(MODEL_PATH)

    if os.path.exists(SCALER_PATH):
        _scaler = joblib.load(SCALER_PATH)

    with open(METRICS_PATH) as f:
        _metadata = json.load(f)

    with open(FEATURE_IMPORTANCE_PATH) as f:
        _feature_importance = json.load(f)


load_model_artifacts()
db.init_db()

# Generate charts at startup if they don't exist yet (keeps first request fast)
IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")
_required_charts = [
    "approval_distribution.png", "income_distribution.png",
    "correlation_heatmap.png", "feature_importance.png", "model_comparison.png"
]
if not all(os.path.exists(os.path.join(IMAGES_DIR, c)) for c in _required_charts):
    try:
        visualizations.generate_all_charts()
    except Exception as chart_err:  # pragma: no cover - defensive
        app.logger.warning(f"Could not pre-generate charts: {chart_err}")


# ----------------------------------------------------------------------
# Form options (kept in one place so templates and validation stay in sync)
# ----------------------------------------------------------------------
FORM_OPTIONS = {
    "Gender": ["Male", "Female"],
    "Married": ["Yes", "No"],
    "Dependents": ["0", "1", "2", "3+"],
    "Education": ["Graduate", "Not Graduate"],
    "Self_Employed": ["Yes", "No"],
    "Property_Area": ["Urban", "Semiurban", "Rural"],
    "Loan_Amount_Term": [12, 36, 60, 84, 120, 180, 240, 300, 360],
    "Credit_History": [("1", "Good (1)"), ("0", "Poor / None (0)")],
}


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/")
def home():
    """Render the loan application form (home page)."""
    return render_template("index.html", options=FORM_OPTIONS, active_page="home")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle the loan application form submission:
      1. Validate input
      2. Build feature frame
      3. Run the best model to get prediction + probability
      4. Compute risk score & explainability factors
      5. Store the result in the database
      6. Render the prediction result page
    """
    try:
        form = request.form

        # ---- Basic validation ----
        required_fields = [
            "Gender", "Married", "Dependents", "Education", "Self_Employed",
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
            "Loan_Amount_Term", "Credit_History", "Property_Area",
        ]
        missing = [f for f in required_fields if not form.get(f)]
        if missing:
            flash(f"Missing required field(s): {', '.join(missing)}", "danger")
            return redirect(url_for("home"))

        try:
            applicant_income = float(form.get("ApplicantIncome"))
            coapplicant_income = float(form.get("CoapplicantIncome"))
            loan_amount = float(form.get("LoanAmount"))
            loan_amount_term = float(form.get("Loan_Amount_Term"))
            credit_history = float(form.get("Credit_History"))
        except ValueError:
            flash("Numeric fields must contain valid numbers.", "danger")
            return redirect(url_for("home"))

        if applicant_income < 0 or coapplicant_income < 0 or loan_amount <= 0:
            flash("Income and Loan Amount must be positive values.", "danger")
            return redirect(url_for("home"))

        input_dict = {
            "Gender": form.get("Gender"),
            "Married": form.get("Married"),
            "Dependents": form.get("Dependents"),
            "Education": form.get("Education"),
            "Self_Employed": form.get("Self_Employed"),
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_amount_term,
            "Credit_History": credit_history,
            "Property_Area": form.get("Property_Area"),
        }

        # ---- Feature engineering ----
        feature_row = build_feature_frame(input_dict)

        # ---- Prediction ----
        model_name = _metadata["best_model_name"]
        if _metadata.get("uses_scaler") and _scaler is not None:
            X = _scaler.transform(feature_row)
        else:
            X = feature_row

        pred_class = int(_model.predict(X)[0])

        if hasattr(_model, "predict_proba"):
            proba = _model.predict_proba(X)[0]
            approval_probability = float(proba[1]) * 100
        else:
            approval_probability = 100.0 if pred_class == 1 else 0.0

        prediction_label = "Approved" if pred_class == 1 else "Rejected"
        risk_score = compute_risk_score(approval_probability / 100.0)

        # ---- Explainability ----
        importance_dict = _feature_importance.get("best_model_importance", {})
        top_factors = explain_prediction(
            _model, model_name, _scaler, feature_row, importance_dict, top_n=3
        )

        # ---- Persist to database ----
        record = {
            "gender": input_dict["Gender"],
            "married": input_dict["Married"],
            "dependents": input_dict["Dependents"],
            "education": input_dict["Education"],
            "self_employed": input_dict["Self_Employed"],
            "applicant_income": applicant_income,
            "coapplicant_income": coapplicant_income,
            "loan_amount": loan_amount,
            "loan_amount_term": loan_amount_term,
            "credit_history": credit_history,
            "property_area": input_dict["Property_Area"],
            "prediction": prediction_label,
            "probability": round(approval_probability, 2),
            "risk_score": risk_score,
            "model_used": model_name,
            "top_factors": json.dumps(top_factors),
            "created_at": db.now_timestamp(),
        }
        new_id = db.insert_prediction(record)
        record["id"] = new_id

        return render_template(
            "result.html",
            record=record,
            top_factors=top_factors,
            active_page="home",
        )

    except Exception as e:
        app.logger.exception("Prediction failed")
        flash(f"An error occurred while processing your request: {str(e)}", "danger")
        return redirect(url_for("home"))


@app.route("/history")
def history():
    """Display the full / recent prediction history."""
    try:
        limit = request.args.get("limit", default=50, type=int)
        records = db.fetch_recent_predictions(limit=limit)
        return render_template("history.html", records=records, active_page="history")
    except Exception as e:
        app.logger.exception("Failed to load history")
        flash(f"Could not load prediction history: {str(e)}", "danger")
        return render_template("history.html", records=[], active_page="history")


@app.route("/admin")
def admin():
    """Admin dashboard: summary statistics + recent predictions + charts."""
    try:
        stats = db.get_summary_stats()
        recent = db.fetch_recent_predictions(limit=10)
        return render_template(
            "admin.html",
            stats=stats,
            recent=recent,
            metadata=_metadata,
            active_page="admin",
        )
    except Exception as e:
        app.logger.exception("Failed to load admin dashboard")
        flash(f"Could not load admin dashboard: {str(e)}", "danger")
        return render_template(
            "admin.html",
            stats={"total": 0, "approved": 0, "rejected": 0, "approval_rate": 0},
            recent=[],
            metadata=_metadata,
            active_page="admin",
        )


@app.route("/analytics")
def analytics():
    """Data analytics dashboard with all visualization charts."""
    return render_template("analytics.html", metadata=_metadata, active_page="analytics")


@app.route("/model-comparison")
def model_comparison():
    """Model comparison page showing accuracy / precision / recall / f1 for all models."""
    return render_template(
        "model_comparison.html",
        metadata=_metadata,
        feature_importance=_feature_importance,
        active_page="model-comparison",
    )


@app.route("/report/<int:pred_id>")
def report(pred_id):
    """Generate and download a PDF report for a specific prediction."""
    record = db.fetch_prediction_by_id(pred_id)
    if record is None:
        abort(404)

    # Reconstruct top_factors_list for the PDF
    try:
        record["top_factors_list"] = json.loads(record.get("top_factors") or "[]")
    except (TypeError, json.JSONDecodeError):
        record["top_factors_list"] = []

    pdf_bytes = generate_pdf_report(record)

    filename = f"loan_report_{pred_id}.pdf"
    return send_file(
        __import__("io").BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/admin/refresh-charts", methods=["POST"])
def refresh_charts():
    """Regenerate all dashboard charts (useful after retraining)."""
    try:
        visualizations.generate_all_charts()
        return jsonify({"status": "success", "message": "Charts refreshed successfully."})
    except Exception as e:
        app.logger.exception("Chart refresh failed")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/health")
def health_check():
    """Simple health check endpoint for deployment platforms."""
    return jsonify({
        "status": "ok",
        "model_loaded": _model is not None,
        "best_model": _metadata.get("best_model_name") if _metadata else None,
        "timestamp": datetime.now().isoformat(),
    })


# ----------------------------------------------------------------------
# Error handlers
# ----------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return render_template("404.html", active_page=""), 404


@app.errorhandler(500)
def server_error(e):
    app.logger.exception("Server error")
    return render_template("500.html", active_page=""), 500


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
