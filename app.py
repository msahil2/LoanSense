"""
app.py
-------
Main Flask application for the Loan Approval Prediction System.

Authentication:
    - Users register/login with a username + password.
    - Each user can only see/download their OWN predictions.
    - Admin accounts (is_admin=1) can see ALL predictions via /admin.
    - An initial admin account is auto-created from ADMIN_USERNAME /
      ADMIN_EMAIL / ADMIN_PASSWORD environment variables (if provided
      and no admin exists yet).

Routes:
    GET  /                  -> Home page (loan application form) [login required]
    POST /predict           -> Run prediction, store result, show result page [login required]
    GET  /history           -> Prediction history (own predictions, or all for admin)
    GET  /admin             -> Admin dashboard (ALL users' stats + charts) [admin only]
    GET  /analytics          -> Data analytics dashboard
    GET  /model-comparison   -> Model comparison page
    GET  /report/<id>        -> Download PDF report for a prediction (owner or admin only)
    POST /admin/refresh-charts -> Regenerate all chart images [admin only]
    GET  /register / /login / /logout -> Authentication
    GET  /api/health          -> Health check endpoint (for deployment platforms)

Environment variables used:
    PORT            - port to bind to (Render/Railway provide this)
    DATABASE_URL    - if set and starts with 'postgres', use PostgreSQL
    SECRET_KEY      - Flask secret key
    MODEL_DIR       - override path to the model directory (optional)
    ADMIN_USERNAME, ADMIN_EMAIL, ADMIN_PASSWORD - optional, auto-creates an admin account
"""

import os
import json
from functools import wraps
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_file, abort, session, g
)
from werkzeug.security import generate_password_hash, check_password_hash
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


def bootstrap_admin_account():
    """
    Auto-create an admin account from environment variables if provided
    and no account with that username exists yet. This makes it easy to
    get a working admin login immediately after deployment:

        ADMIN_USERNAME=admin
        ADMIN_EMAIL=admin@example.com
        ADMIN_PASSWORD=ChangeMe123!
    """
    admin_username = os.environ.get("ADMIN_USERNAME")
    admin_password = os.environ.get("ADMIN_PASSWORD")
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@example.com")

    if not admin_username or not admin_password:
        return

    existing = db.get_user_by_username(admin_username)
    if existing is None:
        db.create_user(
            username=admin_username,
            email=admin_email,
            password_hash=generate_password_hash(admin_password),
            is_admin=True,
        )
        app.logger.info(f"Created admin account '{admin_username}' from environment variables.")
    elif not existing.get("is_admin"):
        # Promote existing user to admin if env vars say so
        ph = "%s" if db.USE_POSTGRES else "?"
        with db.get_db() as conn:
            cur = conn.cursor()
            cur.execute(f"UPDATE users SET is_admin = 1 WHERE username = {ph};", (admin_username,))
            conn.commit()


bootstrap_admin_account()

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
# Authentication helpers
# ----------------------------------------------------------------------
@app.before_request
def load_logged_in_user():
    """Attach the currently logged-in user (if any) to flask.g."""
    user_id = session.get("user_id")
    if user_id is None:
        g.user = None
    else:
        g.user = db.get_user_by_id(user_id)


@app.context_processor
def inject_user():
    """Make the current user available in all templates as `current_user`."""
    return {"current_user": g.get("user")}


def login_required(view):
    """Redirect to the login page if the user is not authenticated."""
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if g.user is None:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped_view


def admin_required(view):
    """Restrict a view to admin accounts only."""
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if g.user is None:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login", next=request.path))
        if not g.user.get("is_admin"):
            flash("You do not have permission to access that page.", "danger")
            return redirect(url_for("home"))
        return view(*args, **kwargs)
    return wrapped_view


# ----------------------------------------------------------------------
# Authentication routes
# ----------------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    """User self-registration. New accounts are regular (non-admin) users."""
    if g.user is not None:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "danger")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))

        if db.username_or_email_exists(username, email):
            flash("Username or email is already registered.", "danger")
            return redirect(url_for("register"))

        db.create_user(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            is_admin=False,
        )
        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html", active_page="register")


@app.route("/login", methods=["GET", "POST"])
def login():
    """User login."""
    if g.user is not None:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        user = db.get_user_by_username(username)
        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Invalid username or password.", "danger")
            return redirect(url_for("login"))

        session.clear()
        session["user_id"] = user["id"]
        flash(f"Welcome back, {user['username']}!", "success")

        next_url = request.args.get("next")
        return redirect(next_url or url_for("home"))

    return render_template("login.html", active_page="login")


@app.route("/logout")
def logout():
    """Log the current user out."""
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


# ----------------------------------------------------------------------
# Application Routes
# ----------------------------------------------------------------------
@app.route("/")
@login_required
def home():
    """Render the loan application form (home page)."""
    return render_template("index.html", options=FORM_OPTIONS, active_page="home")


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    """
    Handle the loan application form submission:
      1. Validate input
      2. Build feature frame
      3. Run the best model to get prediction + probability
      4. Compute risk score & explainability factors
      5. Store the result in the database (linked to the current user)
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

        # ---- Persist to database (linked to current user) ----
        record = {
            "user_id": g.user["id"],
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
@login_required
def history():
    """
    Display prediction history.

    - Regular users see ONLY their own predictions.
    - Admin users see ALL predictions across every account.
    """
    try:
        limit = request.args.get("limit", default=50, type=int)
        if g.user.get("is_admin"):
            records = db.fetch_recent_predictions(limit=limit, user_id=None)
        else:
            records = db.fetch_recent_predictions(limit=limit, user_id=g.user["id"])
        return render_template("history.html", records=records, active_page="history")
    except Exception as e:
        app.logger.exception("Failed to load history")
        flash(f"Could not load prediction history: {str(e)}", "danger")
        return render_template("history.html", records=[], active_page="history")


@app.route("/admin")
@admin_required
def admin():
    """Admin dashboard: summary statistics + recent predictions (ALL users) + charts."""
    try:
        stats = db.get_summary_stats(user_id=None)
        recent = db.fetch_recent_predictions(limit=10, user_id=None)
        all_records = db.fetch_all_predictions()
        live_charts = visualizations.generate_live_charts(all_records)
        return render_template(
            "admin.html",
            stats=stats,
            recent=recent,
            metadata=_metadata,
            live_charts=live_charts,
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
            live_charts={},
            active_page="admin",
        )


@app.route("/analytics")
@login_required
def analytics():
    """Data analytics dashboard with all visualization charts."""
    return render_template("analytics.html", metadata=_metadata, active_page="analytics")


@app.route("/model-comparison")
@login_required
def model_comparison():
    """Model comparison page showing accuracy / precision / recall / f1 for all models."""
    return render_template(
        "model_comparison.html",
        metadata=_metadata,
        feature_importance=_feature_importance,
        active_page="model-comparison",
    )


@app.route("/report/<int:pred_id>")
@login_required
def report(pred_id):
    """
    Generate and download a PDF report for a specific prediction.

    Only the owner of the prediction or an admin may download it.
    """
    record = db.fetch_prediction_by_id(pred_id)
    if record is None:
        abort(404)

    is_owner = record.get("user_id") == g.user["id"]
    if not is_owner and not g.user.get("is_admin"):
        flash("You do not have permission to view that report.", "danger")
        return redirect(url_for("history"))

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
@admin_required
def refresh_charts():
    """Regenerate all dashboard charts (useful after retraining). Admin only."""
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