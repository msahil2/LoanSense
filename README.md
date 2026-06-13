# 🏦 Loan Approval Prediction System

A complete, production-ready end-to-end Machine Learning web application that predicts
whether a loan application should be **Approved** or **Rejected**, with a professional
banking-style interface, an admin dashboard, data analytics, model comparison, PDF
reports, and an **Explainable AI** section that shows the top factors behind every
decision.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Local Setup](#-local-setup)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Deployment](#-deployment)
  - [Render Deployment](#render-deployment)
  - [Railway Deployment](#railway-deployment)
  - [AWS Deployment](#aws-deployment)
- [Environment Variables](#-environment-variables)
- [API Endpoints](#-api-endpoints)
- [Screenshots Section](#-screenshots)
- [Security Notes](#-security-notes)
- [License](#-license)

---

## 🚀 Project Overview

This project simulates a real-world **bank loan approval system** powered by Machine
Learning. Applicants fill out a form with their personal, financial, and credit
information. The system then:

1. Cleans and engineers the input data exactly as it was processed during training.
2. Runs the **best-performing model** (automatically selected during training between
   Logistic Regression, Decision Tree, and Random Forest).
3. Returns an **Approved/Rejected** decision, an **approval probability**, and a
   **risk score**.
4. Explains the decision using the **top 3 contributing factors** (Explainable AI).
5. Stores every prediction permanently in a database.
6. Allows downloading a **PDF report** of the prediction.
7. Provides an **Admin Dashboard** and **Analytics Dashboard** with charts.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Loan Approval Prediction** | Real-time prediction via a clean banking-style form |
| **Explainable AI** | Top 3 factors behind every decision (Credit History, Income, Loan Amount, etc.) |
| **Admin Dashboard** | Total predictions, approval rate, recent activity, charts |
| **Prediction History** | Full searchable log of every prediction made |
| **Model Comparison** | Side-by-side metrics for Logistic Regression, Decision Tree, Random Forest |
| **Feature Importance Visualization** | Bar chart of model feature importances |
| **Data Analytics Dashboard** | Approval distribution, income histogram, correlation heatmap, etc. |
| **PDF Report Generation** | Downloadable PDF with applicant details and explanation |
| **SQLite / PostgreSQL** | Works locally with SQLite, switches to PostgreSQL via `DATABASE_URL` |
| **Deployment Ready** | Procfile, runtime.txt, render.yaml, Gunicorn, env-based config |

---

## 🛠 Tech Stack

- **Backend:** Python 3.11, Flask, Gunicorn
- **Machine Learning:** Scikit-learn, Pandas, NumPy, Joblib
- **Database:** SQLite (local) / PostgreSQL (production, via `psycopg2`)
- **PDF Generation:** ReportLab
- **Visualizations:** Matplotlib
- **Frontend:** HTML5, CSS3, Bootstrap 5, Bootstrap Icons, Vanilla JavaScript

---

## 📁 Project Structure

```
loan-approval-system/
│
├── app.py                     # Main Flask application (routes & logic)
├── train.py                   # ML training pipeline (run before first start)
├── requirements.txt           # Python dependencies
├── Procfile                    # Gunicorn start command for Render/Railway
├── runtime.txt                 # Python version pin
├── render.yaml                  # Render Blueprint configuration
├── .env.example                 # Example environment variables
├── .gitignore
│
├── dataset/
│   ├── generate_dataset.py     # Generates the synthetic loan dataset
│   └── loan_data.csv           # Training dataset (1200 rows)
│
├── model/
│   ├── best_model.pkl           # Best trained model (Joblib)
│   ├── scaler.pkl                # StandardScaler (for Logistic Regression)
│   ├── all_models.pkl            # All 3 trained models
│   ├── metrics.json               # Evaluation metrics for all models
│   └── feature_importance.json     # Feature importance data
│
├── utils/
│   ├── preprocessing.py          # Shared cleaning / encoding / feature engineering
│   ├── explainability.py          # "SHAP-style" top-factor explanation logic
│   ├── visualizations.py           # Matplotlib chart generation
│   └── pdf_report.py                # PDF report builder (ReportLab)
│
├── database/
│   └── db.py                       # SQLite / PostgreSQL data access layer
│
├── templates/
│   ├── base.html                   # Shared layout (navbar, footer, toasts)
│   ├── index.html                  # Home page / loan application form
│   ├── result.html                 # Prediction result + Explainable AI
│   ├── history.html                # Prediction history table
│   ├── admin.html                  # Admin dashboard
│   ├── analytics.html              # Data analytics dashboard
│   ├── model_comparison.html       # Model comparison page
│   ├── 404.html
│   └── 500.html
│
└── static/
    ├── css/style.css               # Banking-style theme
    ├── js/main.js                  # Loading spinner, toasts, validation
    └── images/                     # Auto-generated chart images
```

---

## 💻 Installation

### Prerequisites
- Python 3.11+
- `pip`

### Clone & Install

```bash
git clone <your-repo-url>
cd loan-approval-system
pip install -r requirements.txt
```

---

## 🖥 Local Setup

### 1. Generate the dataset (already included, but you can regenerate it)

```bash
python dataset/generate_dataset.py
```

### 2. Train the models

This runs the full ML pipeline: cleaning → encoding → feature engineering →
train/test split → training Logistic Regression, Decision Tree & Random Forest →
evaluation → automatic best-model selection → saving artifacts with Joblib.

```bash
python train.py
```

You should see output similar to:

```
LOAN APPROVAL PREDICTION SYSTEM - MODEL TRAINING
============================================================
[1/6] Loading dataset...
[2/6] Cleaning data, encoding labels, engineering features...
[3/6] Splitting data into train and test sets (80/20)...
[4/6] Training models: Logistic Regression, Decision Tree, Random Forest...
      Logistic Regression    | Accuracy:  88.33% | F1:  92.86%
      Decision Tree          | Accuracy:  85.83% | F1:  91.30%
      Random Forest          | Accuracy:  87.50% | F1:  92.13%
[5/6] Selecting best model based on F1-score...
      >> Best model: Logistic Regression (F1: 92.86%, Accuracy: 88.33%)
[6/6] Computing feature importances & saving artifacts...
```

### 3. Run the application

```bash
python app.py
```

The app will be available at **http://localhost:5000**

### 4. (Optional) Run with Gunicorn (production-style locally)

```bash
gunicorn app:app --bind 0.0.0.0:5000 --workers 2
```

---

## 🧠 Machine Learning Pipeline

The pipeline (in `train.py` and `utils/preprocessing.py`) performs:

1. **Data Cleaning** — fills missing categorical values with mode, numeric values with
   median, and `Credit_History` with 1 (the most common, "good history" case).
2. **Label Encoding** — converts categorical text fields (Gender, Married, Education,
   etc.) into numeric values using fixed mappings.
3. **Feature Engineering** —
   - `TotalIncome = ApplicantIncome + CoapplicantIncome`
   - `IncomeLoanRatio = TotalIncome / (LoanAmount * 1000)`
4. **Train/Test Split** — 80/20 stratified split.
5. **Model Training** — Logistic Regression, Decision Tree, Random Forest.
6. **Evaluation** — Accuracy, Precision, Recall, F1-score, Confusion Matrix.
7. **Automatic Best Model Selection** — selects the model with the highest F1-score
   (tie-broken by accuracy).
8. **Persistence** — saves the best model, scaler, all models, metrics, and feature
   importances using **Joblib** / JSON into `/model`.

### Explainable AI

For every prediction, `utils/explainability.py`:

- For **Logistic Regression**, computes each feature's contribution as
  `coefficient x scaled_value`.
- For **tree-based models**, approximates a local contribution using the global
  feature importance weighted by how far the applicant's value deviates from a
  typical reference applicant.
- Ranks features by absolute contribution and returns the **top 3**, along with a
  human-readable explanation (e.g. *"Credit History: a clean credit history record,
  which increased the chance of approval."*).

---

## ☁️ Deployment

The app is designed to run with **zero code changes** on Render, Railway, or AWS.
It reads all configuration from environment variables and uses **relative paths**
throughout (`os.path.dirname(...)`), so it works regardless of the deployment
directory.

> ⚠️ **Important:** The `/model` folder (containing `best_model.pkl`, `scaler.pkl`,
> `metrics.json`, `feature_importance.json`) is committed to the repository so the
> app works immediately after deployment **without** needing to run `train.py` on
> the server. You can always retrain locally and re-commit the updated model files,
> or run `python train.py` manually on the server's shell if your platform allows it.

### Render Deployment

1. Push this project to a GitHub repository.
2. Go to [Render Dashboard](https://dashboard.render.com) → **New +** → **Web Service**.
3. Connect your GitHub repo.
4. Configure:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt && python train.py`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
5. Add environment variables (see [Environment Variables](#-environment-variables)).
6. (Optional) Create a **PostgreSQL** database on Render and set `DATABASE_URL`
   to its connection string for persistent storage across deploys.
7. Click **Create Web Service**. Render will build and deploy automatically.

Alternatively, use the included `render.yaml` Blueprint:
1. Go to **New +** → **Blueprint**.
2. Select your repo — Render will detect `render.yaml` and configure everything
   automatically.

### Railway Deployment

1. Push this project to a GitHub repository.
2. Go to [Railway](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
3. Railway auto-detects Python. Set the following in **Settings → Deploy**:
   - **Build Command:** `pip install -r requirements.txt && python train.py`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
4. Go to **Variables** and add the environment variables from
   [Environment Variables](#-environment-variables).
5. (Optional) Add a **PostgreSQL** plugin from Railway's marketplace — Railway will
   automatically inject `DATABASE_URL` into your service's environment.
6. Deploy. Railway will provide a public URL automatically.

### AWS Deployment

You can deploy this app on **AWS Elastic Beanstalk**, **AWS EC2**, or **AWS App Runner**.
Below are instructions for **Elastic Beanstalk** (simplest):

1. Install the EB CLI: `pip install awsebcli`
2. Initialize:
   ```bash
   eb init -p python-3.11 loan-approval-system
   ```
3. Create an environment:
   ```bash
   eb create loan-approval-env
   ```
4. Set environment variables:
   ```bash
   eb setenv SECRET_KEY=your-secret-key FLASK_DEBUG=false
   # Optional, if using RDS PostgreSQL:
   eb setenv DATABASE_URL=postgresql://user:pass@your-rds-endpoint:5432/dbname
   ```
5. Ensure `requirements.txt` and `Procfile` are present (already included —
   Elastic Beanstalk's Python platform uses Gunicorn automatically via the Procfile).
6. Deploy:
   ```bash
   eb deploy
   ```
7. Open the app:
   ```bash
   eb open
   ```

For **EC2** (manual deployment):
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv
git clone <your-repo-url> && cd loan-approval-system
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python train.py
gunicorn app:app --bind 0.0.0.0:8000 --workers 2 --daemon
```
Then configure an Nginx reverse proxy and a systemd service for persistence, and
open port 80/443 in your EC2 Security Group.

---

## 🔑 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `SECRET_KEY` | Recommended | dev key | Flask session/flash signing key |
| `PORT` | Auto-set by platform | `5000` | Port the app binds to |
| `FLASK_DEBUG` | No | `false` | Enable Flask debug mode (dev only) |
| `DATABASE_URL` | No | *(empty)* | PostgreSQL connection string. If empty, SQLite is used |
| `DB_SSLMODE` | No | `require` | SSL mode for PostgreSQL connections |
| `MODEL_DIR` | No | `./model` | Override path to model artifacts |

Copy `.env.example` to `.env` for local development:
```bash
cp .env.example .env
```

---

## 🔐 Authentication & Access Control

- **Register/Login required** for all prediction features (`/`, `/predict`, `/history`,
  `/analytics`, `/model-comparison`).
- **Regular users** see and download PDF reports for **only their own** predictions
  (`/history` shows "My Predictions").
- **Admin accounts** (`is_admin = 1`) can access `/admin` (system-wide stats + charts)
  and `/history` shows **all users' predictions** with a User ID column. Admins can
  also download any user's PDF report.

### Creating the Admin Account

Set these environment variables before first run (locally in `.env`, or in your
Render/Railway dashboard) - the app auto-creates (or promotes) this account on startup:

```
ADMIN_USERNAME=admin
ADMIN_EMAIL=admin@yourbank.com
ADMIN_PASSWORD=ChangeMe123!
```

Then simply log in with that username/password - you'll see the **Admin** link in the
navbar and an "Admin" badge next to your name. All other users who register via
`/register` are regular (non-admin) accounts by default.

---

## 🔌 API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET/POST` | `/register` | Create a new user account |
| `GET/POST` | `/login` | User login |
| `GET` | `/logout` | Log out |
| `GET` | `/` | Home page — loan application form (login required) |
| `POST` | `/predict` | Submit application, get prediction result page (login required) |
| `GET` | `/history` | Own predictions (or all, for admin) |
| `GET` | `/admin` | Admin dashboard - ALL users (admin only) |
| `GET` | `/analytics` | Data analytics dashboard |
| `GET` | `/model-comparison` | Model comparison page |
| `GET` | `/report/<id>` | Download PDF report (owner or admin only) |
| `POST` | `/admin/refresh-charts` | Regenerate all dashboard charts (admin only) |
| `GET` | `/api/health` | Health check (JSON) |

---

## 📸 Screenshots

> Add your application screenshots here after deployment.

- `Home Page (Loan Application Form)`
- `Prediction Result + Explainable AI`
- `Admin Dashboard`
- `Analytics Dashboard`
- `Model Comparison`

---

## 🔒 Security Notes

- Set a strong, random `SECRET_KEY` in production (never use the default).
- Use `DATABASE_URL` with PostgreSQL + SSL for production data persistence — SQLite
  files on platforms like Render/Railway are **ephemeral** and may be lost on
  redeploys.
- The application validates and sanitizes all numeric form inputs server-side.
- Never commit your `.env` file (already excluded via `.gitignore`).
- Run `FLASK_DEBUG=false` in production to avoid leaking stack traces.

---

## 📄 License

This project is provided for educational and demonstration purposes. Predictions are
generated by a Machine Learning model trained on synthetic data and do **not**
constitute real financial or lending advice.
