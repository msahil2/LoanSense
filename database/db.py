"""
db.py
------
Database access layer for the Loan Approval Prediction System.

- Uses SQLite by default (file-based, zero-config, great for local dev
  and for small deployments on Render/Railway).
- If the environment variable DATABASE_URL is set (e.g. a Postgres
  connection string provided by Render/Railway), the app will use
  PostgreSQL instead via psycopg2.

All SQL is written in a dialect-agnostic way as much as possible, with
small branches where SQLite and PostgreSQL syntax differ
(e.g. AUTOINCREMENT vs SERIAL, placeholder style ? vs %s).
"""

import os
import sqlite3
from datetime import datetime
from contextlib import contextmanager

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
USE_POSTGRES = DATABASE_URL.startswith("postgres")

# Local SQLite database path (no hardcoded absolute paths - relative to
# this file's directory so it works regardless of the working directory).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_PATH = os.environ.get(
    "SQLITE_PATH", os.path.join(BASE_DIR, "loan_database.db")
)

if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras


def _get_connection():
    """Return a new database connection (Postgres or SQLite)."""
    if USE_POSTGRES:
        # Render/Railway sometimes provide URLs starting with postgres://
        # which psycopg2 handles fine, but normalize just in case.
        conn = psycopg2.connect(DATABASE_URL, sslmode=os.environ.get("DB_SSLMODE", "require"))
        return conn
    else:
        conn = sqlite3.connect(SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        return conn


@contextmanager
def get_db():
    """Context manager that yields a connection and closes it afterwards."""
    conn = _get_connection()
    try:
        yield conn
    finally:
        conn.close()


def _placeholder():
    """Return the correct SQL parameter placeholder for the active DB."""
    return "%s" if USE_POSTGRES else "?"


def init_db():
    """
    Create the predictions table if it does not already exist.
    Safe to call on every application startup.
    """
    with get_db() as conn:
        cur = conn.cursor()
        if USE_POSTGRES:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    gender TEXT,
                    married TEXT,
                    dependents TEXT,
                    education TEXT,
                    self_employed TEXT,
                    applicant_income REAL,
                    coapplicant_income REAL,
                    loan_amount REAL,
                    loan_amount_term REAL,
                    credit_history REAL,
                    property_area TEXT,
                    prediction TEXT,
                    probability REAL,
                    risk_score REAL,
                    model_used TEXT,
                    top_factors TEXT,
                    created_at TEXT
                );
                """
            )
        else:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gender TEXT,
                    married TEXT,
                    dependents TEXT,
                    education TEXT,
                    self_employed TEXT,
                    applicant_income REAL,
                    coapplicant_income REAL,
                    loan_amount REAL,
                    loan_amount_term REAL,
                    credit_history REAL,
                    property_area TEXT,
                    prediction TEXT,
                    probability REAL,
                    risk_score REAL,
                    model_used TEXT,
                    top_factors TEXT,
                    created_at TEXT
                );
                """
            )
        conn.commit()


def insert_prediction(record: dict) -> int:
    """
    Insert a new prediction record into the database.

    Parameters
    ----------
    record : dict
        Dictionary containing all the column values.

    Returns
    -------
    int
        The id of the newly inserted row.
    """
    ph = _placeholder()
    columns = [
        "gender", "married", "dependents", "education", "self_employed",
        "applicant_income", "coapplicant_income", "loan_amount",
        "loan_amount_term", "credit_history", "property_area",
        "prediction", "probability", "risk_score", "model_used",
        "top_factors", "created_at",
    ]
    values = [record.get(col) for col in columns]
    placeholders = ", ".join([ph] * len(columns))
    col_str = ", ".join(columns)

    with get_db() as conn:
        cur = conn.cursor()
        if USE_POSTGRES:
            cur.execute(
                f"INSERT INTO predictions ({col_str}) VALUES ({placeholders}) RETURNING id;",
                values,
            )
            new_id = cur.fetchone()[0]
        else:
            cur.execute(
                f"INSERT INTO predictions ({col_str}) VALUES ({placeholders});",
                values,
            )
            new_id = cur.lastrowid
        conn.commit()
        return new_id


def fetch_recent_predictions(limit: int = 10):
    """Return the most recent N predictions as a list of dicts."""
    ph = _placeholder()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT * FROM predictions ORDER BY id DESC LIMIT {ph};",
            (limit,),
        )
        rows = cur.fetchall()
        return [_row_to_dict(cur, row) for row in rows]


def fetch_all_predictions():
    """Return every prediction record (used for analytics & charts)."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM predictions ORDER BY id ASC;")
        rows = cur.fetchall()
        return [_row_to_dict(cur, row) for row in rows]


def fetch_prediction_by_id(pred_id: int):
    """Return a single prediction record by its id (for PDF report generation)."""
    ph = _placeholder()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM predictions WHERE id = {ph};", (pred_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_dict(cur, row)


def get_summary_stats():
    """Compute aggregate statistics for the admin dashboard."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM predictions;")
        total = cur.fetchone()[0] or 0

        cur.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Approved';")
        approved = cur.fetchone()[0] or 0

        cur.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Rejected';")
        rejected = cur.fetchone()[0] or 0

    approval_rate = (approved / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "approved": approved,
        "rejected": rejected,
        "approval_rate": round(approval_rate, 2),
    }


def _row_to_dict(cur, row):
    """Convert a DB row (sqlite3.Row or psycopg2 tuple) to a dict."""
    if USE_POSTGRES:
        colnames = [desc[0] for desc in cur.description]
        return dict(zip(colnames, row))
    else:
        return dict(row)


def now_timestamp() -> str:
    """Return a human-readable current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
