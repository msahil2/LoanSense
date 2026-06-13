"""
visualizations.py
-------------------
Generates all chart images used in the Data Analytics Dashboard and
Admin Dashboard. Charts are rendered server-side with Matplotlib (Agg
backend, so no display is required) and saved as PNG files inside
static/images/ so they can be served directly by Flask / any static
file host after deployment.

Charts generated:
    1. Loan Approval Distribution (pie chart)
    2. Income Distribution (histogram)
    3. Correlation Heatmap
    4. Feature Importance Graph (bar chart)
    5. Model Accuracy Comparison (bar chart)
"""

import os
import json

import matplotlib
matplotlib.use("Agg")  # Headless backend - required for servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.preprocessing import clean_and_engineer, FEATURE_COLUMNS, FEATURE_LABELS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "loan_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")

os.makedirs(IMAGES_DIR, exist_ok=True)

# Banking-style color palette
PRIMARY_COLOR = "#0B3D91"
ACCENT_COLOR = "#1A8FE3"
SUCCESS_COLOR = "#1FAA59"
DANGER_COLOR = "#E0433B"
NEUTRAL_COLOR = "#7C8A99"
PALETTE = [PRIMARY_COLOR, ACCENT_COLOR, SUCCESS_COLOR, DANGER_COLOR, "#F4A623", NEUTRAL_COLOR]


def _save_fig(fig, filename):
    path = os.path.join(IMAGES_DIR, filename)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"images/{filename}"


def generate_approval_distribution_chart():
    """Pie chart of Approved vs Rejected loans from the training dataset."""
    df = pd.read_csv(DATASET_PATH)
    counts = df["Loan_Status"].value_counts()
    labels = ["Approved" if k == "Y" else "Rejected" for k in counts.index]
    colors = [SUCCESS_COLOR if k == "Y" else DANGER_COLOR for k in counts.index]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11},
    )
    ax.set_title("Loan Approval Distribution", fontsize=13, fontweight="bold", color=PRIMARY_COLOR)
    return _save_fig(fig, "approval_distribution.png")


def generate_income_distribution_chart():
    """Histogram of applicant income."""
    df = pd.read_csv(DATASET_PATH)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.hist(df["ApplicantIncome"].dropna(), bins=30, color=ACCENT_COLOR, edgecolor="white")
    ax.set_title("Applicant Income Distribution", fontsize=13, fontweight="bold", color=PRIMARY_COLOR)
    ax.set_xlabel("Applicant Income")
    ax.set_ylabel("Number of Applicants")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    return _save_fig(fig, "income_distribution.png")


def generate_correlation_heatmap():
    """Correlation heatmap of all numeric/engineered features + target."""
    df = pd.read_csv(DATASET_PATH)
    cleaned = clean_and_engineer(df, is_training=True)

    cols = FEATURE_COLUMNS + ["Loan_Status"]
    corr = cleaned[cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)

    labels = [FEATURE_LABELS.get(c, c) for c in cols[:-1]] + ["Loan Status"]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")

    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold", color=PRIMARY_COLOR)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save_fig(fig, "correlation_heatmap.png")


def generate_feature_importance_chart():
    """Bar chart of feature importances from the best model."""
    fi_path = os.path.join(MODEL_DIR, "feature_importance.json")
    with open(fi_path) as f:
        fi_data = json.load(f)

    importance = fi_data["best_model_importance"]
    sorted_items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    labels = [FEATURE_LABELS.get(k, k) for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(7, 6))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=PRIMARY_COLOR)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Relative Importance")
    ax.set_title(
        f"Feature Importance ({fi_data['best_model_name']})",
        fontsize=13, fontweight="bold", color=PRIMARY_COLOR
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    return _save_fig(fig, "feature_importance.png")


def generate_model_comparison_chart():
    """Bar chart comparing accuracy / F1 of all trained models."""
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path) as f:
        metadata = json.load(f)

    results = metadata["results"]
    model_names = list(results.keys())
    accuracy = [results[m]["accuracy"] for m in model_names]
    f1 = [results[m]["f1_score"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, accuracy, width, label="Accuracy (%)", color=PRIMARY_COLOR)
    ax.bar(x + width / 2, f1, width, label="F1-Score (%)", color=ACCENT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold", color=PRIMARY_COLOR)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    best_name = metadata["best_model_name"]
    if best_name in model_names:
        idx = model_names.index(best_name)
        ax.annotate("Best Model", xy=(idx, accuracy[idx]),
                     xytext=(idx, accuracy[idx] + 8),
                     ha="center", fontsize=9, fontweight="bold", color=SUCCESS_COLOR,
                     arrowprops=dict(arrowstyle="->", color=SUCCESS_COLOR))

    return _save_fig(fig, "model_comparison.png")


def generate_all_charts():
    """Generate every chart at once (used at startup and via /admin/refresh)."""
    paths = {}
    paths["approval_distribution"] = generate_approval_distribution_chart()
    paths["income_distribution"] = generate_income_distribution_chart()
    paths["correlation_heatmap"] = generate_correlation_heatmap()
    paths["feature_importance"] = generate_feature_importance_chart()
    paths["model_comparison"] = generate_model_comparison_chart()
    return paths


def generate_live_charts(records):
    """
    Generate charts based on ACTUAL predictions stored in the database
    (i.e. real usage data from app users), as opposed to the static
    training dataset. Used on the Admin Dashboard.

    Parameters
    ----------
    records : list of dict
        Rows from the `predictions` table (as returned by
        database.db.fetch_all_predictions()).

    Returns
    -------
    dict
        Mapping of chart name -> relative path under static/, or None
        for charts that could not be generated (e.g. no data yet).
    """
    paths = {"live_approval_distribution": None, "live_income_distribution": None,
             "live_risk_distribution": None, "live_approval_trend": None}

    if not records:
        return paths

    df = pd.DataFrame(records)

    # ---- 1. Live Approval vs Rejected distribution ----
    if "prediction" in df.columns and not df["prediction"].dropna().empty:
        counts = df["prediction"].value_counts()
        colors = [SUCCESS_COLOR if k == "Approved" else DANGER_COLOR for k in counts.index]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(
            counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=colors, startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
            textprops={"fontsize": 11},
        )
        ax.set_title("Live Approval Distribution\n(Actual User Predictions)",
                      fontsize=12, fontweight="bold", color=PRIMARY_COLOR)
        paths["live_approval_distribution"] = _save_fig(fig, "live_approval_distribution.png")

    # ---- 2. Live Applicant Income distribution ----
    if "applicant_income" in df.columns and not df["applicant_income"].dropna().empty:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        bins = min(20, max(5, df["applicant_income"].nunique()))
        ax.hist(df["applicant_income"].dropna(), bins=bins, color=ACCENT_COLOR, edgecolor="white")
        ax.set_title("Live Applicant Income Distribution\n(Actual User Submissions)",
                      fontsize=12, fontweight="bold", color=PRIMARY_COLOR)
        ax.set_xlabel("Applicant Income")
        ax.set_ylabel("Number of Applications")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        paths["live_income_distribution"] = _save_fig(fig, "live_income_distribution.png")

    # ---- 3. Live Risk Score distribution ----
    if "risk_score" in df.columns and not df["risk_score"].dropna().empty:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        bins = min(20, max(5, df["risk_score"].nunique()))
        ax.hist(df["risk_score"].dropna(), bins=bins, color=PRIMARY_COLOR, edgecolor="white")
        ax.set_title("Live Risk Score Distribution\n(Actual User Predictions)",
                      fontsize=12, fontweight="bold", color=PRIMARY_COLOR)
        ax.set_xlabel("Risk Score (0 = Low Risk, 100 = High Risk)")
        ax.set_ylabel("Number of Applications")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        paths["live_risk_distribution"] = _save_fig(fig, "live_risk_distribution.png")

    # ---- 4. Approval rate over time (by date) ----
    if "created_at" in df.columns and "prediction" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["created_at"]).dt.date
            grouped = df.groupby("date")["prediction"].apply(
                lambda s: (s == "Approved").sum() / len(s) * 100
            )
            if len(grouped) >= 1:
                fig, ax = plt.subplots(figsize=(7, 4.2))
                ax.plot(grouped.index.astype(str), grouped.values,
                        marker="o", color=PRIMARY_COLOR, linewidth=2)
                ax.set_title("Daily Approval Rate\n(Actual User Predictions)",
                              fontsize=12, fontweight="bold", color=PRIMARY_COLOR)
                ax.set_xlabel("Date")
                ax.set_ylabel("Approval Rate (%)")
                ax.set_ylim(0, 100)
                ax.grid(axis="y", linestyle="--", alpha=0.4)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
                paths["live_approval_trend"] = _save_fig(fig, "live_approval_trend.png")
        except Exception:
            pass

    return paths


if __name__ == "__main__":
    generated = generate_all_charts()
    for name, path in generated.items():
        print(f"{name}: static/{path}")
