"""Helpers shared by training, scoring and the Streamlit app."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, CATEGORICAL_FEATURES, RISK_BANDS
from src.features import FEATURE_LABELS, FEATURES

MODEL_PATH = ARTIFACTS_DIR / "churn_model.joblib"


def prepare_X(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Select model features and lock categorical levels to those seen in training."""
    X = df[FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = pd.Categorical(X[c].astype(str), categories=categories[c])
    return X


def load_bundle(path=MODEL_PATH) -> dict:
    return joblib.load(path)


def risk_band(p: float) -> str:
    for lo, hi, name in RISK_BANDS:
        if lo <= p < hi:
            return name
    return "High"


def contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Per-row SHAP values (log-odds) from LightGBM's built-in TreeSHAP."""
    contrib = model.booster_.predict(X, pred_contrib=True)
    return pd.DataFrame(contrib[:, :-1], columns=X.columns, index=X.index)


def _fmt(feature: str, value) -> str:
    """Turn a feature value into a short, readable phrase for business users."""
    pct = lambda v: f"{abs(v):.0%}"  # noqa: E731
    if feature == "months_to_renewal":
        n = int(value)
        return f"Contract renews in {n} month{'s' if n != 1 else ''}"
    if feature == "port_out_requests_3m":
        return f"{int(value)} port-out requests in last 3 months"
    if feature == "lines_change_pct_3m":
        return (f"Lines down {pct(value)} in last 3 months" if value < 0
                else f"Lines up {pct(value)} in last 3 months")
    if feature == "lines_disconnected_3m":
        return f"{int(value)} lines disconnected in last 3 months"
    if feature in ("mrr_change_pct", "data_usage_change_pct", "voice_usage_change_pct"):
        what = {"mrr_change_pct": "Monthly revenue", "data_usage_change_pct": "Data usage",
                "voice_usage_change_pct": "Voice usage"}[feature]
        return f"{what} {'down' if value < 0 else 'up'} {pct(value)} vs previous quarter"
    if feature == "network_tickets_3m":
        return f"{int(value)} network/data-speed complaints in last 3 months"
    if feature == "billing_disputes_3m":
        return f"{int(value)} billing disputes in last 3 months"
    if feature == "sla_breaches_3m":
        return f"{int(value)} SLA breaches in last 3 months"
    if feature == "tickets_3m":
        return f"{int(value)} support tickets in last 3 months"
    if feature == "open_tickets":
        return f"{int(value)} unresolved support tickets"
    if feature == "avg_resolution_days_3m":
        return f"Tickets take {value:.0f} days to resolve"
    if feature == "avg_days_late_3m":
        return f"Pays {value:.0f} days late on average"
    if feature == "products_dropped_6m":
        return f"Dropped {int(value)} product{'s' if int(value) != 1 else ''} in last 6 months"
    if feature == "products_count":
        return f"Subscribes to {int(value)} product{'s' if int(value) != 1 else ''}"
    if feature == "active_lines":
        return f"{int(value):,} active lines"
    if feature == "data_gb_per_line":
        return f"{value:.1f} GB data per line per month"
    if feature in ("overdue_amount", "mrr"):
        return f"{FEATURE_LABELS[feature]}: KWD {value:,.0f}".replace(" (KWD)", "")
    if feature == "tenure_months":
        return f"Customer for {value:.0f} months"
    if feature == "contract_term":
        return ("Month-to-month contract (no lock-in)" if value == "Month-to-month"
                else f"On a {value} contract")
    if feature in CATEGORICAL_FEATURES:
        return f"{FEATURE_LABELS[feature]}: {value}"
    if isinstance(value, (int, float, np.number)):
        return f"{FEATURE_LABELS.get(feature, feature)}: {value:,.1f}"
    return f"{FEATURE_LABELS.get(feature, feature)}: {value}"


def top_reasons(X: pd.DataFrame, contrib: pd.DataFrame, n: int = 3,
                direction: str = "risk") -> list[list[str]]:
    """Top-n features pushing each row towards churn ('risk') or away ('protect')."""
    out = []
    for idx in X.index:
        row = contrib.loc[idx]
        row = row[row > 0].sort_values(ascending=False) if direction == "risk" \
            else row[row < 0].sort_values()
        out.append([_fmt(f, X.at[idx, f]) for f in row.index[:n]])
    return out
