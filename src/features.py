"""Feature engineering: one row per active business account per month-end snapshot.

Every feature is computed ONLY from events on or before the snapshot date.
The label looks only at the PREDICTION_WINDOW_DAYS after the snapshot. This
is the guard against data leakage (tests/test_features.py checks it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (CATEGORICAL_FEATURES, PREDICTION_WINDOW_DAYS, RAW_DIR,
                        TRAIN_SNAPSHOT_END, TRAIN_SNAPSHOT_START)

NUMERIC_FEATURES = [
    "tenure_months",
    "months_to_renewal",
    "active_lines",
    "lines_change_pct_3m",
    "lines_disconnected_3m",
    "port_out_requests_3m",
    "mrr",
    "mrr_change_pct",
    "data_gb_per_line",
    "data_usage_change_pct",
    "voice_usage_change_pct",
    "products_count",
    "products_dropped_6m",
    "tickets_3m",
    "network_tickets_3m",
    "billing_disputes_3m",
    "sla_breaches_3m",
    "open_tickets",
    "avg_resolution_days_3m",
    "avg_days_late_3m",
    "overdue_amount",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Plain-English names used in the app, Power BI and SHAP reasons
FEATURE_LABELS = {
    "tenure_months": "Tenure (months)",
    "months_to_renewal": "Months to contract renewal",
    "active_lines": "Active lines",
    "lines_change_pct_3m": "Change in lines, last 3 months",
    "lines_disconnected_3m": "Lines disconnected, last 3 months",
    "port_out_requests_3m": "Port-out requests, last 3 months",
    "mrr": "Monthly recurring revenue (KWD)",
    "mrr_change_pct": "MRR change vs previous 3 months",
    "data_gb_per_line": "Data per line (GB/month)",
    "data_usage_change_pct": "Data usage change vs previous 3 months",
    "voice_usage_change_pct": "Voice usage change vs previous 3 months",
    "products_count": "Products subscribed",
    "products_dropped_6m": "Products dropped, last 6 months",
    "tickets_3m": "Support tickets, last 3 months",
    "network_tickets_3m": "Network complaints, last 3 months",
    "billing_disputes_3m": "Billing disputes, last 3 months",
    "sla_breaches_3m": "SLA breaches, last 3 months",
    "open_tickets": "Open support tickets",
    "avg_resolution_days_3m": "Avg ticket resolution (days)",
    "avg_days_late_3m": "Avg days paid late",
    "overdue_amount": "Overdue amount (KWD)",
    "segment": "Segment",
    "industry": "Industry",
    "region": "Region",
    "contract_term": "Contract term",
}

TERM_MONTHS = {"Month-to-month": 1, "12 months": 12, "24 months": 24}


def load_raw() -> dict[str, pd.DataFrame]:
    date_cols = {
        "customers": ["contract_start", "termination_date"],
        "monthly_usage": ["month"],
        "invoices": ["invoice_date", "due_date", "paid_date"],
        "tickets": ["created_date", "resolved_date"],
    }
    return {name: pd.read_csv(RAW_DIR / f"{name}.csv", parse_dates=cols)
            for name, cols in date_cols.items()}


def _months_between(later: pd.Timestamp, earlier: pd.Series) -> pd.Series:
    return (later.year - earlier.dt.year) * 12 + (later.month - earlier.dt.month)


def _pct_change(new: pd.Series, old: pd.Series) -> pd.Series:
    return pd.Series(np.where(old > 0, (new - old) / old, 0.0), index=new.index).clip(-1, 3)


def build_snapshot(tables: dict[str, pd.DataFrame], snapshot: str | pd.Timestamp,
                   with_label: bool = True) -> pd.DataFrame:
    s = pd.Timestamp(snapshot)
    m0 = s.to_period("M").to_timestamp()                       # snapshot month start
    m = lambda k: m0 - pd.DateOffset(months=k)                  # noqa: E731
    cust, usage = tables["customers"], tables["monthly_usage"]
    invoices, tickets = tables["invoices"], tables["tickets"]

    # Active = billed in the snapshot month and not terminated on/before snapshot
    billed = usage.loc[usage["month"] == m0, "customer_id"]
    alive = cust["termination_date"].isna() | (cust["termination_date"] > s)
    df = cust[alive & cust["customer_id"].isin(billed)].set_index("customer_id").copy()
    df = df.drop(columns=["status", "termination_date"])        # never used as features
    df["snapshot_date"] = s

    df["tenure_months"] = _months_between(s, df["contract_start"]).astype(float)
    term = df["contract_term"].map(TERM_MONTHS)
    since = _months_between(s, df["contract_start"])
    df["months_to_renewal"] = np.where(term == 1, 1, term - (since % term))

    u = usage[(usage["month"] <= m0) & (usage["month"] >= m(8))]
    u = u[u["customer_id"].isin(df.index)]
    piv = lambda col: u.pivot_table(index="customer_id", columns="month", values=col,  # noqa: E731
                                    aggfunc="sum").reindex(df.index)
    last3, prev3 = [m(0), m(1), m(2)], [m(3), m(4), m(5)]

    def window_mean(p, cols):
        return p.reindex(columns=cols).mean(axis=1)

    def window_sum(p, cols):
        return p.reindex(columns=cols).sum(axis=1, min_count=1).fillna(0)

    lines, mrr = piv("active_lines"), piv("mrr_kwd")
    data, voice, prods = piv("data_usage_gb"), piv("voice_minutes"), piv("products_count")

    df["active_lines"] = lines[m0]
    df["lines_change_pct_3m"] = _pct_change(lines[m0], lines.reindex(columns=[m(3)]).iloc[:, 0]
                                            .fillna(lines[m0]))
    df["lines_disconnected_3m"] = window_sum(piv("lines_disconnected"), last3)
    df["port_out_requests_3m"] = window_sum(piv("port_out_requests"), last3)
    df["mrr"] = mrr[m0].round(2)
    df["mrr_change_pct"] = _pct_change(window_mean(mrr, last3),
                                       window_mean(mrr, prev3).fillna(window_mean(mrr, last3)))
    df["data_gb_per_line"] = (window_mean(data, last3) / window_mean(lines, last3)).round(2)
    df["data_usage_change_pct"] = _pct_change(window_mean(data, last3),
                                              window_mean(data, prev3).fillna(window_mean(data, last3)))
    df["voice_usage_change_pct"] = _pct_change(window_mean(voice, last3),
                                               window_mean(voice, prev3).fillna(window_mean(voice, last3)))
    df["products_count"] = prods[m0]
    p6 = prods.reindex(columns=[m(6)]).iloc[:, 0].fillna(prods[m0])
    df["products_dropped_6m"] = (p6 - prods[m0]).clip(lower=0)

    # Support tickets - "open" means not resolved as of the snapshot date
    t_past = tickets[(tickets["created_date"] <= s) & tickets["customer_id"].isin(df.index)]
    t3 = t_past[t_past["created_date"] >= m(2)]
    count = lambda d: d.groupby("customer_id").size().reindex(df.index).fillna(0)  # noqa: E731
    df["tickets_3m"] = count(t3)
    df["network_tickets_3m"] = count(t3[t3["category"].isin(["Network / Coverage", "Data speed"])])
    df["billing_disputes_3m"] = count(t3[t3["category"] == "Billing dispute"])
    t3_res = t3[t3["resolved_date"] <= s]
    df["sla_breaches_3m"] = t3_res.groupby("customer_id")["sla_breached"].sum().reindex(df.index).fillna(0)
    is_open = t_past["resolved_date"].isna() | (t_past["resolved_date"] > s)
    df["open_tickets"] = count(t_past[is_open])
    res_days = (t3_res["resolved_date"] - t3_res["created_date"]).dt.days
    df["avg_resolution_days_3m"] = (res_days.groupby(t3_res["customer_id"]).mean()
                                    .reindex(df.index).fillna(0).round(2))

    # Payments - a bill not yet paid at the snapshot counts as late up to that date
    inv = invoices[invoices["customer_id"].isin(df.index)]
    inv3 = inv[(inv["due_date"] > s - pd.Timedelta(days=90)) & (inv["due_date"] <= s)].copy()
    paid_by_s = inv3["paid_date"].where(inv3["paid_date"] <= s, s)
    inv3["days_late"] = (paid_by_s - inv3["due_date"]).dt.days.clip(lower=0)
    df["avg_days_late_3m"] = (inv3.groupby("customer_id")["days_late"].mean()
                              .reindex(df.index).fillna(0).round(2))
    due = inv[inv["due_date"] <= s]
    unpaid = due[due["paid_date"].isna() | (due["paid_date"] > s)]
    df["overdue_amount"] = (unpaid.groupby("customer_id")["amount_kwd"].sum()
                            .reindex(df.index).fillna(0).round(2))

    if with_label:
        term_date = cust.set_index("customer_id")["termination_date"].reindex(df.index)
        horizon = s + pd.Timedelta(days=PREDICTION_WINDOW_DAYS)
        df["churned"] = ((term_date > s) & (term_date <= horizon)).astype(int)

    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype("category")
    return df.reset_index()


def build_training_set(tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    tables = tables or load_raw()
    snapshots = pd.date_range(TRAIN_SNAPSHOT_START, TRAIN_SNAPSHOT_END, freq="ME")
    data = pd.concat([build_snapshot(tables, s, with_label=True) for s in snapshots],
                     ignore_index=True)
    for c in CATEGORICAL_FEATURES:
        data[c] = data[c].astype("category")
    return data
