"""Tests that protect the most important modelling rules."""
import pandas as pd
import pytest

from src.features import FEATURES, build_snapshot, load_raw

SNAPSHOT = pd.Timestamp("2025-06-30")


@pytest.fixture(scope="module")
def tables():
    return load_raw()


def _as_of(tables, s):
    """What the business could actually see on date s."""
    cust = tables["customers"].copy()
    cust["termination_date"] = cust["termination_date"].where(cust["termination_date"] <= s)
    inv = tables["invoices"][tables["invoices"]["invoice_date"] <= s].copy()
    inv["paid_date"] = inv["paid_date"].where(inv["paid_date"] <= s)
    tix = tables["tickets"][tables["tickets"]["created_date"] <= s].copy()
    tix["resolved_date"] = tix["resolved_date"].where(tix["resolved_date"] <= s)
    use = tables["monthly_usage"][tables["monthly_usage"]["month"] <= s]
    return {"customers": cust, "monthly_usage": use, "invoices": inv, "tickets": tix}


def test_no_future_data_leaks_into_features(tables):
    """Features must be identical whether or not data after the snapshot exists."""
    full = build_snapshot(tables, SNAPSHOT, with_label=False)
    truncated = build_snapshot(_as_of(tables, SNAPSHOT), SNAPSHOT, with_label=False)
    numeric = [f for f in FEATURES if f in full.select_dtypes("number").columns]
    pd.testing.assert_frame_equal(
        full.set_index("customer_id")[numeric].sort_index(),
        truncated.set_index("customer_id")[numeric].sort_index(),
        check_dtype=False)


def test_terminated_accounts_are_not_scored(tables):
    snap = build_snapshot(tables, SNAPSHOT)
    term = tables["customers"].set_index("customer_id")["termination_date"]
    assert not (term.reindex(snap["customer_id"]) <= SNAPSHOT).any()
    assert snap["customer_id"].is_unique


def test_label_is_binary_and_churn_rate_is_plausible(tables):
    snap = build_snapshot(tables, SNAPSHOT)
    assert set(snap["churned"].unique()) <= {0, 1}
    assert 0.01 < snap["churned"].mean() < 0.20


def test_no_missing_feature_values(tables):
    snap = build_snapshot(tables, SNAPSHOT)
    assert not snap[FEATURES].isna().any().any()


def test_status_and_termination_date_are_not_features():
    assert "status" not in FEATURES and "termination_date" not in FEATURES
