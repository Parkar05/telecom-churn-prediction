"""Batch scoring job - the part that would run on a schedule in production.

Scores every active business account as of SCORING_SNAPSHOT and writes:
    artifacts/churn_scores.csv            - one row per account: probability, band,
                                            revenue at risk, top 3 reasons
    artifacts/customer_monthly_usage.csv  - MRR and line trend used by the app
    powerbi/*.csv                         - star-schema tables for Power BI

Run:  python -m src.score
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from src.config import ARTIFACTS_DIR, ROOT, SCORING_SNAPSHOT
from src.features import NUMERIC_FEATURES, build_snapshot, load_raw
from src.model_utils import contributions, load_bundle, prepare_X, risk_band, top_reasons


def score(snapshot: str = SCORING_SNAPSHOT):
    bundle = load_bundle()
    model, categories = bundle["model"], bundle["categories"]
    tables = load_raw()

    snap = build_snapshot(tables, snapshot, with_label=False)
    X = prepare_X(snap, categories)
    proba = model.predict_proba(X)[:, 1]
    contrib = contributions(model, X)
    risk = top_reasons(X, contrib, n=3, direction="risk")
    protect = top_reasons(X, contrib, n=1, direction="protect")

    out = snap[["customer_id", "company_name", "segment", "industry", "region",
                "account_manager", "contract_term", "snapshot_date"]].copy()
    out["churn_probability"] = proba.round(4)
    out["risk_band"] = [risk_band(p) for p in proba]
    out["annual_revenue"] = (snap["mrr"] * 12).round(2)
    out["revenue_at_risk"] = (proba * out["annual_revenue"]).round(2)
    out["lines_at_risk"] = (proba * snap["active_lines"]).round(1)
    for i in range(3):
        out[f"reason_{i + 1}"] = [r[i] if len(r) > i else "" for r in risk]
    out["protective_factor"] = [p[0] if p else "" for p in protect]
    out["scored_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    out["model_version"] = "lgbm-telecom-v1"
    out = pd.concat([out, snap[NUMERIC_FEATURES]], axis=1)
    return out.sort_values("revenue_at_risk", ascending=False).reset_index(drop=True), tables


HISTORY_MONTHS = 6   # monthly scoring runs kept for trends and "what changed" alerts


def score_history(tables, model, categories, months: int = HISTORY_MONTHS) -> pd.DataFrame:
    """Re-run the monthly scoring job for the last `months` month-ends, as the
    production scheduler would have done, so the app can show trends."""
    runs = pd.date_range(end=pd.Timestamp(SCORING_SNAPSHOT), periods=months, freq="ME")
    frames = []
    for run in runs:
        snap = build_snapshot(tables, run, with_label=False)
        p = model.predict_proba(prepare_X(snap, categories))[:, 1]
        frames.append(pd.DataFrame({
            "run_date": run.date(), "customer_id": snap["customer_id"],
            "segment": snap["segment"].astype(str), "region": snap["region"].astype(str),
            "account_manager": snap["account_manager"], "contract_term": snap["contract_term"].astype(str),
            "churn_probability": p.round(4), "risk_band": [risk_band(x) for x in p],
            "revenue_at_risk": (p * snap["mrr"] * 12).round(2),
            "lines_at_risk": (p * snap["active_lines"]).round(1),
        }))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    scores, tables = score()
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    scores.to_csv(ARTIFACTS_DIR / "churn_scores.csv", index=False)

    bundle = load_bundle()
    history = score_history(tables, bundle["model"], bundle["categories"])
    history.to_csv(ARTIFACTS_DIR / "score_history.csv", index=False)

    run_info = {
        "data_as_of": SCORING_SNAPSHOT,
        "scored_at_utc": scores["scored_at_utc"].iloc[0],
        "model_version": scores["model_version"].iloc[0],
        "accounts_scored": int(len(scores)),
        "history_runs": sorted(str(d) for d in history["run_date"].unique()),
        "next_scheduled_run": str((pd.Timestamp(SCORING_SNAPSHOT) + pd.offsets.MonthEnd(1)).date()),
        "source_rows": {k: int(len(v)) for k, v in tables.items()},
        "train_window": bundle["metrics"]["train_snapshots"],
        "test_window": bundle["metrics"]["test_snapshots"],
    }
    (ARTIFACTS_DIR / "run_info.json").write_text(json.dumps(run_info, indent=2))

    usage = tables["monthly_usage"]
    trend = usage[usage["customer_id"].isin(scores["customer_id"])][
        ["customer_id", "month", "active_lines", "mrr_kwd", "data_usage_gb",
         "port_out_requests"]]
    trend.to_csv(ARTIFACTS_DIR / "customer_monthly_usage.csv", index=False)

    # Power BI star schema
    pbi = ROOT / "powerbi"
    pbi.mkdir(exist_ok=True)
    tables["customers"].drop(columns=["status", "termination_date"]).to_csv(
        pbi / "dim_customer.csv", index=False)
    score_cols = ["customer_id", "snapshot_date", "churn_probability", "risk_band",
                  "annual_revenue", "revenue_at_risk", "lines_at_risk", "months_to_renewal",
                  "active_lines", "reason_1", "reason_2", "reason_3", "protective_factor",
                  "model_version", "scored_at_utc"]
    scores[score_cols].to_csv(pbi / "fact_churn_scores.csv", index=False)
    usage.to_csv(pbi / "fact_monthly_usage.csv", index=False)

    print(f"Scored {len(scores):,} active accounts as of {SCORING_SNAPSHOT}")
    print(scores["risk_band"].value_counts().to_string())
    print(f"Revenue at risk (annualised): KWD {scores['revenue_at_risk'].sum():,.0f}")
    print(f"Lines at risk: {scores['lines_at_risk'].sum():,.0f}")
    print(scores[["company_name", "churn_probability", "risk_band", "reason_1"]].head(8).to_string())


if __name__ == "__main__":
    main()
