"""Train and evaluate the churn model.

- Builds monthly snapshots (features + 90-day churn label)
- Time-based split: older snapshots train, the most recent labelled ones test
- Baseline: logistic regression. Main model: LightGBM
- Saves the model bundle, metrics, lift table and feature importance to artifacts/

Run:  python -m src.train
"""
from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (ARTIFACTS_DIR, CATEGORICAL_FEATURES, PREDICTION_WINDOW_DAYS,
                        RANDOM_STATE, TEST_SNAPSHOT_START)
from src.features import FEATURE_LABELS, FEATURES, NUMERIC_FEATURES, build_training_set
from src.model_utils import MODEL_PATH, contributions, prepare_X

# Business-logic constraints: the model may only learn that MORE of these signals
# means MORE risk (+1), or that a HIGHER value means LESS risk (-1). Without them,
# confounders (big accounts log more tickets; regional outages hit healthy accounts)
# can make e.g. network complaints look protective, which no business user would trust.
MONOTONE = {
    "port_out_requests_3m": 1, "lines_disconnected_3m": 1, "network_tickets_3m": 1,
    "billing_disputes_3m": 1, "sla_breaches_3m": 1, "open_tickets": 1,
    "avg_resolution_days_3m": 1, "avg_days_late_3m": 1, "products_dropped_6m": 1,
    "lines_change_pct_3m": -1, "data_usage_change_pct": -1, "voice_usage_change_pct": -1,
    "mrr_change_pct": -1,
}

LGB_PARAMS = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    monotone_constraints=[MONOTONE.get(f, 0) for f in FEATURES],
    monotone_constraints_method="advanced",
    random_state=RANDOM_STATE, verbose=-1,
)


def recall_at_top(y_true, y_score, pct: float) -> float:
    n = max(1, int(len(y_score) * pct))
    top = np.argsort(-np.asarray(y_score))[:n]
    return float(np.asarray(y_true)[top].sum() / max(1, np.sum(y_true)))


def lift_table(y_true, y_score, annual_revenue) -> pd.DataFrame:
    d = pd.DataFrame({"y": y_true, "p": y_score, "rev": annual_revenue})
    d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    t = d.groupby("decile", observed=True).agg(
        accounts=("y", "size"), churners=("y", "sum"), avg_predicted=("p", "mean"),
        annual_revenue=("rev", "sum")).reset_index()
    t["churn_rate"] = t["churners"] / t["accounts"]
    t["lift"] = t["churn_rate"] / d["y"].mean()
    t["cum_share_of_churners"] = t["churners"].cumsum() / t["churners"].sum()
    t["decile"] = t["decile"].astype(int)
    return t


def evaluate(y, p) -> dict:
    return {
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "recall_top_10pct": round(recall_at_top(y, p, 0.10), 4),
        "recall_top_20pct": round(recall_at_top(y, p, 0.20), 4),
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    data = build_training_set()
    categories = {c: sorted(data[c].astype(str).unique()) for c in CATEGORICAL_FEATURES}

    is_test = data["snapshot_date"] >= pd.Timestamp(TEST_SNAPSHOT_START)
    # Leave a gap equal to the prediction window so training labels never
    # overlap the test period.
    gap_end = pd.Timestamp(TEST_SNAPSHOT_START) - pd.Timedelta(days=PREDICTION_WINDOW_DAYS)
    train, test = data[data["snapshot_date"] <= gap_end], data[is_test]
    X_tr, X_te = prepare_X(train, categories), prepare_X(test, categories)
    y_tr, y_te = train["churned"].values, test["churned"].values

    # Baseline - logistic regression
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)])
    baseline = make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.5))
    baseline.fit(X_tr, y_tr)
    p_base = baseline.predict_proba(X_te)[:, 1]

    # Main model - LightGBM
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_tr, y_tr, categorical_feature=CATEGORICAL_FEATURES)
    p_lgb = model.predict_proba(X_te)[:, 1]

    metrics = {
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_snapshots": [str(train["snapshot_date"].min().date()),
                            str(train["snapshot_date"].max().date())],
        "test_snapshots": [str(test["snapshot_date"].min().date()),
                           str(test["snapshot_date"].max().date())],
        "test_churn_rate": round(float(y_te.mean()), 4),
        "logistic_regression": evaluate(y_te, p_base),
        "lightgbm": evaluate(y_te, p_lgb),
    }

    lift = lift_table(y_te, p_lgb, (test["mrr"] * 12).values)
    contrib = contributions(model, X_te)
    importance = (contrib.abs().mean().sort_values(ascending=False)
                  .rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"}))
    importance["label"] = importance["feature"].map(FEATURE_LABELS)

    # Calibration table (predicted vs actual by probability bin)
    calib = pd.DataFrame({"p": p_lgb, "y": y_te})
    calib["bin"] = pd.cut(calib["p"], [0, .05, .1, .2, .3, .5, .7, 1.0], include_lowest=True)
    calib = calib.groupby("bin", observed=True).agg(
        predicted=("p", "mean"), actual=("y", "mean"), accounts=("y", "size")).reset_index()
    calib["bin"] = calib["bin"].astype(str)

    # Monthly backtest: predicted vs actual churn per test snapshot
    bt = test[["snapshot_date"]].assign(predicted=p_lgb, actual=y_te)
    bt = bt.groupby("snapshot_date").agg(predicted_churn_rate=("predicted", "mean"),
                                          actual_churn_rate=("actual", "mean"),
                                          accounts=("actual", "size")).reset_index()

    # Refit on ALL labelled snapshots for production scoring
    X_all = prepare_X(data, categories)
    final_model = lgb.LGBMClassifier(**LGB_PARAMS)
    final_model.fit(X_all, data["churned"].values, categorical_feature=CATEGORICAL_FEATURES)

    joblib.dump({"model": final_model, "features": FEATURES, "categories": categories,
                 "metrics": metrics}, MODEL_PATH)
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    lift.to_csv(ARTIFACTS_DIR / "lift_table.csv", index=False)
    importance.to_csv(ARTIFACTS_DIR / "feature_importance.csv", index=False)
    calib.to_csv(ARTIFACTS_DIR / "calibration.csv", index=False)
    bt.to_csv(ARTIFACTS_DIR / "backtest.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print("\nLift by decile (test):\n", lift.round(3).to_string(index=False))
    print(f"\nSaved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
