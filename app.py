"""ChurnRadar for Business - B2B telecom churn prediction app (Streamlit).

Run locally:  streamlit run app.py
The app reads only the artifacts produced by `src/train.py` and `src/score.py`.
It never touches raw data, which mirrors how it would run in production.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import ARTIFACTS_DIR, PREDICTION_WINDOW_DAYS, ROOT
from src.features import FEATURE_LABELS, FEATURES
from src.model_utils import contributions, load_bundle, prepare_X, risk_band, top_reasons

ASSETS = ROOT / "assets"
st.set_page_config(page_title="ChurnRadar for Business", page_icon=str(ASSETS / "icon.svg"),
                   layout="wide")

BAND_COLORS = {"High": "#D1495B", "Medium": "#EDAE49", "Low": "#00798C"}
BAND_ORDER = ["High", "Medium", "Low"]
BLUE, NAVY, GREY = "#30638E", "#1E3A5F", "#6B7C93"

st.html("""
<style>
  .block-container {padding-top: 2.2rem;}
  div[data-testid="stMetric"] {background: #FFFFFF;}
  div[data-testid="stMetricValue"] {font-weight: 700; color: #1E3A5F;}
  .cr-strip {display:flex; flex-wrap:wrap; gap:8px; margin: -6px 0 14px 0;}
  .cr-pill {font-size: 12.5px; padding: 3px 10px; border-radius: 999px; background:#EEF3F8;
            color:#1E3A5F; border:1px solid #D9E3EE;}
  .cr-live {background:#E8F5EE; color:#1B7A45; border-color:#BFE3CD; font-weight:600;}
  .cr-dot {display:inline-block; width:8px; height:8px; border-radius:50%; background:#22A55B;
           margin-right:6px; animation: crpulse 1.6s infinite;}
  @keyframes crpulse {0%{opacity:1} 50%{opacity:.35} 100%{opacity:1}}
  .cr-alert {padding:8px 12px; border-left:4px solid #D1495B; background:#FDF3F4; border-radius:6px;
             margin-bottom:6px; font-size:14px;}
  .cr-alert.up {border-left-color:#00798C; background:#EEF7F8;}
  .cr-muted {color:#6B7C93; font-size:13px;}
</style>
""")


# ---------------------------------------------------------------- data loading
@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_data
def get_data():
    a = ARTIFACTS_DIR
    return dict(
        scores=pd.read_csv(a / "churn_scores.csv"),
        usage=pd.read_csv(a / "customer_monthly_usage.csv", parse_dates=["month"]),
        history=pd.read_csv(a / "score_history.csv", parse_dates=["run_date"]),
        metrics=json.loads((a / "metrics.json").read_text()),
        run=json.loads((a / "run_info.json").read_text()),
        lift=pd.read_csv(a / "lift_table.csv"),
        importance=pd.read_csv(a / "feature_importance.csv"),
        backtest=pd.read_csv(a / "backtest.csv", parse_dates=["snapshot_date"]),
    )


bundle = get_bundle()
model, categories = bundle["model"], bundle["categories"]
D = get_data()
scores_all, history_all, RUN = D["scores"], D["history"], D["run"]


# ---------------------------------------------------------------- helpers
def kwd(x: float) -> str:
    if abs(x) >= 1e6:
        return f"KWD {x / 1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"KWD {x / 1e3:,.0f}K"
    return f"KWD {x:,.0f}"


def header(title: str, subtitle: str) -> None:
    st.title(title)
    st.html(f"""<div class="cr-strip">
      <span class="cr-pill cr-live"><span class="cr-dot"></span>Live · scoring run {RUN['data_as_of']}</span>
      <span class="cr-pill">Model {RUN['model_version']}</span>
      <span class="cr-pill">{RUN['accounts_scored']:,} active business accounts</span>
      <span class="cr-pill">Next scheduled run {RUN['next_scheduled_run']}</span>
    </div>""")
    st.caption(subtitle)


def gauge(p: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "font": {"size": 40}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": BAND_COLORS[risk_band(p)]},
               "steps": [{"range": [0, 15], "color": "#E6F2F4"},
                         {"range": [15, 35], "color": "#FCF1DE"},
                         {"range": [35, 100], "color": "#F8E1E4"}]}))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=30, b=0))
    return fig


def shap_bars(X_row: pd.DataFrame, n: int = 8) -> go.Figure:
    """Which features push this account's risk up (red) or down (teal)."""
    c = contributions(model, X_row).iloc[0]
    top = c.reindex(c.abs().sort_values(ascending=False).index)[:n][::-1]
    row = X_row.iloc[0]
    labels = [f"{FEATURE_LABELS[f]} = {row[f]:,.2f}".rstrip("0").rstrip(".")
              if isinstance(row[f], (int, float, np.number)) else f"{FEATURE_LABELS[f]} = {row[f]}"
              for f in top.index]
    fig = go.Figure(go.Bar(
        x=top.values, y=labels, orientation="h",
        marker_color=[BAND_COLORS["High"] if v > 0 else BAND_COLORS["Low"] for v in top.values]))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Effect on churn risk (log-odds, SHAP)")
    return fig


REASON_GROUPS = [  # (text pattern, business-friendly category)
    ("port-out", "Numbers porting out to a competitor"),
    ("renews in", "Contract renewal coming up"),
    ("lines down", "Lines falling"), ("disconnected", "Lines falling"),
    ("data usage down", "Data usage falling"), ("voice usage down", "Voice usage falling"),
    ("revenue down", "Revenue falling"),
    ("network", "Network / data-speed complaints"), ("billing", "Billing disputes"),
    ("sla", "SLA breaches"), ("unresolved", "Open support tickets"),
    ("resolve", "Slow ticket resolution"), ("late", "Paying late"), ("overdue", "Paying late"),
    ("dropped", "Dropped products"), ("active line", "Small account"),
    ("monthly recurring revenue", "Small account"), ("customer for", "New customer"),
    ("month-to-month", "No contract lock-in"), ("contract", "Contract type"),
    ("industry", "Higher-churn industry"), ("region", "Higher-churn region"),
    ("segment", "Segment"), ("gb data per line", "Data usage pattern"),
]


def reason_group(text: str) -> str:
    t = str(text).lower()
    for pattern, group in REASON_GROUPS:
        if pattern in t:
            return group
    return "Other"


def suggested_action(reasons: str) -> str:
    r = reasons.lower()
    if "port-out" in r or "lines down" in r or "disconnected" in r:
        return ("**Retention team, this week.** Numbers are already moving to another operator. "
                "Call the decision-maker, find the competing offer and prepare a counter-offer.")
    if "renews in" in r:
        return ("**Start the renewal conversation now.** Review usage, right-size the plan and "
                "prepare a loyalty offer before the contract end date.")
    if "network" in r or "sla" in r:
        return ("**Fix the service first.** Escalate to network operations, share an action plan "
                "with the customer and consider SLA credits.")
    if "billing" in r or "late" in r or "overdue" in r:
        return ("**Resolve billing issues.** Close open disputes with finance and agree a "
                "payment plan if needed.")
    if "usage down" in r or "dropped" in r:
        return ("**Usage review.** Check whether the plan still fits; offer a better-fitting "
                "bundle before they shop around.")
    return "Monitor and include in the next quarterly business review."


def risk_table_config() -> dict:
    return {
        "customer_id": "ID", "company_name": "Account", "segment": "Segment", "region": "Region",
        "account_manager": "Account manager", "contract_term": "Contract",
        "months_to_renewal": "Renews in (months)", "risk_band": "Band",
        "active_lines": st.column_config.NumberColumn("Lines", format="localized"),
        "churn_probability": st.column_config.ProgressColumn(
            "Churn probability", format="percent", min_value=0, max_value=1),
        "revenue_at_risk": st.column_config.NumberColumn("Revenue at risk (KWD)", format="localized"),
        "reason_1": "Main reason", "reason_2": "Second reason",
    }


# ---------------------------------------------------------------- filters (shared)
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("segment", "region", "contract_term", "account_manager"):
        chosen = st.session_state.get(f"f_{col}") or []
        if chosen:
            df = df[df[col].isin(chosen)]
    return df


def sidebar_filters() -> None:
    with st.sidebar.expander("Filters", expanded=False, icon=":material/filter_list:"):
        for col, label in (("segment", "Segment"), ("region", "Region"),
                           ("contract_term", "Contract term"), ("account_manager", "Account manager")):
            st.multiselect(label, sorted(scores_all[col].unique()), key=f"f_{col}")
    st.sidebar.caption(f"Data as of **{RUN['data_as_of']}** · scored {RUN['scored_at_utc']} UTC  \n"
                       f"Predicts: account terminates in the next {PREDICTION_WINDOW_DAYS} days")
    st.sidebar.info("All data in this demo is synthetic (simulated Kuwaiti operator, KWD).",
                    icon=":material/info:")


# ================================================================ PAGES
def page_overview():
    header("Churn risk overview",
           "Which business accounts are likely to leave in the next 90 days, how many lines and how "
           "much revenue are at stake, and what changed since the last scoring run.")
    view, hist = apply_filters(scores_all), apply_filters(history_all)
    runs = sorted(hist["run_date"].unique())
    by_run = hist.groupby("run_date").agg(
        accounts=("customer_id", "size"), high=("risk_band", lambda s: (s == "High").sum()),
        lines=("lines_at_risk", "sum"), rar=("revenue_at_risk", "sum")).reindex(runs).fillna(0)
    prev = by_run.iloc[-2] if len(by_run) > 1 else by_run.iloc[-1]

    k = st.columns(4)
    k[0].metric("Active business accounts", f"{len(view):,}",
                f"{len(view) - prev['accounts']:+,.0f} vs last run", delta_color="off", delta_arrow="off",
                border=True, icon=":material/apartment:", chart_data=by_run["accounts"].tolist(), chart_type="area")
    high_now = int((view["risk_band"] == "High").sum())
    k[1].metric("High-risk accounts", f"{high_now:,}", f"{high_now - prev['high']:+,.0f} vs last run",
                delta_color="inverse", border=True, icon=":material/warning:", chart_data=by_run["high"].tolist(),
                chart_type="bar", help="Churn probability ≥ 35%")
    k[2].metric("Lines at risk", f"{view['lines_at_risk'].sum():,.0f}",
                f"{view['lines_at_risk'].sum() - prev['lines']:+,.0f} vs last run",
                delta_color="inverse", border=True, icon=":material/sim_card:", chart_data=by_run["lines"].round().tolist(),
                chart_type="area", help="Sum of churn probability × active lines")
    k[3].metric("Revenue at risk (annual)", kwd(view["revenue_at_risk"].sum()),
                f"{(view['revenue_at_risk'].sum() - prev['rar']) / 1e3:+,.0f}K vs last run",
                delta_color="inverse", border=True, icon=":material/payments:", chart_data=by_run["rar"].round().tolist(),
                chart_type="area", help="Sum of churn probability × annualised monthly recurring revenue")

    # ---- What changed since last run
    if len(runs) > 1:
        cur = hist[hist["run_date"] == runs[-1]].set_index("customer_id")
        old = hist[hist["run_date"] == runs[-2]].set_index("customer_id")
        both = cur.join(old[["risk_band", "churn_probability"]], rsuffix="_prev", how="left")
        rank = {"Low": 0, "Medium": 1, "High": 2}
        both["move"] = both["risk_band"].map(rank) - both["risk_band_prev"].map(rank).fillna(0)
        escalated = both[both["move"] > 0].join(scores_all.set_index("customer_id")[
            ["company_name", "reason_1", "revenue_at_risk"]], rsuffix="_s")
        escalated = escalated.sort_values("revenue_at_risk_s", ascending=False)
        improved = both[both["move"] < 0]
        with st.container(border=True):
            st.markdown(f"##### :material/notifications_active: What changed since the last run "
                        f"({pd.Timestamp(runs[-2]):%d %b} → {pd.Timestamp(runs[-1]):%d %b %Y})")
            a, b = st.columns([2.2, 1])
            with a:
                if escalated.empty:
                    st.markdown("No accounts moved to a higher risk band this month.")
                for cid, r in escalated.head(5).iterrows():
                    st.html(f"""<div class="cr-alert"><b>{r['company_name']}</b> moved
                        <b>{r['risk_band_prev'] if pd.notna(r['risk_band_prev']) else 'new'} → {r['risk_band']}</b>
                        · {r['churn_probability']:.0%} · {kwd(r['revenue_at_risk_s'])} at risk
                        <span class="cr-muted"> · {r['reason_1']}</span></div>""")
            with b:
                st.metric("Escalated to a higher band", f"{len(escalated):,}")
                st.metric("Improved to a lower band", f"{len(improved):,}")
                st.metric("Left since last run", f"{len(set(old.index) - set(cur.index)):,}",
                          help="Accounts scored last run that are no longer active")

    c1, c2 = st.columns(2)
    band = (view.groupby("risk_band").agg(accounts=("customer_id", "size"),
                                          revenue_at_risk=("revenue_at_risk", "sum"))
            .reindex(BAND_ORDER).fillna(0).reset_index())
    band["label"] = [f"{kwd(r)} · {int(a):,} accounts" for r, a in
                     zip(band["revenue_at_risk"], band["accounts"])]
    fig = px.bar(band, x="risk_band", y="revenue_at_risk", color="risk_band",
                 color_discrete_map=BAND_COLORS, text="label", title="Revenue at risk by risk band")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="KWD per year", height=360,
                      margin=dict(t=60))
    c1.plotly_chart(fig, width="stretch")

    buckets = pd.cut(view["months_to_renewal"], [0, 1, 2, 3, 6, 12, 24],
                     labels=["1 mo", "2 mo", "3 mo", "4–6 mo", "7–12 mo", "13–24 mo"])
    ren = (view.groupby(buckets, observed=False)
           .agg(avg_risk=("churn_probability", "mean"), accounts=("customer_id", "size"))
           .reset_index())
    fig = px.bar(ren, x="months_to_renewal", y="avg_risk", hover_data=["accounts"],
                 title="Average churn risk by months to contract renewal",
                 color_discrete_sequence=[BAND_COLORS["High"]])
    fig.update_layout(height=360, yaxis_tickformat=".0%", yaxis_title="Avg churn probability",
                      xaxis_title="Months to renewal", xaxis_type="category")
    c2.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    seg = view.groupby("segment")["revenue_at_risk"].sum().sort_values().reset_index()
    fig = px.bar(seg, x="revenue_at_risk", y="segment", orientation="h",
                 title="Revenue at risk by segment", color_discrete_sequence=[BLUE])
    fig.update_layout(xaxis_title="KWD per year", yaxis_title=None, height=340)
    c3.plotly_chart(fig, width="stretch")

    at_risk = view[view["risk_band"] != "Low"]
    reasons = (at_risk.melt("customer_id", [f"reason_{i}" for i in (1, 2, 3)], value_name="reason")
               .dropna(subset=["reason"]))
    reasons = reasons[reasons["reason"] != ""].assign(reason=lambda d: d["reason"].map(reason_group))
    reasons = (reasons.drop_duplicates(["customer_id", "reason"])["reason"]
               .value_counts().head(8).sort_values().reset_index())
    reasons.columns = ["reason", "accounts"]
    fig = px.bar(reasons, x="accounts", y="reason", orientation="h", text="accounts",
                 title=f"Why accounts are at risk ({len(at_risk)} Medium + High accounts)",
                 color_discrete_sequence=["#EDAE49"])
    fig.update_traces(textposition="outside", textangle=0, cliponaxis=False)
    fig.update_layout(yaxis_title=None, xaxis_title="Accounts", height=340)
    c4.plotly_chart(fig, width="stretch")

    st.subheader("Top 15 accounts for the retention team")
    st.caption("High and Medium risk accounts, ranked by revenue at risk. Plan the campaign on the "
               "**Retention planner** page.")
    st.dataframe(at_risk.head(15).round({"revenue_at_risk": 0})[
        ["company_name", "segment", "account_manager", "contract_term", "months_to_renewal",
         "active_lines", "churn_probability", "risk_band", "revenue_at_risk", "reason_1", "reason_2"]],
        hide_index=True, width="stretch", column_config=risk_table_config())


def page_planner():
    header("Retention planner",
           "How many accounts should the retention team work this month? Set the campaign "
           "assumptions and see the expected revenue protected, cost and return.")
    view = apply_filters(scores_all).copy()

    with st.container(border=True):
        st.markdown("##### :material/tune: Campaign assumptions")
        a, b, c, d = st.columns(4)
        rank_by = a.segmented_control("Prioritise by", ["Revenue at risk", "Probability"],
                                      default="Revenue at risk")
        save_rate = b.slider("Save rate when contacted (%)", 5, 80, 35,
                             help="Share of would-be churners kept after a retention call and offer") / 100
        contact_cost = c.number_input("Cost per account contacted (KWD)", 0, 5000, 150, step=25,
                                      help="Account manager time, visit, solution review")
        discount = d.slider("Retention discount for saved accounts (%)", 0, 40, 10,
                            help="Price concession on annual revenue for accounts that stay") / 100

    col = "revenue_at_risk" if rank_by != "Probability" else "churn_probability"
    ranked = view.sort_values(col, ascending=False).reset_index(drop=True)
    ranked["exp_churn"] = ranked["churn_probability"]
    n_max = int(min(300, len(ranked)))

    def campaign(n: int, df: pd.DataFrame) -> dict:
        top = df.head(n)
        protected = save_rate * top["revenue_at_risk"].sum()
        cost = n * contact_cost + discount * protected
        return dict(n=n, churners=top["exp_churn"].sum(), saved=save_rate * top["exp_churn"].sum(),
                    protected=protected, cost=cost, net=protected - cost)

    curve = pd.DataFrame([campaign(n, ranked) for n in range(0, n_max + 1)])
    # Random targeting: every account has the average expected value
    avg_rar = view["revenue_at_risk"].mean()
    curve["net_random"] = [save_rate * n * avg_rar * (1 - discount) - n * contact_cost
                           for n in curve["n"]]
    best = curve.loc[curve["net"].idxmax()]

    n = st.slider("Accounts to contact this month", 0, n_max, int(min(best["n"], n_max)),
                  help=f"Best net value at {int(best['n'])} accounts with these assumptions")
    res = campaign(n, ranked)
    rnd = curve.loc[n, "net_random"]

    short = lambda x: f"{x / 1e3:,.0f}K" if abs(x) >= 1e3 else f"{x:,.0f}"  # noqa: E731
    k = st.columns(5)
    k[0].metric("Churners reached", f"{res['churners']:.1f}",
                f"{res['churners'] / max(view['churn_probability'].sum(), 1e-9):.0%} of expected",
                delta_color="off", delta_arrow="off", border=True,
                help="Expected number of would-be churners in the contacted list")
    k[1].metric("Accounts saved", f"{res['saved']:.1f}", border=True,
                help="Expected churners reached × save rate")
    k[2].metric("Revenue kept (KWD)", short(res["protected"]), border=True,
                help="Annual revenue protected by the saved accounts")
    k[3].metric("Campaign cost (KWD)", short(res["cost"]), border=True,
                help="Contact cost + retention discount on saved revenue")
    k[4].metric("Net value (KWD)", short(res["net"]),
                f"ROI {res['net'] / res['cost']:.1f}×" if res["cost"] else None, border=True)

    c1, c2 = st.columns([1.6, 1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["n"], y=curve["net"], name="Model-ranked list",
                             line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=curve["n"], y=curve["net_random"], name="Random accounts",
                             line=dict(color=GREY, dash="dash")))
    fig.add_vline(x=n, line_color=BAND_COLORS["High"], annotation_text=f"Your plan: {n}",
                  annotation_position="top")
    fig.add_trace(go.Scatter(x=[best["n"]], y=[best["net"]], mode="markers+text",
                             marker=dict(size=11, color="#22A55B"), name="Best net value",
                             text=[f"Best: {int(best['n'])} accounts"], textposition="bottom right"))
    fig.update_layout(title="Net value by number of accounts contacted", height=400,
                      xaxis_title="Accounts contacted (in priority order)", yaxis_title="Net value (KWD)",
                      legend=dict(orientation="h", y=-0.2))
    c1.plotly_chart(fig, width="stretch")
    with c2:
        with st.container(border=True):
            st.markdown("##### Why the model matters")
            st.markdown(f"Contacting **{n} accounts** chosen by the model is worth "
                        f"**{kwd(res['net'])}** net. Contacting {n} accounts at random would be "
                        f"worth **{kwd(rnd)}**.")
            if rnd != 0:
                st.markdown(f"That is **{kwd(res['net'] - rnd)} more value** from the same effort.")
            st.markdown(f"<span class='cr-muted'>Assumes a {save_rate:.0%} save rate, "
                        f"KWD {contact_cost:,} per contact and a {discount:.0%} retention discount. "
                        "Estimates are expected values from churn probabilities, not guarantees.</span>",
                        unsafe_allow_html=True)

    st.subheader(f"Call list: {n} accounts")
    call = ranked.head(n)[["company_name", "account_manager", "segment", "contract_term",
                           "months_to_renewal", "churn_probability", "risk_band",
                           "revenue_at_risk", "reason_1"]].round({"revenue_at_risk": 0})
    st.dataframe(call, hide_index=True, width="stretch", column_config=risk_table_config(), height=320)
    st.download_button("Download call list (CSV)", call.to_csv(index=False).encode(),
                       f"retention_call_list_{RUN['data_as_of']}.csv", "text/csv",
                       icon=":material/download:")


def page_trends():
    header("Trends & heatmaps",
           "How risk has moved over the last six monthly scoring runs, where it is concentrated, "
           "and which renewals are coming up.")
    hist, view = apply_filters(history_all), apply_filters(scores_all)
    st.caption("History shows the last six monthly runs, re-scored with the current model version.")

    by_run = (hist.groupby("run_date").agg(
        high=("risk_band", lambda s: (s == "High").sum()),
        medium=("risk_band", lambda s: (s == "Medium").sum()),
        rar=("revenue_at_risk", "sum"), accounts=("customer_id", "size")).reset_index())
    by_run["run"] = by_run["run_date"].dt.strftime("%b %Y")
    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=by_run["run"], y=by_run["high"], name="High",
                         marker_color=BAND_COLORS["High"]))
    fig.add_trace(go.Bar(x=by_run["run"], y=by_run["medium"], name="Medium",
                         marker_color=BAND_COLORS["Medium"]))
    fig.update_layout(barmode="stack", title="High and Medium risk accounts by scoring run",
                      height=360, yaxis_title="Accounts", legend=dict(orientation="h", y=-0.15))
    c1.plotly_chart(fig, width="stretch")
    fig = px.area(by_run, x="run", y="rar", title="Revenue at risk by scoring run",
                  color_discrete_sequence=[BAND_COLORS["High"]], markers=True)
    fig.update_layout(height=360, xaxis_title=None, yaxis_title="KWD per year")
    c2.plotly_chart(fig, width="stretch")

    st.subheader("Where risk is concentrated")
    metric = st.segmented_control("Show", ["Revenue at risk", "Average churn probability",
                                           "High-risk accounts"], default="Revenue at risk")
    agg = {"Revenue at risk": ("revenue_at_risk", "sum"),
           "Average churn probability": ("churn_probability", "mean"),
           "High-risk accounts": ("risk_band", lambda s: (s == "High").sum())}[metric or "Revenue at risk"]
    heat = view.groupby(["region", "segment"]).agg(v=agg).reset_index().pivot(
        index="region", columns="segment", values="v").fillna(0)
    heat = heat[[c for c in ["SME", "Corporate", "Enterprise", "Government"] if c in heat.columns]]
    fmt = {"Revenue at risk": lambda v: kwd(v), "Average churn probability": lambda v: f"{v:.1%}",
           "High-risk accounts": lambda v: f"{int(v)}"}[metric or "Revenue at risk"]
    fig = go.Figure(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale="Reds",
                               text=[[fmt(v) for v in row] for row in heat.values],
                               texttemplate="%{text}", hovertemplate="%{y} · %{x}: %{text}<extra></extra>"))
    fig.update_layout(height=380, title=f"{metric or 'Revenue at risk'} by governorate and segment",
                      margin=dict(t=50))
    st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns([1.4, 1])
    fixed = view[view["contract_term"] != "Month-to-month"].copy()
    fixed = fixed[fixed["months_to_renewal"] <= 6]
    base = pd.Timestamp(RUN["data_as_of"])
    fixed["renewal_month"] = [(base + pd.DateOffset(months=int(m))).strftime("%b %Y")
                              for m in fixed["months_to_renewal"]]
    order = [(base + pd.DateOffset(months=m)).strftime("%b %Y") for m in range(1, 7)]
    cal = (fixed.groupby(["renewal_month", "risk_band"]).agg(accounts=("customer_id", "size"))
           .reset_index())
    fig = px.bar(cal, x="renewal_month", y="accounts", color="risk_band",
                 color_discrete_map=BAND_COLORS, category_orders={"renewal_month": order,
                                                                  "risk_band": BAND_ORDER},
                 title="Renewal calendar: fixed-term contracts renewing in the next 6 months")
    fig.update_layout(height=380, xaxis_title=None, yaxis_title="Accounts",
                      legend=dict(orientation="h", y=-0.15, title=None))
    c3.plotly_chart(fig, width="stretch")

    with c4:
        runs = sorted(hist["run_date"].unique())
        if len(runs) > 1:
            cur = hist[hist["run_date"] == runs[-1]].set_index("customer_id")["risk_band"]
            old = hist[hist["run_date"] == runs[-2]].set_index("customer_id")["risk_band"]
            mig = pd.crosstab(old.reindex(cur.index).fillna("New"), cur).reindex(
                index=["High", "Medium", "Low", "New"], columns=BAND_ORDER).fillna(0).astype(int)
            mig = mig[mig.sum(axis=1) > 0]
            fig = go.Figure(go.Heatmap(z=mig.values, x=[f"Now: {c}" for c in mig.columns],
                                       y=[f"Last run: {r}" for r in mig.index], colorscale="Blues",
                                       zmin=0, zmax=max(10, int(np.sort(mig.values.ravel())[-2]) * 1.5), zauto=False,
                                       texttemplate="%{z}", showscale=False))
            fig.update_layout(height=380, title="Band movement since the last run",
                              yaxis_autorange="reversed", margin=dict(t=50))
            st.plotly_chart(fig, width="stretch")


def page_explorer():
    header("Account explorer", "Search the risk list, open an account and see why it is flagged.")
    view = apply_filters(scores_all)
    bands = st.pills("Risk band", BAND_ORDER, default=["High", "Medium"], selection_mode="multi")
    sub = view[view["risk_band"].isin(bands)] if bands else view
    query = st.text_input("Search account name or ID", placeholder="e.g. Bay Works or T01344",
                          icon=":material/search:")
    if query:
        q = query.lower()
        sub = sub[sub["company_name"].str.lower().str.contains(q) | sub["customer_id"].str.lower().str.contains(q)]
    st.dataframe(
        sub[["customer_id", "company_name", "segment", "region", "account_manager", "contract_term",
             "churn_probability", "risk_band", "revenue_at_risk", "reason_1"]].round({"revenue_at_risk": 0}),
        hide_index=True, width="stretch", height=280, column_config=risk_table_config())
    st.download_button("Download list as CSV", sub.to_csv(index=False).encode(),
                       "telecom_churn_risk_accounts.csv", "text/csv", icon=":material/download:")

    if not len(sub):
        st.info("No accounts match.")
        return
    options = sub["customer_id"] + " · " + sub["company_name"]
    pick = st.selectbox("Open an account", options)
    acc = sub[options == pick].iloc[0]
    st.divider()
    st.subheader(f"{acc['company_name']}  ·  {acc['segment']} · {acc['industry']} · {acc['region']}")
    a, b, c = st.columns([1, 1.3, 1.6])
    with a:
        st.plotly_chart(gauge(acc["churn_probability"]), width="stretch")
        st.markdown(f"**Risk band:** {acc['risk_band']}  \n"
                    f"**Contract:** {acc['contract_term']}, renews in "
                    f"{int(acc['months_to_renewal'])} month(s)  \n"
                    f"**Active lines:** {int(acc['active_lines']):,}  \n"
                    f"**Monthly revenue:** {kwd(acc['mrr'])}  \n"
                    f"**Revenue at risk:** {kwd(acc['revenue_at_risk'])} per year  \n"
                    f"**Account manager:** {acc['account_manager']}")
    with b:
        st.markdown("**Why the model flags this account**")
        reasons = [acc[f"reason_{i}"] for i in (1, 2, 3)
                   if isinstance(acc[f"reason_{i}"], str) and acc[f"reason_{i}"]]
        for r in reasons:
            st.markdown(f"🔺 {r}")
        if isinstance(acc["protective_factor"], str) and acc["protective_factor"]:
            st.markdown(f"🟢 {acc['protective_factor']}")
        st.markdown("**Suggested action**")
        st.markdown(suggested_action(" ".join(reasons)))
    with c:
        hist = history_all[history_all["customer_id"] == acc["customer_id"]]
        fig = px.line(hist, x="run_date", y="churn_probability", markers=True,
                      title="Churn risk over the last 6 scoring runs",
                      color_discrete_sequence=[BAND_COLORS[acc["risk_band"]]])
        fig.add_hrect(y0=0.35, y1=1, fillcolor=BAND_COLORS["High"], opacity=0.06, line_width=0)
        fig.add_hrect(y0=0.15, y1=0.35, fillcolor=BAND_COLORS["Medium"], opacity=0.08, line_width=0)
        fig.update_layout(height=280, yaxis_tickformat=".0%", yaxis_range=[0, 1], xaxis_title=None,
                          yaxis_title=None, margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns([1, 1.3])
    with c1:
        u = D["usage"][D["usage"]["customer_id"] == acc["customer_id"]]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=u["month"], y=u["active_lines"], fill="tozeroy",
                                 name="Active lines", line_color=BLUE), secondary_y=False)
        fig.add_trace(go.Bar(x=u["month"], y=u["port_out_requests"], name="Port-out requests",
                             marker_color=BAND_COLORS["High"]), secondary_y=True)
        fig.update_layout(title="Active lines and port-out requests by month", height=340,
                          margin=dict(t=40, b=10), legend=dict(orientation="h", y=-0.15))
        fig.update_yaxes(title_text="Lines", rangemode="tozero", secondary_y=False)
        fig.update_yaxes(title_text="Port-outs", rangemode="tozero", showgrid=False, tickformat="d",
                         dtick=max(1, int(u["port_out_requests"].max() // 4) or 1),
                         range=[0, max(5, u["port_out_requests"].max() * 1.3)], secondary_y=True)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Feature contributions (SHAP)**")
        st.plotly_chart(shap_bars(prepare_X(sub[options == pick], categories)), width="stretch")


def page_whatif():
    header("What-if scorer",
           "Start from an account, change its situation and see how the predicted risk responds. "
           "Useful for explaining the model to sales and retention leaders.")
    options = scores_all["customer_id"] + " · " + scores_all["company_name"]
    default = int(np.argmax(scores_all["risk_band"].eq("Low").values))
    base = scores_all[options == st.selectbox("Start from account", options, index=default)].iloc[0]

    def pick(name, col):
        return st.selectbox(name, categories[col], index=categories[col].index(base[col]))

    with st.form("whatif"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Account & contract**")
            segment = pick("Segment", "segment")
            industry = pick("Industry", "industry")
            region = pick("Region", "region")
            term = pick("Contract term", "contract_term")
            tenure = st.number_input("Tenure (months)", 0, 300, int(base["tenure_months"]))
            renewal = st.slider("Months to contract renewal", 1, 24, int(base["months_to_renewal"]))
        with c2:
            st.markdown("**Lines, revenue & usage**")
            lines = st.number_input("Active lines", 1, 100_000, int(base["active_lines"]))
            lines_chg = st.slider("Change in lines, last 3 months (%)", -100, 100,
                                  int(round(base["lines_change_pct_3m"] * 100)))
            port_outs = st.number_input("Port-out requests, last 3 months", 0, 500,
                                        int(base["port_out_requests_3m"]))
            disconnected = st.number_input("Lines disconnected, last 3 months", 0, 100_000,
                                           int(base["lines_disconnected_3m"]))
            mrr = st.number_input("Monthly recurring revenue (KWD)", 0.0, 1e8, float(base["mrr"]),
                                  step=100.0)
            data_chg = st.slider("Data usage change vs previous quarter (%)", -100, 100,
                                 int(round(base["data_usage_change_pct"] * 100)))
            products = st.slider("Products subscribed", 1, 5, int(base["products_count"]))
        with c3:
            st.markdown("**Service & payments**")
            net = st.number_input("Network complaints, last 3 months", 0, 200,
                                  int(base["network_tickets_3m"]))
            disputes = st.number_input("Billing disputes, last 3 months", 0, 100,
                                       int(base["billing_disputes_3m"]))
            sla = st.number_input("SLA breaches, last 3 months", 0, 200, int(base["sla_breaches_3m"]))
            open_t = st.number_input("Open tickets", 0, 200, int(base["open_tickets"]))
            late = st.slider("Avg days paid late", 0.0, 60.0, float(base["avg_days_late_3m"]))
            overdue = st.number_input("Overdue amount (KWD)", 0.0, 1e8, float(base["overdue_amount"]),
                                      step=100.0)
        st.form_submit_button("Score this account", type="primary", icon=":material/bolt:")

    def keep(slider_pct, col):
        """Use the exact stored value unless the user moved the % slider."""
        return base[col] if slider_pct == int(round(base[col] * 100)) else slider_pct / 100

    row = base.copy()
    row.update({
        "segment": segment, "industry": industry, "region": region, "contract_term": term,
        "tenure_months": tenure, "months_to_renewal": 1 if term == "Month-to-month" else renewal,
        "active_lines": lines, "lines_change_pct_3m": keep(lines_chg, "lines_change_pct_3m"),
        "port_out_requests_3m": port_outs, "lines_disconnected_3m": disconnected,
        "mrr": mrr, "data_usage_change_pct": keep(data_chg, "data_usage_change_pct"),
        "products_count": products, "network_tickets_3m": net, "billing_disputes_3m": disputes,
        "sla_breaches_3m": sla, "open_tickets": open_t, "avg_days_late_3m": late,
        "overdue_amount": overdue,
        "tickets_3m": base["tickets_3m"] + max(0, net - base["network_tickets_3m"])
                      + max(0, disputes - base["billing_disputes_3m"])})
    X = prepare_X(pd.DataFrame([row]), categories)
    p = float(model.predict_proba(X)[:, 1][0])
    reasons = top_reasons(X, contributions(model, X), n=3)[0]

    a, b = st.columns([1, 2])
    with a:
        st.plotly_chart(gauge(p), width="stretch")
        delta = p - base["churn_probability"]
        st.metric("Churn probability", f"{p:.1%}",
                  f"{delta * 100:+.1f} pts vs current" if abs(delta) >= 0.001 else None,
                  delta_color="inverse", border=True)
        st.markdown(f"**Risk band:** {risk_band(p)}  \n**Revenue at risk:** {kwd(p * mrr * 12)} per year")
        for r in reasons:
            st.markdown(f"🔺 {r}")
    with b:
        st.plotly_chart(shap_bars(X), width="stretch")


def page_model():
    m = D["metrics"]
    header("Model performance",
           f"Evaluated on unseen, later snapshots ({m['test_snapshots'][0]} → {m['test_snapshots'][1]}, "
           f"{m['test_rows']:,} account-months, 90-day churn rate {m['test_churn_rate']:.1%}). "
           "Training used earlier snapshots only.")
    lg, lr = m["lightgbm"], m["logistic_regression"]
    k = st.columns(4)
    k[0].metric("ROC-AUC", f"{lg['roc_auc']:.3f}", f"{lg['roc_auc'] - lr['roc_auc']:+.3f} vs baseline",
                border=True)
    k[1].metric("PR-AUC", f"{lg['pr_auc']:.3f}", f"{lg['pr_auc'] - lr['pr_auc']:+.3f} vs baseline",
                help=f"A random model would score {m['test_churn_rate']:.3f}", border=True)
    k[2].metric("Churners caught in top 10%", f"{lg['recall_top_10pct']:.0%}",
                f"{(lg['recall_top_10pct'] - lr['recall_top_10pct']) * 100:+.0f} pts vs baseline",
                border=True)
    k[3].metric("Churners caught in top 20%", f"{lg['recall_top_20pct']:.0%}",
                f"{(lg['recall_top_20pct'] - lr['recall_top_20pct']) * 100:+.0f} pts vs baseline",
                border=True)

    lift = D["lift"]
    c1, c2 = st.columns(2)
    fig = px.bar(lift, x="decile", y="lift", text=lift["lift"].round(1),
                 title="Lift by risk decile (1 = riskiest 10%)", color_discrete_sequence=[BLUE])
    fig.add_hline(y=1, line_dash="dash", annotation_text="random")
    fig.update_layout(height=360, xaxis=dict(dtick=1), xaxis_title="Risk decile",
                      yaxis_title="Lift vs average")
    c1.plotly_chart(fig, width="stretch")
    fig = px.line(lift, x="decile", y="cum_share_of_churners", markers=True,
                  title="Cumulative share of churners captured")
    fig.add_scatter(x=lift["decile"], y=lift["decile"] / 10, mode="lines",
                    line=dict(dash="dash", color="grey"), name="random")
    fig.update_layout(height=360, yaxis_tickformat=".0%", showlegend=False, xaxis=dict(dtick=1),
                      yaxis_title="Share of all churners", xaxis_title="Risk decile")
    c2.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    imp = D["importance"].head(12).sort_values("mean_abs_shap")
    fig = px.bar(imp, x="mean_abs_shap", y="label", orientation="h",
                 title="What drives churn (mean |SHAP|)", color_discrete_sequence=[BAND_COLORS["High"]])
    fig.update_layout(height=400, yaxis_title=None, xaxis_title="Average impact on prediction")
    c3.plotly_chart(fig, width="stretch")
    bt = (D["backtest"].rename(columns={"predicted_churn_rate": "Predicted", "actual_churn_rate": "Actual"})
          .assign(snapshot=lambda d: d["snapshot_date"].dt.strftime("%b %Y"))
          .melt("snapshot", ["Predicted", "Actual"], var_name="series", value_name="rate"))
    fig = px.line(bt, x="snapshot", y="rate", color="series", markers=True,
                  title="Backtest: predicted vs actual 90-day churn rate, by snapshot month",
                  color_discrete_map={"Predicted": BLUE, "Actual": BAND_COLORS["High"]})
    fig.update_layout(height=400, yaxis_tickformat=".1%", xaxis_title=None, yaxis_title="Churn rate",
                      yaxis_rangemode="tozero", legend=dict(orientation="h", y=-0.2, title=None))
    c4.plotly_chart(fig, width="stretch")

    st.subheader("Baseline vs main model")
    names = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier score (lower is better)",
             "recall_top_10pct": "Churners caught in top 10%", "recall_top_20pct": "Churners caught in top 20%"}
    comp = pd.DataFrame({
        name: {names[k_]: (f"{v:.0%}" if k_.startswith("recall") else f"{v:.3f}") for k_, v in mm.items()}
        for name, mm in (("Logistic regression (baseline)", lr), ("LightGBM", lg))})
    st.table(comp)
    st.caption("LightGBM uses monotonic constraints: more port-outs, complaints, disputes, SLA breaches "
               "or late payments can only increase predicted risk.")


def page_about():
    header("How it's built", "Method, pipeline status and how this demo maps to a production deployment.")
    k = st.columns(4)
    k[0].metric("Last scoring run", RUN["data_as_of"], f"scored {RUN['scored_at_utc']} UTC",
                delta_color="off", delta_arrow="off", border=True)
    k[1].metric("Next scheduled run", RUN["next_scheduled_run"], "after billing cycle closes",
                delta_color="off", delta_arrow="off", border=True)
    k[2].metric("Source rows processed", f"{sum(RUN['source_rows'].values()):,}",
                f"{len(RUN['source_rows'])} source tables", delta_color="off", delta_arrow="off", border=True)
    k[3].metric("Data-quality & leakage tests", "5 / 5 passing", "run on every push (CI)",
                delta_color="off", delta_arrow="off", border=True)

    arch = ROOT / "docs" / "architecture.png"
    if arch.exists():
        st.image(str(arch), width="stretch")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(f"""
**Business question:** which business (B2B) telecom accounts will terminate their services in the
next {PREDICTION_WINDOW_DAYS} days, and why, so the account and retention teams can act before lines and
revenue move to a competitor.

**Churn definition:** an active business account (billed this month) that **terminates all
services within the next {PREDICTION_WINDOW_DAYS} days**.

**Pipeline**
1. `src/generate_data.py`: simulates 1,500 business accounts over 46 months
2. `src/features.py`: monthly snapshots; {len(FEATURES)} features using only data up to each snapshot
3. `src/train.py`: time-based split, logistic regression baseline vs LightGBM (monotonic constraints)
4. `src/score.py`: monthly batch scoring, score history and the tables for this app and Power BI
5. `app.py`: this app
""")
    with c2:
        src = pd.DataFrame({"Source table": list(RUN["source_rows"]),
                            "Rows": [f"{v:,}" for v in RUN["source_rows"].values()],
                            "System": ["CRM", "Billing (BSS) / usage (OSS)", "Finance", "Service desk"]})
        st.markdown("**Data sources (synthetic)**")
        st.dataframe(src, hide_index=True, width="stretch")

    st.markdown("**Demo vs production**")
    st.table(pd.DataFrame({
        "This demo": ["Synthetic CSV files", "Streamlit Community Cloud", "Public link",
                      "Run scripts manually", "Files in repo", "Backtest page"],
        "Production design": ["Read-only views on CRM, billing (BSS) and usage (OSS) data",
                              "Azure App Service in the company network",
                              "Entra ID single sign-on + row-level security per account manager",
                              "Monthly scheduled job after the billing cycle closes",
                              "Azure Key Vault for secrets",
                              "Monthly predicted-vs-actual tracking, drift checks, quarterly retraining"],
    }, index=["Data", "Hosting", "Access", "Scoring", "Secrets", "Monitoring"]))


# ================================================================ navigation
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), size="large")
nav = st.navigation({
    "Monitor": [
        st.Page(page_overview, title="Overview", icon=":material/dashboard:", url_path="overview",
                default=True),
        st.Page(page_trends, title="Trends & heatmaps", icon=":material/trending_up:", url_path="trends"),
    ],
    "Act": [
        st.Page(page_planner, title="Retention planner", icon=":material/savings:", url_path="planner"),
        st.Page(page_explorer, title="Account explorer", icon=":material/manage_search:",
                url_path="accounts"),
        st.Page(page_whatif, title="What-if scorer", icon=":material/science:", url_path="what-if"),
    ],
    "Model": [
        st.Page(page_model, title="Model performance", icon=":material/insights:", url_path="model"),
        st.Page(page_about, title="How it's built", icon=":material/account_tree:", url_path="about"),
    ],
})
sidebar_filters()
nav.run()
