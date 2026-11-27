# Power BI: B2B telecom churn risk report

The scoring job (`python -m src.score`) writes three CSV files to this folder. They form a small
star schema, so the report needs no Power Query logic beyond setting data types.

| File | Grain | Key columns |
|---|---|---|
| `dim_customer.csv` | one row per business account | customer_id, company_name, segment, industry, region, account_manager, contract_term, contract_start |
| `fact_churn_scores.csv` | one row per active account per scoring run | customer_id, snapshot_date, churn_probability, risk_band, annual_revenue, revenue_at_risk, lines_at_risk, months_to_renewal, reason_1–3 |
| `fact_monthly_usage.csv` | account × month | customer_id, month, active_lines, port_out_requests, data_usage_gb, voice_minutes, products_count, mrr_kwd |

## 1. Load and model

1. **Get data → Text/CSV** for each file. Set the date columns (`snapshot_date`, `month`,
   `contract_start`) to Date, and `churn_probability` to Decimal number.
2. **Model view:** relate `dim_customer[customer_id]` 1 → * `fact_churn_scores[customer_id]` and
   1 → * `fact_monthly_usage[customer_id]`. Single direction, from dim to fact.
3. In Power Query, add a sort column for risk band:
   `= Table.AddColumn(Source, "risk_band_sort", each if [risk_band] = "High" then 1 else if [risk_band] = "Medium" then 2 else 3)`
4. Add a renewal bucket column:
   `= Table.AddColumn(Source, "renewal_bucket", each if [months_to_renewal] <= 3 then "0–3 months" else if [months_to_renewal] <= 6 then "4–6 months" else "7+ months")`

## 2. DAX measures

```DAX
Active Accounts = DISTINCTCOUNT ( fact_churn_scores[customer_id] )

High Risk Accounts =
CALCULATE ( [Active Accounts], fact_churn_scores[risk_band] = "High" )

High Risk % = DIVIDE ( [High Risk Accounts], [Active Accounts] )

Expected Churners = SUM ( fact_churn_scores[churn_probability] )

Lines at Risk = SUM ( fact_churn_scores[lines_at_risk] )

Revenue at Risk (Annual) = SUM ( fact_churn_scores[revenue_at_risk] )

Annual Revenue = SUM ( fact_churn_scores[annual_revenue] )

Revenue at Risk % = DIVIDE ( [Revenue at Risk (Annual)], [Annual Revenue] )

Renewals Next 90 Days =
CALCULATE ( [Active Accounts], fact_churn_scores[months_to_renewal] <= 3 )

High Risk Renewals Next 90 Days =
CALCULATE ( [High Risk Accounts], fact_churn_scores[months_to_renewal] <= 3 )

Port-out Requests (3m) =
CALCULATE (
    SUM ( fact_monthly_usage[port_out_requests] ),
    DATESINPERIOD ( fact_monthly_usage[month], MAX ( fact_monthly_usage[month] ), -3, MONTH )
)

-- Latest scoring run only (useful once you append score history)
Latest Snapshot = MAX ( fact_churn_scores[snapshot_date] )
```

## 3. Report pages

**Page 1: Risk overview**
- Cards: Active Accounts, High Risk Accounts, Lines at Risk, Revenue at Risk (Annual)
- Column chart: average churn_probability by renewal_bucket
- Bar charts: Revenue at Risk by segment and by account manager
- Slicers: segment, region, contract_term, account manager

**Page 2: Retention call list**
- Table: company_name, account_manager, contract_term, months_to_renewal, active_lines,
  churn_probability (data bars), risk_band, revenue_at_risk, reason_1, reason_2, reason_3
- Sorted by revenue_at_risk descending, with conditional formatting on risk_band

**Page 3: Account drill-through**
- Drill-through on customer_id
- Line chart from `fact_monthly_usage`: mrr_kwd and active_lines by month
- Column chart: port_out_requests by month
- Multi-row card: churn probability, band, months to renewal, the three reasons

**Page 4: Model health** (optional)
- Import `artifacts/backtest.csv` and `artifacts/lift_table.csv`
- Line chart of predicted vs actual churn rate per month; column chart of lift by decile

## 4. Row-level security (production)

So each account manager sees only their own accounts:

1. **Modeling → Manage roles → New role** `Account Manager`
2. Table filter on `dim_customer`:
   ```DAX
   [account_manager_email] = USERPRINCIPALNAME ()
   ```
   (In production, add an `account_manager_email` column to the customer dimension.)
3. Publish, then assign people or groups to the role in the semantic model's **Security** settings.

## 5. Refresh

- **Demo:** files are in the repo; re-run `python run_pipeline.py` and refresh.
- **Production:** the monthly scoring job runs after the billing cycle closes and writes
  `fact_churn_scores` to a SQL table or lakehouse; the semantic model refreshes after it.
  No Python runs inside Power BI.
