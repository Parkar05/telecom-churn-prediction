"""Generate a realistic, fully synthetic B2B telecom dataset.

A telecom operator's business unit sells mobile lines, fixed broadband, cloud,
IoT and SIP trunking to companies on 12-month, 24-month or month-to-month
contracts. Tables written to data/raw/:

    customers.csv     - one row per business account (CRM)
    monthly_usage.csv - one row per account per month: lines, port-out
                        requests, data/voice usage, products, MRR (billing/OSS)
    invoices.csv      - monthly invoices with due and paid dates (finance)
    tickets.csv       - support tickets: category, priority, SLA (service desk)

Churn is simulated with a hidden monthly hazard. It peaks when a contract
reaches its renewal date and depends on segment, contract term, network
quality in the region, payment behaviour and the account manager. Most
churners show warning signs first (lines ported out, usage falling, network and
billing complaints, later payments); some leave quietly at renewal; and
regional network outages create complaint spikes in healthy accounts too, so
the model has a realistic, non-trivial job.

No real company, operator or person data is used. Run:  python -m src.generate_data
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATA_END, DATA_START, RANDOM_STATE, RAW_DIR

N_CUSTOMERS = 1500

# share, typical lines, hazard shift, ticket base rate
SEGMENTS = {
    "SME": (0.50, 15, 0.45, 0.25),
    "Corporate": (0.28, 90, 0.0, 0.45),
    "Enterprise": (0.12, 650, -0.55, 0.8),
    "Government": (0.10, 400, -0.8, 0.6),
}
CONTRACT_MIX = {  # Month-to-month, 12 months, 24 months
    "SME": (0.35, 0.40, 0.25), "Corporate": (0.15, 0.40, 0.45),
    "Enterprise": (0.05, 0.30, 0.65), "Government": (0.05, 0.55, 0.40),
}
PRODUCT_P = {  # broadband, cloud, iot, sip
    "SME": (0.50, 0.15, 0.10, 0.20), "Corporate": (0.70, 0.35, 0.25, 0.50),
    "Enterprise": (0.90, 0.60, 0.45, 0.80), "Government": (0.85, 0.40, 0.30, 0.70),
}
PRODUCT_MRR = {"broadband": 30, "cloud": 75, "iot": 50, "sip": 40}   # KWD per month
INDUSTRIES = {"Retail": 0.25, "Banking & Finance": -0.15, "Healthcare": -0.1,
              "Construction": 0.35, "Hospitality": 0.3, "Logistics": 0.1,
              "Education": -0.1, "Oil & Gas": -0.2}
REGIONS = {  # Kuwait governorates: share, outage probability per month, hazard shift
    "Capital": (0.30, 0.04, 0.0), "Hawalli": (0.20, 0.04, 0.0),
    "Farwaniya": (0.17, 0.06, 0.1), "Ahmadi": (0.15, 0.08, 0.2),
    "Mubarak Al-Kabeer": (0.08, 0.05, 0.05), "Jahra": (0.10, 0.11, 0.3),
}
ACCOUNT_MANAGERS = {"Account Manager A": -0.35, "Account Manager B": -0.1,
                    "Account Manager C": 0.0, "Account Manager D": 0.15,
                    "Account Manager E": 0.4, "Account Manager F": 0.05}
TICKET_MIX_NORMAL = {"Service request": 0.45, "Device / SIM": 0.20, "Billing dispute": 0.12,
                     "Network / Coverage": 0.15, "Data speed": 0.08}
TICKET_MIX_UNHAPPY = {"Billing dispute": 0.35, "Network / Coverage": 0.35,
                      "Data speed": 0.20, "Service request": 0.10}
SLA_DAYS = {"High": 1, "Medium": 3, "Low": 5}

NAME_A = ["Al", "Nova", "Prime", "Gulf", "Desert", "Summit", "Atlas", "Falcon",
          "Oasis", "Crescent", "Horizon", "Vertex", "Failaka", "Bay", "Palm",
          "Delta", "Apex", "Sahara", "Coral", "Granite"]
NAME_B = ["Trading", "Industries", "Solutions", "Group", "Holdings", "Logistics",
          "Partners", "Systems", "Enterprises", "Services", "Works", "Supply"]


def _company_names(rng: np.random.Generator, n: int) -> list[str]:
    names, seen = [], set()
    while len(names) < n:
        name = f"{rng.choice(NAME_A)} {rng.choice(NAME_B)} {rng.integers(10, 99)}"
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _midx(ts: pd.Timestamp) -> int:
    return ts.year * 12 + ts.month - 1


def _pick(rng, mix: dict) -> str:
    return rng.choice(list(mix), p=list(mix.values()))


def generate(n_customers: int = N_CUSTOMERS, seed: int = RANDOM_STATE) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp(DATA_START), pd.Timestamp(DATA_END)
    months = pd.date_range(start, end, freq="MS")
    # Simulate 6 months past the data end so accounts about to leave already show
    # warning signs in the last observed months (their exit is just not visible yet).
    sim_months = pd.date_range(start, end + pd.DateOffset(months=6), freq="MS")
    region_names = list(REGIONS)
    # Regional network outages hit every account in the region that month
    outages = {r: set(sim_months[rng.random(len(sim_months)) < REGIONS[r][1]]) for r in region_names}

    customers, usage, invoices, tickets = [], [], [], []
    ticket_id = 1
    names = _company_names(rng, n_customers)

    for i in range(n_customers):
        cid = f"T{i + 1:05d}"
        segment = rng.choice(list(SEGMENTS), p=[s[0] for s in SEGMENTS.values()])
        _, typical_lines, seg_shift, ticket_base = SEGMENTS[segment]
        industry = rng.choice(list(INDUSTRIES))
        region = rng.choice(region_names, p=[r[0] for r in REGIONS.values()])
        am = rng.choice(list(ACCOUNT_MANAGERS))
        term = rng.choice(["Month-to-month", "12 months", "24 months"], p=CONTRACT_MIX[segment])
        term_months = {"Month-to-month": 1, "12 months": 12, "24 months": 24}[term]
        contract_start = pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(rng.integers(0, 3100)))

        # Hidden account traits
        lines = max(3.0, typical_lines * rng.lognormal(0, 0.5))
        arpu = 10 * rng.lognormal(0, 0.2) * (0.85 if segment in ("Enterprise", "Government") else 1)
        products = {p: rng.random() < pr for p, pr in zip(PRODUCT_MRR, PRODUCT_P[segment])}
        gb_per_line = 12 * rng.lognormal(0, 0.4)
        min_per_line = 250 * rng.lognormal(0, 0.4)
        late_prop = rng.gamma(2.0, 3.0)
        price_sensitivity = rng.normal(0, 0.6)
        restructure_idx = int(rng.integers(0, len(sim_months))) if rng.random() < 0.12 else None

        first_month = max(start, contract_start.to_period("M").to_timestamp())
        active = sim_months[sim_months >= first_month]
        if first_month > end:
            continue
        start_idx = _midx(contract_start)

        # ---- Decide churn month (first month with no service) ------------------
        churn_k = None
        recent_outage = 0.0
        for k, m in enumerate(active):
            recent_outage = recent_outage * 0.6 + (1.0 if m in outages[region] else 0.0)
            months_since = _midx(m) - start_idx
            at_renewal = term_months > 1 and months_since > 0 and months_since % term_months == 0
            logit = (-6.6 + seg_shift + INDUSTRIES[industry] + ACCOUNT_MANAGERS[am]
                     + REGIONS[region][2] + 0.05 * late_prop + price_sensitivity
                     + 0.5 * recent_outage + (0.4 if months_since < 12 else 0.0))
            if term_months == 1:
                logit += 2.3
            elif at_renewal:
                logit += 4.6
            if k >= 3 and rng.random() < 1 / (1 + np.exp(-logit)):
                churn_k = k
                break

        quiet = rng.random() < 0.3          # leaves at renewal without visible warning
        decline_len = int(rng.integers(2, 6))

        for k, m in enumerate(active):
            if churn_k is not None and k >= churn_k:
                break
            p = 0.0
            if churn_k is not None and not quiet and k >= churn_k - decline_len:
                p = (k - (churn_k - decline_len) + 1) / decline_len   # 0..1 as exit nears
            outage = m in outages[region]
            sf = float(np.clip((lines / 20) ** 0.35, 1, 8))

            # Lines: organic adds/removals, port-outs when leaving, one-off restructures
            added = rng.poisson(lines * 0.02 * (1 - p))
            removed = rng.poisson(lines * 0.015) + rng.binomial(int(lines), min(0.9, 0.14 * p))
            if restructure_idx == k:
                removed += int(lines * 0.15)
            lines = max(1.0, lines + added - removed)
            port_outs = rng.poisson(0.05 + 2.2 * p * min(sf, 3))

            if p > 0.6:
                for prod in ("cloud", "iot"):
                    if products[prod] and rng.random() < 0.3:
                        products[prod] = False
            elif rng.random() < 0.008:
                prod = rng.choice(list(PRODUCT_MRR))
                products[prod] = True

            trend = 1 + 0.008 * k
            data_gb = lines * gb_per_line * trend * (1 - 0.45 * p) * (0.9 if outage else 1) \
                * rng.lognormal(0, 0.08)
            voice_min = lines * min_per_line * (1 - 0.005 * k) * (1 - 0.35 * p) * rng.lognormal(0, 0.08)
            mrr = lines * arpu + sum(PRODUCT_MRR[x] * (lines / 20) ** 0.6 for x, on in products.items() if on)

            if m > end:      # beyond the data we can see
                continue
            usage.append((cid, m, int(lines), int(added), int(removed), int(port_outs),
                          round(data_gb, 1), int(voice_min), 1 + sum(products.values()),
                          int(products["broadband"]), int(products["cloud"]), int(products["iot"]),
                          int(products["sip"]), round(mrr, 2)))

            # Support tickets
            n_normal = rng.poisson(ticket_base * sf * 0.5)
            n_outage = rng.poisson(1.2 * sf) if outage else 0
            n_unhappy = rng.poisson(1.0 * p * sf)
            dispute_this_month = False
            for kind, n in (("normal", n_normal), ("outage", n_outage), ("unhappy", n_unhappy)):
                for _ in range(n):
                    if kind == "outage":
                        cat = "Network / Coverage"
                    else:
                        cat = _pick(rng, TICKET_MIX_NORMAL if kind == "normal" else TICKET_MIX_UNHAPPY)
                    dispute_this_month |= cat == "Billing dispute"
                    high_p = 0.5 if cat == "Network / Coverage" else 0.15 + 0.3 * p
                    priority = rng.choice(["High", "Medium", "Low"],
                                          p=[high_p, (1 - high_p) * 0.6, (1 - high_p) * 0.4])
                    created = m + pd.Timedelta(days=int(rng.integers(0, m.days_in_month)))
                    res_days = int(rng.gamma(1.6, 1.5 + 4 * p + (1.5 if outage else 0)))
                    resolved = created + pd.Timedelta(days=res_days)
                    if created > end:
                        continue
                    tickets.append((ticket_id, cid, created, cat, priority,
                                    resolved if resolved <= end else pd.NaT,
                                    int(res_days > SLA_DAYS[priority])))
                    ticket_id += 1

            # Monthly invoice
            inv_date = m + pd.offsets.MonthEnd(0)
            due = inv_date + pd.Timedelta(days=30)
            late = max(0, int(rng.normal(late_prop + 12 * p + (10 if dispute_this_month else 0), 4)))
            paid = due + pd.Timedelta(days=late) - pd.Timedelta(days=int(rng.integers(0, 6)))
            invoices.append((f"INV-{cid}-{m:%Y%m}", cid, inv_date, due,
                             paid if paid <= end else pd.NaT, round(mrr, 2)))

        termination = pd.NaT
        if churn_k is not None:
            termination = active[churn_k] + pd.Timedelta(days=int(rng.integers(0, 27)))
            if termination > end:            # future exit - unknown at data end
                termination = pd.NaT
        customers.append((cid, names[i], segment, industry, region, am, term,
                          contract_start.normalize(),
                          "Active" if pd.isna(termination) else "Terminated", termination))

    return {
        "customers": pd.DataFrame(customers, columns=[
            "customer_id", "company_name", "segment", "industry", "region",
            "account_manager", "contract_term", "contract_start", "status", "termination_date"]),
        "monthly_usage": pd.DataFrame(usage, columns=[
            "customer_id", "month", "active_lines", "lines_added", "lines_disconnected",
            "port_out_requests", "data_usage_gb", "voice_minutes", "products_count",
            "has_broadband", "has_cloud", "has_iot", "has_sip", "mrr_kwd"]),
        "invoices": pd.DataFrame(invoices, columns=[
            "invoice_id", "customer_id", "invoice_date", "due_date", "paid_date", "amount_kwd"]),
        "tickets": pd.DataFrame(tickets, columns=[
            "ticket_id", "customer_id", "created_date", "category", "priority",
            "resolved_date", "sla_breached"]),
    }


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tables = generate()
    for name, df in tables.items():
        df.to_csv(RAW_DIR / f"{name}.csv", index=False)
        print(f"{name:<14} {len(df):>8,} rows -> {RAW_DIR / (name + '.csv')}")


if __name__ == "__main__":
    main()
