"""Seed the warehouse with reproducible demo data.

Run:  uv run python -m db.seed

Deterministic (seed=42). Injects a *discoverable* root cause for the pilot
question: West-region revenue drops ~25% in 2026Q3 driven by (a) volume loss
in Beverages/Produce, (b) enterprise/SMB churn, (c) marketing underspend.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from core.config import ROOT
from core.db import connection, execute, execute_many, query, query_one
from core.logging import configure_logging, get_logger

log = get_logger("db.seed")

RNG = random.Random(42)

REGIONS = [
    ("WEST", "West", "Maya Chen"),
    ("EAST", "East", "Daniel Ortiz"),
    ("CENTRAL", "Central", "Priya Nair"),
    ("SOUTH", "South", "Luis Alvarez"),
]

# product_id, name, category, subcategory, list_price, margin
PRODUCTS = [
    ("P100", "Aurora Cola 24pk", "Beverages", "Carbonated", 18.99, 0.52),
    ("P101", "Spring Water 12pk", "Beverages", "Water", 9.49, 0.44),
    ("P110", "Kettle Chips 6pk", "Snacks", "Chips", 12.75, 0.48),
    ("P111", "Trail Mix 500g", "Snacks", "Nuts", 15.20, 0.41),
    ("P120", "Organic Avocado 4pk", "Produce", "Fresh Fruit", 21.50, 0.33),
    ("P121", "Baby Spinach 300g", "Produce", "Leafy Greens", 11.10, 0.30),
    ("P130", "Greek Yogurt 1kg", "Dairy", "Yogurt", 13.40, 0.38),
    ("P140", "Paper Towels 12pk", "Household", "Paper", 16.80, 0.46),
]

SEGMENTS = ["enterprise", "smb", "retail"]
CHANNELS = ["direct", "distributor", "online"]

# 18 months of history ending 2026-09-30
START = date(2025, 4, 1)
END = date(2026, 9, 30)

MONTH_SEASONALITY = {1: 0.92, 2: 0.90, 3: 0.97, 4: 1.00, 5: 1.02, 6: 1.05,
                     7: 0.98, 8: 1.01, 9: 1.03, 10: 1.10, 11: 1.22, 12: 1.30}

# The injected shock: West, 2026Q3, these categories lose volume.
SHOCK_QUARTER = (2026, 3)
SHOCK_REGION = "WEST"
SHOCK_CATEGORIES = {"Beverages", "Produce"}
SHOCK_VOLUME_MULT = 0.68


def _quarter(d: date) -> int:
    return (d.month - 1) // 3 + 1


def apply_schema() -> None:
    schema = (ROOT / "db" / "schema.sql").read_text()
    with connection() as conn, conn.cursor() as cur:
        cur.execute(schema)
        conn.commit()
    log.info("schema applied")


def truncate_all() -> None:
    execute(
        "TRUNCATE fact_sales, fact_finance, fact_customer, raw_sales_events, "
        "agg_daily_kpi, agent_steps, agent_traces, guardrail_events, messages, "
        "conversations, model_metrics, dim_region, dim_product, dim_date, users "
        "RESTART IDENTITY CASCADE"
    )
    log.info("tables truncated")


def seed_dates() -> None:
    rows = []
    d = START
    while d <= END:
        rows.append((
            int(d.strftime("%Y%m%d")), d, d.year, _quarter(d), d.month,
            d.isocalendar()[1], d.weekday(), d.weekday() >= 5,
        ))
        d += timedelta(days=1)
    execute_many(
        "INSERT INTO dim_date (date_id, full_date, year, quarter, month, week, day_of_week, is_weekend) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
        rows,
    )
    log.info("seeded dates", count=len(rows))


def seed_regions_products() -> None:
    execute_many(
        "INSERT INTO dim_region (region_id, region_name, manager) VALUES (%s,%s,%s)",
        [(r[0], r[1], r[2]) for r in REGIONS],
    )
    execute_many(
        "INSERT INTO dim_product (product_id, product_name, category, subcategory, list_price) "
        "VALUES (%s,%s,%s,%s,%s)",
        [(p[0], p[1], p[2], p[3], p[4]) for p in PRODUCTS],
    )
    log.info("seeded regions + products")


def seed_customers() -> None:
    rows = []
    cust_no = 0
    # churn pool: which customers churn in the shock quarter
    for region_id, _, _ in REGIONS:
        n = 160
        for _ in range(n):
            cust_no += 1
            customer_id = f"C{cust_no:05d}"
            segment = RNG.choices(SEGMENTS, weights=[0.15, 0.35, 0.50])[0]
            signup = date(2022, 1, 1) + timedelta(days=RNG.randint(0, 1200))
            tenure = max(1, int((END - signup).days / 30))
            ltv_base = {"enterprise": 420000, "smb": 95000, "retail": 12000}[segment]
            ltv = round(ltv_base * RNG.uniform(0.5, 1.8), 2)

            churned = False
            churn_q = None
            last_order = END - timedelta(days=RNG.randint(0, 20))
            if region_id == SHOCK_REGION and segment in ("enterprise", "smb") and RNG.random() < 0.10:
                churned = True
                churn_q = "2026Q3"
                last_order = date(2026, 7, RNG.randint(1, 25))

            rows.append((
                customer_id, f"Customer {cust_no:05d}", region_id, segment, signup,
                tenure, ltv, churned, churn_q, last_order, not churned,
            ))
    execute_many(
        "INSERT INTO fact_customer (customer_id, customer_name, region_id, segment, signup_date, "
        "tenure_months, lifetime_value, churn_flag, churn_quarter, last_order_date, is_active) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        rows,
    )
    log.info("seeded customers", count=len(rows))
    active_west = query_one(
        "SELECT count(*) AS n FROM fact_customer WHERE region_id='WEST' AND churn_flag"
    )
    log.info("west churned customers", count=active_west["n"])


def seed_sales() -> None:
    customers_by_region: dict[str, list[str]] = {}
    for r in REGIONS:
        ids = query(
            "SELECT customer_id FROM fact_customer WHERE region_id=%s AND is_active", (r[0],)
        )
        customers_by_region[r[0]] = [x["customer_id"] for x in ids]

    rows = []
    order_seq = 0
    d = START
    while d <= END:
        date_id = int(d.strftime("%Y%m%d"))
        season = MONTH_SEASONALITY[d.month]
        weekend_mult = 0.55 if d.weekday() >= 5 else 1.0
        for region_id, _, _ in REGIONS:
            for product_id, _name, category, _sub, price, margin in PRODUCTS:
                if RNG.random() > 0.72:
                    continue
                qty_mean = 14 * season * weekend_mult
                if (d.year, _quarter(d)) == SHOCK_QUARTER and region_id == SHOCK_REGION \
                        and category in SHOCK_CATEGORIES:
                    qty_mean *= SHOCK_VOLUME_MULT
                qty = max(1, int(RNG.gauss(qty_mean, qty_mean * 0.3)))
                discount = round(RNG.choice([0.0, 0.05, 0.08, 0.10, 0.15]), 4)
                unit_price = round(price * RNG.uniform(0.97, 1.03), 2)
                revenue = round(qty * unit_price * (1 - discount), 2)
                cost = round(revenue * (1 - margin) * RNG.uniform(0.97, 1.03), 2)
                order_seq += 1
                cust = RNG.choice(customers_by_region[region_id])
                rows.append((
                    f"SO{order_seq:07d}", date_id, region_id, product_id, cust,
                    RNG.choice(CHANNELS), qty, unit_price, discount, revenue, cost,
                ))
        d += timedelta(days=1)

    execute_many(
        "INSERT INTO fact_sales (order_id, date_id, region_id, product_id, customer_id, channel, "
        "quantity, unit_price, discount_pct, revenue, cost) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        rows,
    )
    log.info("seeded sales rows", count=len(rows))


def seed_finance() -> None:
    rows = []
    accounts = [("revenue", 1.0), ("marketing_spend", 0.06), ("cogs", 0.55), ("opex", 0.18)]
    for region_id, _, _ in REGIONS:
        # monthly revenue baseline per region
        d = date(START.year, START.month, 1)
        while d <= END:
            month_days = (date(d.year + (d.month == 12), d.month % 12 + 1, 1) - d).days
            rev = (
                query_one(
                    "SELECT COALESCE(sum(revenue),0) AS r FROM fact_sales f JOIN dim_date dd "
                    "ON f.date_id=dd.date_id WHERE f.region_id=%s AND dd.full_date >= %s "
                    "AND dd.full_date < %s",
                    (region_id, d, d + timedelta(days=month_days)),
                )["r"]
            ) or 0
            for account, pct in accounts:
                if account == "revenue":
                    actual = float(rev)
                    budget = round(actual * RNG.uniform(0.98, 1.12) * 1.05, 2)
                else:
                    actual = round(float(rev) * pct, 2)
                    budget = round(float(rev) * pct * 1.05, 2)
                # shock: West marketing underspend in 2026Q3
                if account == "marketing_spend" and region_id == SHOCK_REGION \
                        and _quarter(d) == 3 and d.year == 2026:
                    actual = round(actual * 0.55, 2)
                rows.append((
                    int(d.strftime("%Y%m%d")), region_id, account,
                    round(budget, 2), round(actual, 2),
                ))
            d = d + timedelta(days=month_days)
    execute_many(
        "INSERT INTO fact_finance (date_id, region_id, account, budget, actual) "
        "VALUES (%s,%s,%s,%s,%s)",
        rows,
    )
    log.info("seeded finance rows", count=len(rows))


def seed_users() -> None:
    from core.security import hash_password

    users = [
        ("analyst", hash_password("analyst123"), "analyst"),
        ("admin", hash_password("admin123"), "admin"),
    ]
    execute_many(
        "INSERT INTO users (username, password_hash, role) VALUES (%s,%s,%s) "
        "ON CONFLICT (username) DO NOTHING",
        users,
    )
    log.info("seeded users", users=[u[0] for u in users])


def main() -> None:
    configure_logging()
    apply_schema()
    truncate_all()
    seed_dates()
    seed_regions_products()
    seed_customers()
    seed_sales()
    seed_finance()
    seed_users()
    total = query_one("SELECT count(*) AS n FROM fact_sales")
    log.info("seed complete", fact_sales=total["n"])


if __name__ == "__main__":
    main()
