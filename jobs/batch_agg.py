"""Batch KPI aggregation job.

Daily (cron) KPI roll-up from fact_sales into agg_daily_kpi, by
(date, region, category). Idempotent: re-running upserts.

Run:  uv run python -m jobs.batch_agg [--full]
"""
from __future__ import annotations

import argparse

from core.db import execute, query_one
from core.logging import configure_logging, get_logger

log = get_logger("jobs.batch_agg")

UPSERT = """
INSERT INTO agg_daily_kpi
    (kpi_date, region_id, category, revenue, units, orders, avg_order_value, gross_margin, computed_at)
SELECT
    dd.full_date,
    f.region_id,
    p.category,
    round(sum(f.revenue), 2)                              AS revenue,
    sum(f.quantity)                                       AS units,
    count(DISTINCT f.order_id)                            AS orders,
    round(sum(f.revenue) / nullif(count(DISTINCT f.order_id), 0), 2) AS avg_order_value,
    round(sum(f.revenue - f.cost), 2)                     AS gross_margin,
    now()
FROM fact_sales f
JOIN dim_date dd   ON f.date_id = dd.date_id
JOIN dim_product p ON f.product_id = p.product_id
GROUP BY dd.full_date, f.region_id, p.category
ON CONFLICT (kpi_date, region_id, category) DO UPDATE SET
    revenue         = EXCLUDED.revenue,
    units           = EXCLUDED.units,
    orders          = EXCLUDED.orders,
    avg_order_value = EXCLUDED.avg_order_value,
    gross_margin    = EXCLUDED.gross_margin,
    computed_at     = now()
"""

INCREMENTAL = """
INSERT INTO agg_daily_kpi
    (kpi_date, region_id, category, revenue, units, orders, avg_order_value, gross_margin, computed_at)
SELECT dd.full_date, f.region_id, p.category,
       round(sum(f.revenue),2), sum(f.quantity), count(DISTINCT f.order_id),
       round(sum(f.revenue)/nullif(count(DISTINCT f.order_id),0),2),
       round(sum(f.revenue-f.cost),2), now()
FROM fact_sales f
JOIN dim_date dd   ON f.date_id = dd.date_id
JOIN dim_product p ON f.product_id = p.product_id
WHERE dd.full_date NOT IN (SELECT DISTINCT kpi_date FROM agg_daily_kpi)
GROUP BY dd.full_date, f.region_id, p.category
ON CONFLICT (kpi_date, region_id, category) DO UPDATE SET
    revenue=EXCLUDED.revenue, units=EXCLUDED.units, orders=EXCLUDED.orders,
    avg_order_value=EXCLUDED.avg_order_value, gross_margin=EXCLUDED.gross_margin,
    computed_at=now()
"""


def run(full: bool = False) -> int:
    if full:
        rows = execute(UPSERT)
    else:
        rows = execute(INCREMENTAL)
    total = query_one("SELECT count(*) AS n FROM agg_daily_kpi")
    log.info("batch_agg complete", upserted=rows, total_kpi_rows=total["n"])
    return rows


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="rebuild all dates")
    args = parser.parse_args()
    run(full=args.full)


if __name__ == "__main__":
    main()
