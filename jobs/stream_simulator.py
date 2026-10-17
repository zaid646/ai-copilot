"""Near-real-time sales event simulator.

Two roles (run together by default):
  * producer: writes synthetic sales events into raw_sales_events every N seconds
  * consumer: rolls unprocessed events into fact_sales, marks them processed

Run:  uv run python -m jobs.stream_simulator --mode both --interval 2 --limit 20
"""
from __future__ import annotations

import argparse
import random
import signal
import threading
import time

from core.db import execute, execute_many, query
from core.logging import configure_logging, get_logger

log = get_logger("jobs.stream")

CATEGORY_MARGIN = {
    "Beverages": 0.48, "Snacks": 0.44, "Produce": 0.31,
    "Dairy": 0.38, "Household": 0.46,
}

_stop = threading.Event()


def _load_reference() -> dict:
    regions = [r["region_id"] for r in query("SELECT region_id FROM dim_region")]
    products = query("SELECT product_id, category, list_price FROM dim_product")
    customers = [c["customer_id"] for c in query("SELECT customer_id FROM fact_customer WHERE is_active")]
    return {"regions": regions, "products": products, "customers": customers}


def producer(ref: dict, interval: float, stop: threading.Event) -> None:
    rng = random.Random()
    while not stop.is_set():
        batch = []
        for _ in range(rng.randint(1, 4)):
            p = rng.choice(ref["products"])
            qty = rng.randint(1, 25)
            price = round(float(p["list_price"]) * rng.uniform(0.97, 1.03), 2)
            disc = rng.choice([0.0, 0.05, 0.10])
            batch.append((
                rng.choice(ref["regions"]), p["product_id"], rng.choice(ref["customers"]),
                qty, price, disc, rng.choice(["direct", "distributor", "online"]),
            ))
        execute_many(
            "INSERT INTO raw_sales_events (region_id, product_id, customer_id, quantity, "
            "unit_price, discount_pct, channel) VALUES (%s,%s,%s,%s,%s,%s,%s)",
            batch,
        )
        log.info("produced events", n=len(batch))
        time.sleep(interval)


def _ensure_dates(dates: set) -> None:
    """Extend dim_date for any new calendar dates seen in the stream."""
    rows = []
    for d in dates:
        rows.append((
            int(d.strftime("%Y%m%d")), d, d.year, (d.month - 1) // 3 + 1, d.month,
            d.isocalendar()[1], d.weekday(), d.weekday() >= 5,
        ))
    if rows:
        execute_many(
            "INSERT INTO dim_date (date_id, full_date, year, quarter, month, week, day_of_week, is_weekend) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (date_id) DO NOTHING",
            rows,
        )


def consumer(ref: dict, interval: float, stop: threading.Event) -> None:
    price_map = {p["product_id"]: (p["category"], float(p["list_price"])) for p in ref["products"]}
    while not stop.is_set():
        rows = query(
            "SELECT event_id, event_ts, region_id, product_id, customer_id, quantity, "
            "unit_price, discount_pct, channel FROM raw_sales_events "
            "WHERE processed = false ORDER BY event_id LIMIT 200"
        )
        if rows:
            _ensure_dates({r["event_ts"].date() for r in rows})
            inserts = []
            for r in rows:
                category = price_map.get(r["product_id"], ("Unknown", 0.0))[0]
                revenue = round(r["quantity"] * float(r["unit_price"]) * (1 - float(r["discount_pct"])), 2)
                cost = round(revenue * (1 - CATEGORY_MARGIN.get(category, 0.4)), 2)
                date_id = int(r["event_ts"].strftime("%Y%m%d"))
                inserts.append((
                    f"EV{r['event_id']:08d}", date_id, r["region_id"], r["product_id"],
                    r["customer_id"] or "C00000", r["channel"], r["quantity"],
                    float(r["unit_price"]), float(r["discount_pct"]), revenue, cost,
                ))
            execute_many(
                "INSERT INTO fact_sales (order_id, date_id, region_id, product_id, customer_id, "
                "channel, quantity, unit_price, discount_pct, revenue, cost) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                inserts,
            )
            ids = [r["event_id"] for r in rows]
            execute(
                "UPDATE raw_sales_events SET processed = true, processed_at = now() "
                "WHERE event_id = ANY(%s)",
                (ids,),
            )
            log.info("consumed events", n=len(rows))
        time.sleep(interval)


def _handle_sigint(*_: object) -> None:
    _stop.set()


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["produce", "consume", "both"], default="both")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = run until Ctrl-C)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)
    ref = _load_reference()
    log.info("stream simulator starting", mode=args.mode, region_count=len(ref["regions"]))

    threads = []
    if args.mode in ("produce", "both"):
        threads.append(threading.Thread(target=producer, args=(ref, args.interval, _stop), daemon=True))
    if args.mode in ("consume", "both"):
        threads.append(threading.Thread(target=consumer, args=(ref, args.interval, _stop), daemon=True))
    for t in threads:
        t.start()

    started = time.time()
    try:
        while not _stop.is_set():
            if args.seconds and (time.time() - started) >= args.seconds:
                break
            time.sleep(0.3)
    finally:
        _stop.set()
        for t in threads:
            t.join(timeout=3)
        log.info("stream simulator stopped", ran_seconds=round(time.time() - started, 1))


if __name__ == "__main__":
    main()
