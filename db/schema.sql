-- AI Copilot warehouse schema (Postgres 16)
-- Idempotent: safe to re-apply on every startup / seed run.

-- =========================
-- Dimensions
-- =========================
CREATE TABLE IF NOT EXISTS dim_region (
    region_id     TEXT PRIMARY KEY,
    region_name   TEXT NOT NULL,
    country       TEXT NOT NULL DEFAULT 'US',
    manager       TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dim_product (
    product_id    TEXT PRIMARY KEY,
    product_name  TEXT NOT NULL,
    category      TEXT NOT NULL,
    subcategory   TEXT,
    list_price    NUMERIC(12,2) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dim_date (
    date_id       INTEGER PRIMARY KEY,          -- yyyymmdd
    full_date     DATE NOT NULL UNIQUE,
    year          INTEGER NOT NULL,
    quarter       INTEGER NOT NULL,
    month         INTEGER NOT NULL,
    week          INTEGER NOT NULL,
    day_of_week   INTEGER NOT NULL,
    is_weekend    BOOLEAN NOT NULL
);

-- =========================
-- Facts
-- =========================
CREATE TABLE IF NOT EXISTS fact_sales (
    sale_id       BIGSERIAL PRIMARY KEY,
    order_id      TEXT NOT NULL,
    date_id       INTEGER NOT NULL REFERENCES dim_date(date_id),
    region_id     TEXT NOT NULL REFERENCES dim_region(region_id),
    product_id    TEXT NOT NULL REFERENCES dim_product(product_id),
    customer_id   TEXT NOT NULL,
    channel       TEXT NOT NULL DEFAULT 'direct',
    quantity      INTEGER NOT NULL,
    unit_price    NUMERIC(12,2) NOT NULL,
    discount_pct  NUMERIC(5,4) NOT NULL DEFAULT 0,
    revenue       NUMERIC(14,2) NOT NULL,
    cost          NUMERIC(14,2) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_fact_sales_date ON fact_sales(date_id);
CREATE INDEX IF NOT EXISTS idx_fact_sales_region ON fact_sales(region_id);
CREATE INDEX IF NOT EXISTS idx_fact_sales_product ON fact_sales(product_id);
CREATE INDEX IF NOT EXISTS idx_fact_sales_customer ON fact_sales(customer_id);

CREATE TABLE IF NOT EXISTS fact_finance (
    finance_id    BIGSERIAL PRIMARY KEY,
    date_id       INTEGER NOT NULL REFERENCES dim_date(date_id),
    region_id     TEXT NOT NULL REFERENCES dim_region(region_id),
    account       TEXT NOT NULL,               -- revenue | marketing_spend | cogs | opex
    budget        NUMERIC(14,2) NOT NULL,
    actual        NUMERIC(14,2) NOT NULL,
    variance      NUMERIC(14,2) GENERATED ALWAYS AS (actual - budget) STORED,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_fact_finance_date ON fact_finance(date_id);
CREATE INDEX IF NOT EXISTS idx_fact_finance_region ON fact_finance(region_id);

CREATE TABLE IF NOT EXISTS fact_customer (
    customer_id      TEXT PRIMARY KEY,
    customer_name    TEXT NOT NULL,
    region_id        TEXT NOT NULL REFERENCES dim_region(region_id),
    segment          TEXT NOT NULL,            -- enterprise | smb | retail
    signup_date      DATE NOT NULL,
    tenure_months    INTEGER NOT NULL,
    lifetime_value   NUMERIC(14,2) NOT NULL,
    churn_flag       BOOLEAN NOT NULL DEFAULT false,
    churn_quarter    TEXT,                     -- e.g. 2026Q3 when churned
    last_order_date  DATE,
    is_active        BOOLEAN NOT NULL DEFAULT true,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =========================
-- Near-real-time stream
-- =========================
CREATE TABLE IF NOT EXISTS raw_sales_events (
    event_id      BIGSERIAL PRIMARY KEY,
    event_ts      TIMESTAMPTZ NOT NULL DEFAULT now(),
    region_id     TEXT NOT NULL,
    product_id    TEXT NOT NULL,
    customer_id   TEXT,
    quantity      INTEGER NOT NULL,
    unit_price    NUMERIC(12,2) NOT NULL,
    discount_pct  NUMERIC(5,4) NOT NULL DEFAULT 0,
    channel       TEXT NOT NULL DEFAULT 'direct',
    processed     BOOLEAN NOT NULL DEFAULT false,
    processed_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_raw_events_unprocessed ON raw_sales_events(processed) WHERE processed = false;

-- =========================
-- Batch aggregate
-- =========================
CREATE TABLE IF NOT EXISTS agg_daily_kpi (
    kpi_date      DATE NOT NULL,
    region_id     TEXT NOT NULL REFERENCES dim_region(region_id),
    category      TEXT NOT NULL,
    revenue       NUMERIC(14,2) NOT NULL,
    units         INTEGER NOT NULL,
    orders        INTEGER NOT NULL,
    avg_order_value NUMERIC(14,2) NOT NULL,
    gross_margin  NUMERIC(14,2) NOT NULL,
    computed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (kpi_date, region_id, category)
);

-- =========================
-- Copilot persistence
-- =========================
CREATE TABLE IF NOT EXISTS users (
    user_id       BIGSERIAL PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('analyst','admin')),
    tenant_id     TEXT NOT NULL DEFAULT 'default',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id BIGSERIAL PRIMARY KEY,
    user_id         BIGINT NOT NULL REFERENCES users(user_id),
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    title           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_conversations_tenant ON conversations(tenant_id);

CREATE TABLE IF NOT EXISTS messages (
    message_id      BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT NOT NULL REFERENCES conversations(conversation_id),
    tenant_id       TEXT NOT NULL DEFAULT 'default',
    role            TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
    content         TEXT NOT NULL,
    trace_id        TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_tenant ON messages(tenant_id);

CREATE TABLE IF NOT EXISTS agent_traces (
    trace_id      TEXT PRIMARY KEY,
    user_id       BIGINT REFERENCES users(user_id),
    tenant_id     TEXT NOT NULL DEFAULT 'default',
    question      TEXT NOT NULL,
    answer        TEXT,
    status        TEXT NOT NULL DEFAULT 'running',  -- running | completed | failed | blocked
    guardrail_blocked BOOLEAN NOT NULL DEFAULT false,
    total_steps   INTEGER NOT NULL DEFAULT 0,
    latency_ms    INTEGER,
    tokens_used   INTEGER,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS agent_steps (
    step_id       BIGSERIAL PRIMARY KEY,
    trace_id      TEXT NOT NULL REFERENCES agent_traces(trace_id),
    seq           INTEGER NOT NULL,
    node          TEXT NOT NULL,
    input_json    JSONB,
    output_json   JSONB,
    latency_ms    INTEGER,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (trace_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_agent_steps_trace ON agent_steps(trace_id);

CREATE TABLE IF NOT EXISTS guardrail_events (
    event_id      BIGSERIAL PRIMARY KEY,
    trace_id      TEXT,
    tenant_id     TEXT NOT NULL DEFAULT 'default',
    direction     TEXT NOT NULL CHECK (direction IN ('ingress','egress')),
    rule          TEXT NOT NULL,
    action        TEXT NOT NULL CHECK (action IN ('passed','redacted','blocked')),
    reason        TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_guardrail_trace ON guardrail_events(trace_id);
CREATE INDEX IF NOT EXISTS idx_guardrail_tenant ON guardrail_events(tenant_id);
CREATE INDEX IF NOT EXISTS idx_agent_traces_tenant ON agent_traces(tenant_id);

CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id     BIGSERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    endpoint      TEXT NOT NULL,
    model_version TEXT NOT NULL,
    latency_ms    INTEGER NOT NULL,
    score         NUMERIC(12,4),
    inputs_json   JSONB
);
