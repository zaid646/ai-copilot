-- Tenant isolation for conversational + audit tables.
-- Safe to apply after db/schema.sql (default tenant keeps existing rows visible).
ALTER TABLE users             ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';
ALTER TABLE conversations     ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';
ALTER TABLE messages          ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';
ALTER TABLE agent_traces      ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';
ALTER TABLE guardrail_events  ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';

CREATE INDEX IF NOT EXISTS idx_conversations_tenant ON conversations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_messages_tenant ON messages(tenant_id);
CREATE INDEX IF NOT EXISTS idx_agent_traces_tenant ON agent_traces(tenant_id);
CREATE INDEX IF NOT EXISTS idx_guardrail_tenant ON guardrail_events(tenant_id);

-- Application queries always pass tenant_id. RLS is provided as defence in
-- depth for deployments that connect with a non-owner role.
ALTER TABLE conversations    ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages         ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_traces     ENABLE ROW LEVEL SECURITY;
ALTER TABLE guardrail_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation_conversations ON conversations;
CREATE POLICY tenant_isolation_conversations ON conversations
  USING (tenant_id = COALESCE(current_setting('app.tenant_id', true), 'default'));

DROP POLICY IF EXISTS tenant_isolation_messages ON messages;
CREATE POLICY tenant_isolation_messages ON messages
  USING (tenant_id = COALESCE(current_setting('app.tenant_id', true), 'default'));

DROP POLICY IF EXISTS tenant_isolation_traces ON agent_traces;
CREATE POLICY tenant_isolation_traces ON agent_traces
  USING (tenant_id = COALESCE(current_setting('app.tenant_id', true), 'default'));

DROP POLICY IF EXISTS tenant_isolation_guardrails ON guardrail_events;
CREATE POLICY tenant_isolation_guardrails ON guardrail_events
  USING (tenant_id = COALESCE(current_setting('app.tenant_id', true), 'default'));
