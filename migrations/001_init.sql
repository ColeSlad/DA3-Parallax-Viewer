CREATE TABLE IF NOT EXISTS jobs (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    status      text NOT NULL DEFAULT 'queued',  -- queued|running|succeeded|failed
    input_keys  text[] NOT NULL,
    result_key  text,
    point_count int,
    view_count  int,
    duration_ms int,
    error       text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS jobs_status_created_at ON jobs (status, created_at);
