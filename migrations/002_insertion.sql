-- Extend jobs table to support both reconstruction and insertion job kinds.
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS kind text NOT NULL DEFAULT 'reconstruction';
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS parent_id uuid REFERENCES jobs(id);
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS params jsonb;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS splat_key text;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS meta jsonb;

CREATE INDEX IF NOT EXISTS jobs_parent_id ON jobs (parent_id) WHERE parent_id IS NOT NULL;
