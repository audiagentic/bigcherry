-- BigCherry dispatch DB schema 2 -> 3.
-- Historical rows remain readable for diagnostics but have NULL campaign
-- identity until explicitly regenerated; production campaign writes must fill
-- the new namespace fields.
PRAGMA foreign_keys = ON;
BEGIN;

ALTER TABLE build ADD COLUMN source_slice_id TEXT;
ALTER TABLE build ADD COLUMN build_plan_id TEXT;
ALTER TABLE build ADD COLUMN effective_build_id TEXT;
ALTER TABLE build ADD COLUMN campaign_run_id TEXT;
ALTER TABLE build ADD COLUMN workload_id TEXT;
ALTER TABLE observation ADD COLUMN source_slice_id TEXT;
ALTER TABLE observation ADD COLUMN workload_id TEXT;
ALTER TABLE observation ADD COLUMN campaign_run_id TEXT;
ALTER TABLE measurement ADD COLUMN source_slice_id TEXT;
ALTER TABLE measurement ADD COLUMN build_plan_id TEXT;
ALTER TABLE measurement ADD COLUMN effective_build_id TEXT;
ALTER TABLE measurement ADD COLUMN workload_id TEXT;
ALTER TABLE measurement ADD COLUMN campaign_run_id TEXT;
ALTER TABLE winner ADD COLUMN source_slice_id TEXT;
ALTER TABLE winner ADD COLUMN build_plan_id TEXT;
ALTER TABLE winner ADD COLUMN effective_build_id TEXT;
ALTER TABLE winner ADD COLUMN workload_id TEXT;
ALTER TABLE winner ADD COLUMN campaign_run_id TEXT;
ALTER TABLE replay_miss ADD COLUMN source_slice_id TEXT;
ALTER TABLE replay_miss ADD COLUMN workload_id TEXT;
ALTER TABLE replay_miss ADD COLUMN campaign_run_id TEXT;

CREATE INDEX IF NOT EXISTS build_campaign_identity_idx
    ON build(source_slice_id, build_plan_id, effective_build_id);
CREATE INDEX IF NOT EXISTS measurement_campaign_identity_idx
    ON measurement(source_slice_id, workload_id, campaign_run_id);
CREATE INDEX IF NOT EXISTS winner_campaign_identity_idx
    ON winner(source_slice_id, workload_id, campaign_run_id);

UPDATE schema_meta SET value = '3' WHERE key = 'schema_version';
COMMIT;
