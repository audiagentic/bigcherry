-- bigcherry HIP autotune dispatch database.
--
-- Used by record mode (HI10) and the tuning engine (HI12). Production replay
-- builds never link SQLite (standards 9.1) -- winners are exported from here
-- into the compact binary cache instead.
--
-- Identity rules that this schema encodes:
--   * every durable identity is a 128-bit blake2b digest stored as BLOB(16)
--     (standards 5.4); integer primary keys are convenience only.
--   * candidates are referenced by stable_name, never by runtime id
--     (standards 2.1).
--   * a build namespace is (source_revision, manifest_hash, abi versions) so a
--     changed upstream selector or candidate set cannot silently reuse old
--     measurements (standards 13.1).

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------- schema_meta

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Schema 2 (HI48): adds the columns/tables below, matching a real recovered
-- pre-reset schema (schema_version 9 there; renumbered here since the
-- intermediate 2-8 DDL was never recovered -- see
-- docs/recovery/schema9-recovered-ddl.sql for the exact source and
-- docs/recovery/RECOVERY_TEST_LEDGER.md for what carried over vs what did
-- not). A reader must reject any schema_version it does not recognise
-- (standards: current-only, no silent migration) rather than guess at an
-- unlisted intermediate shape.
INSERT OR IGNORE INTO schema_meta(key, value) VALUES
    ('schema_version',    '2'),
    ('signature_schema',  '1'),
    ('hardware_schema',   '1');

-- --------------------------------------------------------------------- build
-- One row per (upstream revision + candidate manifest + ABI) namespace.

CREATE TABLE IF NOT EXISTS build (
    build_id            INTEGER PRIMARY KEY,
    source_revision     TEXT    NOT NULL,   -- llama.cpp git sha
    source_dirty        INTEGER NOT NULL DEFAULT 0,
    manifest_hash       TEXT    NOT NULL,   -- hex, from hip-autotune-build-hash.h
    bigcherry_revision  TEXT,               -- overlay repo sha, if known
    signature_schema    INTEGER NOT NULL,
    hardware_schema     INTEGER NOT NULL,
    variant_set         TEXT    NOT NULL,   -- inventory|workload-max|full-max|...
    rocm_version        TEXT,
    hip_version         TEXT,               -- HIP runtime version string (HI37)
    compiler            TEXT,               -- e.g. clang-18 (B3)
    build_descriptor_hash TEXT,             -- complete compiler/config identity
    dispatch_abi        TEXT,               -- artifact version string (B3)
    created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (source_revision, manifest_hash, signature_schema,
            hardware_schema, variant_set, build_descriptor_hash)
);

-- ------------------------------------------------------------------ hardware
-- Executing GPU *class*, never a device ordinal (standards 1, 10.1).

CREATE TABLE IF NOT EXISTS hardware (
    hardware_id       INTEGER PRIMARY KEY,
    hardware_digest   BLOB    NOT NULL UNIQUE,  -- blake2b(person=llama-hardware)
    architecture      TEXT    NOT NULL,         -- gfx1100, gfx1201, ...
    architecture_code INTEGER NOT NULL,
    wave_size         INTEGER NOT NULL,
    compute_units     INTEGER NOT NULL,
    feature_flags     INTEGER NOT NULL,
    canonical_json    TEXT    NOT NULL,         -- full-key collision check
    created_at        TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ----------------------------------------------------------------- signature
-- Canonical device-local operation description (standards 5).
-- canonical_json holds only hard identity + refinements; diagnostics live in
-- the observation table and never reach the digest (standards 15.1).

CREATE TABLE IF NOT EXISTS signature (
    signature_id     INTEGER PRIMARY KEY,
    signature_digest BLOB    NOT NULL UNIQUE,   -- blake2b(person=llama-hip-tune)
    base_digest      BLOB    NOT NULL,          -- digest with refinements stripped
    schema_version   INTEGER NOT NULL,
    op               TEXT    NOT NULL,          -- MUL_MAT, MUL_MAT_ID, ...
    src0_type        TEXT    NOT NULL,
    src1_type        TEXT    NOT NULL,
    dst_type         TEXT    NOT NULL,
    m                INTEGER NOT NULL,          -- device-local
    n                INTEGER NOT NULL,
    k                INTEGER NOT NULL,
    fusion           TEXT    NOT NULL DEFAULT 'none',
    is_refined       INTEGER NOT NULL DEFAULT 0,
    canonical_json   TEXT    NOT NULL,
    created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS signature_base_idx ON signature(base_digest);

-- ----------------------------------------------------------------- candidate
-- Mirror of the manifest. stable_name is the durable identity (standards 2.1).

CREATE TABLE IF NOT EXISTS candidate (
    candidate_id           INTEGER PRIMARY KEY,
    build_id               INTEGER NOT NULL REFERENCES build(build_id),
    stable_name            TEXT    NOT NULL,
    family                 TEXT    NOT NULL,   -- mmq|mmvf|mmf|mmvq|blas
    source_class           TEXT    NOT NULL,   -- see standards 2.3
    implementation_version INTEGER NOT NULL,
    architectures          TEXT    NOT NULL,   -- json array
    architecture_mask      INTEGER NOT NULL,
    graph_safe             INTEGER NOT NULL,
    deterministic          INTEGER NOT NULL,
    config_json            TEXT    NOT NULL,
    UNIQUE (build_id, stable_name),
    CHECK (family IN ('mmq','mmvf','mmf','mmvq','blas')),
    CHECK (source_class IN ('native_wrapper','existing_runtime',
                            'existing_alternative','new_generated_variant',
                            'vendor_auto','vendor_explicit'))
);

-- --------------------------------------------------------------- observation
-- Record mode. One row per (build, hardware, signature); duplicates merge by
-- incrementing calls and appending to sites_json (standards 15.1).

CREATE TABLE IF NOT EXISTS observation (
    observation_id      INTEGER PRIMARY KEY,
    build_id            INTEGER NOT NULL REFERENCES build(build_id),
    hardware_id         INTEGER NOT NULL REFERENCES hardware(hardware_id),
    signature_id        INTEGER NOT NULL REFERENCES signature(signature_id),
    native_stable_name  TEXT    NOT NULL,
    calls               INTEGER NOT NULL DEFAULT 0,
    est_bytes           INTEGER NOT NULL DEFAULT 0,
    est_flops           INTEGER NOT NULL DEFAULT 0,
    sites_json          TEXT    NOT NULL DEFAULT '[]',
    diagnostics_json    TEXT    NOT NULL DEFAULT '{}',
    first_seen          TEXT    NOT NULL DEFAULT (datetime('now')),
    last_seen           TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (build_id, hardware_id, signature_id)
);

CREATE INDEX IF NOT EXISTS observation_hot_idx
    ON observation(build_id, hardware_id, calls DESC);

-- ------------------------------------------------------------------ tuning_run
-- One row per tuning execution (HI37: build != execution -- two tunes of the
-- same build under different workloads/conditions must coexist).

CREATE TABLE IF NOT EXISTS tuning_run (
    run_id                 INTEGER PRIMARY KEY,
    build_id               INTEGER NOT NULL REFERENCES build(build_id),
    run_digest             BLOB    NOT NULL UNIQUE,
    workload_digest        BLOB,
    workload_label         TEXT,
    started_at             TEXT,
    finished_at            TEXT,
    host_sync_overhead_us  REAL,
    config_json            TEXT    NOT NULL DEFAULT '{}',
    machine_json           TEXT    NOT NULL DEFAULT '{}',
    created_at             TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- --------------------------------------------------------------- measurement
-- One row per (dispatch key, candidate, objective) tuning attempt.
-- Rejected candidates are recorded too, with a reason (standards 7.2).

CREATE TABLE IF NOT EXISTS measurement (
    measurement_id   INTEGER PRIMARY KEY,
    build_id         INTEGER NOT NULL REFERENCES build(build_id),
    hardware_id      INTEGER NOT NULL REFERENCES hardware(hardware_id),
    signature_id     INTEGER REFERENCES signature(signature_id), -- NULL when only dispatch_digest known
    dispatch_digest  BLOB,                                    -- lookup key (standards 5.4)
    candidate_id     INTEGER NOT NULL REFERENCES candidate(candidate_id),
    run_id           INTEGER REFERENCES tuning_run(run_id),
    objective        TEXT    NOT NULL DEFAULT 'latency',
    stage            TEXT    NOT NULL,          -- screen|final
    accepted         INTEGER NOT NULL,
    reject_reason    TEXT,                      -- null when accepted
    samples          INTEGER NOT NULL DEFAULT 0,
    launches_per_sample INTEGER,               -- HI34: adaptive shared batch size for this signature
    median_us        REAL,
    gpu_mad_us       REAL,                       -- MAD of GPU times (B3)
    p95_us           REAL,
    host_median_us   REAL,                       -- host clock around launch (B3)
    min_us           REAL,
    stddev_us        REAL,
    workspace_bytes  INTEGER NOT NULL DEFAULT 0,  -- candidate's DECLARED upper bound
    pool_peak_bytes  INTEGER,                    -- HI52/HI40: MEASURED peak pool bytes; NULL = not captured
    nmse             REAL,
    max_abs_err      REAL,
    max_rel_err      REAL,
    samples_json     TEXT,                      -- raw sample array
    effective_us     REAL,                      -- HI50: ranking metric (max(gpu, host-sync-adjusted))
    measured_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (build_id, hardware_id, candidate_id, objective, stage, dispatch_digest)
);

CREATE INDEX IF NOT EXISTS measurement_dispatch_idx
    ON measurement(build_id, dispatch_digest, objective);

CREATE INDEX IF NOT EXISTS measurement_lookup_idx
    ON measurement(build_id, hardware_id, signature_id, objective);

-- -------------------------------------------------------------------- winner
-- Result of winner selection (standards 7.3). dispatch_digest is the exported
-- lookup key: blake2b(person=llama-dispatch) over
-- (software namespace, hardware key, signature, objective).

CREATE TABLE IF NOT EXISTS winner (
    winner_id           INTEGER PRIMARY KEY,
    build_id            INTEGER NOT NULL REFERENCES build(build_id),
    hardware_id         INTEGER NOT NULL REFERENCES hardware(hardware_id),
    signature_id        INTEGER REFERENCES signature(signature_id), -- NULL when only dispatch_digest known
    objective           TEXT    NOT NULL DEFAULT 'latency',
    dispatch_digest     BLOB    NOT NULL,
    candidate_id        INTEGER NOT NULL REFERENCES candidate(candidate_id),
    run_id              INTEGER REFERENCES tuning_run(run_id),
    stable_name         TEXT    NOT NULL,
    native_stable_name  TEXT    NOT NULL,
    is_native           INTEGER NOT NULL,
    improvement_pct     REAL    NOT NULL DEFAULT 0.0,
    median_us           REAL    NOT NULL,
    p95_us              REAL    NOT NULL,
    workspace_bytes     INTEGER NOT NULL DEFAULT 0,
    pool_peak_bytes     INTEGER,
    reason              TEXT,                      -- text explanation of why winner chosen (B3)
    confidence          REAL,
    seeded              INTEGER NOT NULL DEFAULT 0,  -- 1 = manual seed (HI11)
    validated           INTEGER NOT NULL DEFAULT 0,  -- set by HI14
    promotion_status    TEXT,                      -- HI34/HI50: native|pending_bh|confirmation_rejected|promoted|rejected_bh
    q_value             REAL,                      -- HI34: BH-adjusted q-value, set only after tune-promote
    decided_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (build_id, hardware_id, objective, dispatch_digest)
);

CREATE INDEX IF NOT EXISTS winner_improvement_idx
    ON winner(build_id, improvement_pct DESC);

CREATE UNIQUE INDEX IF NOT EXISTS winner_dispatch_idx
    ON winner(dispatch_digest, objective);

-- --------------------------------------------------------------- replay_miss
-- Bounded miss log written by replay builds via GGML_HIP_DISPATCH_MISS.

CREATE TABLE IF NOT EXISTS replay_miss (
    miss_id          INTEGER PRIMARY KEY,
    build_id         INTEGER NOT NULL REFERENCES build(build_id),
    hardware_id      INTEGER NOT NULL REFERENCES hardware(hardware_id),
    signature_digest BLOB    NOT NULL,
    dispatch_digest  BLOB    NOT NULL,
    canonical_json   TEXT    NOT NULL,
    fallback_name    TEXT    NOT NULL,
    calls            INTEGER NOT NULL DEFAULT 1,
    first_seen       TEXT    NOT NULL DEFAULT (datetime('now')),
    last_seen        TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (build_id, hardware_id, dispatch_digest)
);

-- ----------------------------------------------------------------- blacklist
-- Candidates proven invalid/high-spill on a given architecture (HI15).

CREATE TABLE IF NOT EXISTS candidate_blacklist (
    stable_name  TEXT NOT NULL,
    architecture TEXT NOT NULL,
    reason       TEXT NOT NULL,
    added_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (stable_name, architecture)
);

-- ------------------------------------------------------------------ device_state
-- HI52 part 2: falsification only, never a ranking axis (see
-- hip-autotune-smi.h). One row per (run, dispatch, phase) when
-- GGML_HIP_TUNE_SMI=1 captured a valid snapshot.

CREATE TABLE IF NOT EXISTS device_state (
    device_state_id  INTEGER PRIMARY KEY,
    run_id           INTEGER REFERENCES tuning_run(run_id),
    dispatch_digest  BLOB    NOT NULL,
    phase            TEXT    NOT NULL,          -- pre|post
    sclk_mhz         INTEGER,
    mclk_mhz         INTEGER,
    edge_temp_mc     INTEGER,
    junction_temp_mc INTEGER,
    socket_power_uw  INTEGER,
    busy_percent     INTEGER,
    UNIQUE (run_id, dispatch_digest, phase)
);

-- ------------------------------------------------------------- ranking_decision
-- HI50: every compiled-in policy's full per-signature verdict, not just the
-- production policy's -- see ranking_decisions_json in the tuner and
-- ranking_policy.py/rank_replay.py on the Python side.

CREATE TABLE IF NOT EXISTS ranking_decision (
    decision_id                   INTEGER PRIMARY KEY,
    build_id                      INTEGER NOT NULL REFERENCES build(build_id),
    hardware_id                   INTEGER NOT NULL REFERENCES hardware(hardware_id),
    signature_id                  INTEGER REFERENCES signature(signature_id),
    dispatch_digest                BLOB    NOT NULL,
    run_id                        INTEGER REFERENCES tuning_run(run_id),
    policy_name                   TEXT    NOT NULL,
    policy_version                INTEGER NOT NULL,
    is_production                 INTEGER NOT NULL DEFAULT 0,
    predicted_winner_candidate_id INTEGER NOT NULL REFERENCES candidate(candidate_id),
    decided_at                    TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (build_id, hardware_id, dispatch_digest, run_id, policy_name)
);

CREATE TABLE IF NOT EXISTS ranking_decision_candidate (
    decision_id      INTEGER NOT NULL REFERENCES ranking_decision(decision_id),
    candidate_id     INTEGER NOT NULL REFERENCES candidate(candidate_id),
    effective_us     REAL,
    rank             INTEGER,             -- position within this policy's ranking; NULL if never ranked
    verdict          TEXT    NOT NULL,    -- winner|qualified|near_tie_below_threshold|outside_tie_band|not_attempted|rejected
    rejection_reason TEXT,
    PRIMARY KEY (decision_id, candidate_id)
);
