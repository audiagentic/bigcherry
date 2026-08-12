CREATE TABLE schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE build (
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
    dispatch_abi        TEXT,               -- artifact version string (B3)
    created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (source_revision, manifest_hash, signature_schema,
            hardware_schema, variant_set)
);

CREATE TABLE hardware (
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

CREATE TABLE signature (
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

CREATE TABLE candidate (
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

CREATE TABLE observation (
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

CREATE TABLE tuning_run (
    run_id                 INTEGER PRIMARY KEY,
    build_id                INTEGER NOT NULL REFERENCES build(build_id),
    run_digest              BLOB    NOT NULL UNIQUE,
    workload_digest         BLOB,
    workload_label          TEXT,
    started_at              TEXT,
    finished_at             TEXT,
    host_sync_overhead_us   REAL,
    config_json             TEXT    NOT NULL DEFAULT '{}',
    machine_json            TEXT    NOT NULL DEFAULT '{}',
    created_at              TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE measurement (
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
    launches_per_sample INTEGER NOT NULL,
    median_us        REAL,
    gpu_mad_us       REAL,                       -- MAD of GPU times (B3)
    p95_us           REAL,
    host_median_us   REAL,                       -- host clock around launch (B3)
    min_us           REAL,
    stddev_us        REAL,
    workspace_bytes  INTEGER NOT NULL DEFAULT 0,  -- candidate's DECLARED upper bound
    pool_peak_bytes  INTEGER,                    -- HI52: MEASURED peak pool bytes; NULL = not captured
    nmse             REAL,
    max_abs_err      REAL,
    max_rel_err      REAL,
    samples_json     TEXT,                      -- raw sample array
    effective_us     REAL,                      -- HI50: ranking metric (max(gpu, host-sync-adjusted))
    measured_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (build_id, hardware_id, candidate_id, objective, stage, dispatch_digest, run_id)
);

CREATE TABLE device_state (
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

CREATE TABLE winner (
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
    reason              TEXT,                      -- text explanation of why winner chosen (B3)
    confidence          REAL,
    seeded              INTEGER NOT NULL DEFAULT 0,  -- 1 = manual seed (HI11)
    validated           INTEGER NOT NULL DEFAULT 0,  -- set by HI14
    promotion_status    TEXT,                      -- HI50: native|pending_bh|confirmation_rejected|promoted|rejected_bh
    q_value             REAL,                      -- HI50: BH-adjusted q-value, set only after tune-promote
    decided_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (build_id, hardware_id, objective, dispatch_digest)
);

CREATE TABLE replay_miss (
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

CREATE TABLE candidate_blacklist (
    stable_name  TEXT NOT NULL,
    architecture TEXT NOT NULL,
    reason       TEXT NOT NULL,
    added_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (stable_name, architecture)
);

CREATE TABLE transform_attempt (
    attempt_id             INTEGER PRIMARY KEY,
    build_id               INTEGER NOT NULL REFERENCES build(build_id),
    hardware_id            INTEGER NOT NULL REFERENCES hardware(hardware_id),
    signature_id           INTEGER REFERENCES signature(signature_id),
    signature_digest       BLOB    NOT NULL,
    transformation_id      INTEGER NOT NULL,
    transformation_name    TEXT    NOT NULL,
    source                 TEXT    NOT NULL,   -- predefined|discovered
    original_native_family TEXT    NOT NULL,
    result                 TEXT    NOT NULL,   -- measured|rejected
    rejection_reason       TEXT,
    transformed_digest     BLOB,               -- null when the rewrite was never built
    transformed_winner     TEXT,               -- stable name that served the rewrite
    original_us            REAL,
    transformed_us         REAL,
    improvement_pct        REAL,
    nmse                   REAL,
    max_abs_err            REAL,
    recorded_at            TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (build_id, hardware_id, signature_digest, transformation_id)
);

CREATE TABLE transform_gap (
    gap_id                   INTEGER PRIMARY KEY,
    build_id                 INTEGER NOT NULL REFERENCES build(build_id),
    hardware_id              INTEGER NOT NULL REFERENCES hardware(hardware_id),
    signature_id             INTEGER REFERENCES signature(signature_id),
    signature_digest         BLOB    NOT NULL,
    pattern                  TEXT    NOT NULL,
    native_family            TEXT    NOT NULL,
    transformations_tried_json TEXT  NOT NULL DEFAULT '[]',
    calls                    INTEGER,          -- joined from observation
    est_bytes                INTEGER,          -- tuner's estimate, or observation's
    recorded_at              TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (build_id, hardware_id, signature_digest)
);

CREATE TABLE ranking_decision (
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

CREATE TABLE ranking_decision_candidate (
    decision_id      INTEGER NOT NULL REFERENCES ranking_decision(decision_id),
    candidate_id     INTEGER NOT NULL REFERENCES candidate(candidate_id),
    effective_us     REAL,
    rank             INTEGER,             -- position within this policy's ranking; NULL if never ranked
    verdict          TEXT    NOT NULL,    -- winner|qualified|near_tie_below_threshold|outside_tie_band|not_attempted|rejected
    rejection_reason TEXT,
    PRIMARY KEY (decision_id, candidate_id)
);

CREATE TABLE replay_coverage (
    coverage_id            INTEGER PRIMARY KEY,
    build_id                INTEGER REFERENCES build(build_id),
    hardware_id             INTEGER REFERENCES hardware(hardware_id),
    run_id                  INTEGER REFERENCES tuning_run(run_id),
    source_path              TEXT    NOT NULL UNIQUE,
    schema_version           INTEGER,        -- NULL when the source file predates replay v2's category breakdown
    total_executed           INTEGER NOT NULL,
    total_dispatched         INTEGER NOT NULL,
    replay_entries           INTEGER,
    miss_log_calls           INTEGER,
    exact                    INTEGER,        -- NULL unless schema_version = 2
    candidate_unavailable    INTEGER,        -- NULL unless schema_version = 2
    rerun_required           INTEGER,        -- NULL unless schema_version = 2
    incompatible             INTEGER,        -- NULL unless schema_version = 2
    misses                   INTEGER,
    ingested_at              TEXT    NOT NULL DEFAULT (datetime('now'))
);

