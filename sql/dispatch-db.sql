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

-- Schema 4 (RE09/RV50: identity_scope): the build table's row identity now
-- matches the identity model instead of contradicting it -- a row is either
-- 'campaign' (complete source_slice_id/build_plan_id/effective_build_id,
-- uniquely identified BY that triple) or 'legacy-imported' (uniquely
-- identified by the old (source_revision, manifest_hash, signature_schema,
-- hardware_schema, variant_set, build_descriptor_hash) tuple, exactly as
-- before). Two campaign builds that happen to share a legacy key but carry
-- genuinely different campaign identity can now coexist as two real rows,
-- instead of either colliding or silently aliasing (see the interim
-- fail-closed check inventory.py carried before this schema existed).
-- Named identity_scope, not provenance_class: artifact-level provenance
-- (bigcherry.provenance.ProvenanceClass: production/imported-legacy/
-- development/diagnostic) is a DIFFERENT axis with different values: a
-- same-named DB column would invite conflating the two.
--
-- Schema 3 (campaign identity) added the columns below without changing
-- row identity, matching a real recovered pre-reset schema (schema_version
-- 9 there; renumbered here since the intermediate 2-8 DDL was never
-- recovered -- see docs/recovery/schema9-recovered-ddl.sql for the exact
-- source and docs/recovery/RECOVERY_TEST_LEDGER.md for what carried over vs
-- what did not). A reader must reject any schema_version it does not
-- recognise (standards: current-only, no silent migration) rather than
-- guess at an unlisted intermediate shape.
INSERT OR IGNORE INTO schema_meta(key, value) VALUES
    ('schema_version',    '6'),
    ('signature_schema',  '1'),
    ('hardware_schema',   '1'),
    ('transform_schema',  '1');

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
    source_slice_id      TEXT,
    build_plan_id        TEXT,
    effective_build_id   TEXT,
    campaign_run_id      TEXT,
    workload_id          TEXT,
    dispatch_abi        TEXT,               -- artifact version string (B3)
    identity_scope       TEXT    NOT NULL DEFAULT 'legacy-imported',
    created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    CHECK (identity_scope IN ('campaign', 'legacy-imported')),
    -- SQLite cannot express a conditional NOT NULL directly; this CHECK is
    -- the equivalent: a 'campaign' row must carry the complete triple, or
    -- it is not really campaign-identified at all.
    CHECK (identity_scope = 'legacy-imported' OR (
        source_slice_id IS NOT NULL
        AND build_plan_id IS NOT NULL
        AND effective_build_id IS NOT NULL
    ))
);

-- The real, non-contradictory row identity, one partial index per scope --
-- replaces the old table-level UNIQUE, which used to apply to every row
-- regardless of whether it carried campaign identity at all.
CREATE UNIQUE INDEX IF NOT EXISTS build_campaign_identity_uq
    ON build(source_slice_id, build_plan_id, effective_build_id)
    WHERE identity_scope = 'campaign';

CREATE UNIQUE INDEX IF NOT EXISTS build_legacy_identity_uq
    ON build(source_revision, manifest_hash, signature_schema,
             hardware_schema, variant_set, build_descriptor_hash)
    WHERE identity_scope = 'legacy-imported';

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
    source_slice_id     TEXT,
    workload_id         TEXT,
    campaign_run_id     TEXT,
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
    source_slice_id   TEXT,
    build_plan_id     TEXT,
    effective_build_id TEXT,
    workload_id       TEXT,
    campaign_run_id   TEXT,
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
    source_slice_id    TEXT,
    build_plan_id      TEXT,
    effective_build_id TEXT,
    workload_id        TEXT,
    campaign_run_id    TEXT,
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
    source_slice_id  TEXT,
    workload_id      TEXT,
    campaign_run_id  TEXT,
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

-- ----------------------------------------------------------- transform_attempt
-- HI33 offline evidence.  These tables are additive: runtime transforms,
-- replay builds, and existing measurement/release formats do not depend on
-- them.  Provenance is bound to the existing build/hardware/signature rows.

CREATE TABLE IF NOT EXISTS transform_attempt (
    attempt_id             INTEGER PRIMARY KEY,
    build_id               INTEGER NOT NULL REFERENCES build(build_id),
    hardware_id            INTEGER NOT NULL REFERENCES hardware(hardware_id),
    signature_digest       BLOB    NOT NULL REFERENCES signature(signature_digest),
    dispatch_digest        BLOB,
    transformation_id      INTEGER NOT NULL,
    transformation_name    TEXT    NOT NULL,
    source                  TEXT    NOT NULL CHECK (source IN ('predefined', 'discovered')),
    result                  TEXT    NOT NULL CHECK (result IN ('success', 'rejected')),
    reason                  TEXT    NOT NULL,
    transformed_sig_digest  BLOB,
    evidence_references     TEXT    NOT NULL,
    attempted_at            TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (build_id, hardware_id, signature_digest, transformation_id)
);

CREATE INDEX IF NOT EXISTS transform_attempt_sig_idx
    ON transform_attempt(build_id, signature_digest);
CREATE INDEX IF NOT EXISTS transform_attempt_result_idx
    ON transform_attempt(result, transformation_name);

-- --------------------------------------------------------------- transform_gap
-- One validated gap per build/hardware/source signature.  calls and est_bytes
-- are copied from observation at load time for stable offline prioritisation.

CREATE TABLE IF NOT EXISTS transform_gap (
    gap_id                  INTEGER PRIMARY KEY,
    build_id                INTEGER NOT NULL REFERENCES build(build_id),
    hardware_id             INTEGER NOT NULL REFERENCES hardware(hardware_id),
    signature_digest        BLOB    NOT NULL REFERENCES signature(signature_digest),
    pattern_description     TEXT    NOT NULL,
    native_family           TEXT    NOT NULL,
    transformations_tried  TEXT    NOT NULL,
    calls                   INTEGER NOT NULL DEFAULT 0,
    est_bytes               INTEGER NOT NULL DEFAULT 0,
    evidence_references     TEXT    NOT NULL,
    created_at              TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (build_id, hardware_id, signature_digest)
);

CREATE INDEX IF NOT EXISTS transform_gap_sig_idx
    ON transform_gap(build_id, signature_digest);
CREATE INDEX IF NOT EXISTS transform_gap_pattern_idx
    ON transform_gap(native_family, pattern_description);

-- ============================================================================
-- Vulkan autotune tables (RE30 phase 2, 2026-08-20).
--
-- Deliberately PARALLEL tables, not a `backend` discriminator column added to
-- the HIP tables above. Rationale (see RE30's schema-design decision in the
-- plan item): the HIP tables' CHECK constraints, UNIQUE indexes and every
-- existing Python reader (inventory.py, replay_cache.py, tune_journal.py,
-- rank_replay.py, ...) are written assuming HIP-only content -- e.g.
-- candidate.family's CHECK enumerates {mmq,mmvf,mmf,mmvq,blas} and
-- winner/measurement carry no field a Vulkan pipeline-recipe identity would
-- naturally fill (push-constant layout, SPIR-V fingerprint, coopmat class).
-- A discriminator column would force every one of those constraints and
-- readers to become backend-conditional to stay correct, which is exactly
-- the "silently relabelled and corrupts compatibility" risk RE30's own
-- Effort & Risk section warns about. Parallel `vk_*` tables cost a small
-- amount of duplication but leave every existing HIP table, CHECK, index and
-- reader completely untouched -- zero regression surface by construction.
-- schema_version bumped 4 -> 5 (GPT review + user directive, 2026-08-20:
-- an old schema-4 database with no vk_* tables and a new schema-4 database
-- with six Vulkan tables were materially different shapes carrying the same
-- version -- a real gap, not acceptable to leave silently unversioned).
-- See the migration statement below the INSERT block for how real existing
-- databases move from 4 to 5 IN PLACE, keeping every row of real HIP data.
-- This is NOT the "guess at an unlisted intermediate shape" case
-- schema_meta's own comment above warns readers against: the 4->5 delta is
-- fully known, additive-only (six new tables, zero changes to any existing
-- table/column/index), and versioned explicitly here -- unlike the lost
-- schema 2-8 DDL, there is nothing to guess. A reader must still reject any
-- schema_version it does not recognise; this migration exists so a real
-- database that WAS at the recognised prior version (4) becomes the new
-- recognised version (5) without losing its history, rather than every
-- existing production dispatch database silently failing
-- `_require_current_schema`'s exact-match check the moment this code ships.
--
-- No Vulkan patch, dispatch hook, or measurement code exists yet (RE30 is
-- still pre-implementation past this scaffolding) -- these tables have no
-- writer today. They exist so the identity/persistence shape is settled and
-- reviewable before any Vulkan record/tune/replay code is written against it.
-- ============================================================================

-- ------------------------------------------------------------- vk_hardware
-- Executing GPU+driver *class* for Vulkan, mirroring hardware's role: no
-- device ordinal (same standards-10 sharing rule as HIP). Vulkan identity
-- needs more axes than HIP's architecture_code because ICDs vary more than
-- ROCm's gfx target strings do (RE30 detailed_solution).

CREATE TABLE IF NOT EXISTS vk_hardware (
    vk_hardware_id     INTEGER PRIMARY KEY,
    hardware_digest     BLOB    NOT NULL UNIQUE,  -- blake2b(person=llama-vk-hardware)
    vendor_id           INTEGER NOT NULL,
    device_id           INTEGER NOT NULL,
    device_class        TEXT    NOT NULL,         -- stable class/UUID, not a PCI ordinal
    driver_version       TEXT    NOT NULL,
    api_version          TEXT    NOT NULL,         -- Vulkan API version, e.g. "1.3.280"
    subgroup_size        INTEGER NOT NULL,
    subgroup_ops_mask    INTEGER NOT NULL,         -- bitmask of supported subgroup operations
    extensions_json      TEXT    NOT NULL,         -- sorted JSON array of enabled extension names
    limits_json          TEXT    NOT NULL,         -- relevant VkPhysicalDeviceLimits fields
    shader_toolchain_digest BLOB NOT NULL,          -- glslc/SPIR-V/source fingerprint
    canonical_json       TEXT    NOT NULL,         -- full-key collision check, mirrors hardware.canonical_json
    created_at           TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- -------------------------------------------------------------- vk_signature
-- Canonical device-local Vulkan operation description, mirroring `signature`.
-- Carries fields signature.canonical_json has no room for: layout/alignment,
-- conversion/quantisation route and split-K/fusion condition are Vulkan
-- pipeline-recipe identity, not HIP kernel-variant identity (RE30
-- detailed_solution's signature field list).

CREATE TABLE IF NOT EXISTS vk_signature (
    vk_signature_id     INTEGER PRIMARY KEY,
    signature_digest     BLOB    NOT NULL UNIQUE,   -- blake2b(person=llama-vk-tune)
    base_digest          BLOB    NOT NULL,
    schema_version       INTEGER NOT NULL,
    op                   TEXT    NOT NULL,
    src0_type            TEXT    NOT NULL,
    src1_type            TEXT    NOT NULL,
    dst_type             TEXT    NOT NULL,
    output_precision     TEXT    NOT NULL,
    accumulation_precision TEXT  NOT NULL,
    m                    INTEGER NOT NULL,
    n                    INTEGER NOT NULL,
    k                    INTEGER NOT NULL,
    layout               TEXT    NOT NULL,          -- e.g. "row_major", "col_major", "coopmat"
    alignment_class       INTEGER NOT NULL,
    batching              TEXT    NOT NULL DEFAULT 'none',
    conversion_route      TEXT    NOT NULL DEFAULT 'none',  -- quantise/dequantise path taken
    split_k               INTEGER NOT NULL DEFAULT 0,
    fusion                TEXT    NOT NULL DEFAULT 'none',
    is_refined            INTEGER NOT NULL DEFAULT 0,
    canonical_json        TEXT    NOT NULL,
    created_at            TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS vk_signature_base_idx ON vk_signature(base_digest);

-- --------------------------------------------------------------- vk_candidate
-- A Vulkan candidate is a complete executable *pipeline recipe*
-- (preparation + main pipeline + optional reduction), never a bare dispatch
-- -- see RE30 detailed_solution's "pinned upstream seam" note on why a
-- partial-recipe candidate would time a false winner. pipeline_stage_count
-- records how many command-buffer stages the recipe comprises so a reader
-- can tell a fused single-dispatch candidate from a multi-stage one without
-- re-parsing config_json.

CREATE TABLE IF NOT EXISTS vk_candidate (
    vk_candidate_id         INTEGER PRIMARY KEY,
    build_id                 INTEGER NOT NULL REFERENCES build(build_id),
    stable_name               TEXT    NOT NULL,
    family                    TEXT    NOT NULL,      -- mul_mat|mul_mat_id (grows as families are added)
    source_class              TEXT    NOT NULL,       -- mirrors ggml_hip_source_class's vocabulary
    implementation_version     INTEGER NOT NULL,
    pipeline_stage_count       INTEGER NOT NULL DEFAULT 1,
    shader_module_digests_json TEXT    NOT NULL,      -- JSON array, one SPIR-V digest per stage
    graph_safe                 INTEGER NOT NULL,
    deterministic               INTEGER NOT NULL,
    config_json                 TEXT    NOT NULL,
    UNIQUE (build_id, stable_name),
    CHECK (family IN ('mul_mat', 'mul_mat_id')),
    CHECK (source_class IN ('native_wrapper','existing_runtime',
                            'existing_alternative','new_generated_variant',
                            'vendor_auto','vendor_explicit'))
);

-- ------------------------------------------------------------- vk_observation
-- Vulkan record mode, mirroring `observation` exactly in shape.

CREATE TABLE IF NOT EXISTS vk_observation (
    vk_observation_id    INTEGER PRIMARY KEY,
    build_id              INTEGER NOT NULL REFERENCES build(build_id),
    vk_hardware_id         INTEGER NOT NULL REFERENCES vk_hardware(vk_hardware_id),
    vk_signature_id         INTEGER NOT NULL REFERENCES vk_signature(vk_signature_id),
    native_stable_name      TEXT    NOT NULL,
    calls                    INTEGER NOT NULL DEFAULT 0,
    est_bytes                INTEGER NOT NULL DEFAULT 0,
    est_flops                INTEGER NOT NULL DEFAULT 0,
    sites_json                TEXT    NOT NULL DEFAULT '[]',
    diagnostics_json          TEXT    NOT NULL DEFAULT '{}',
    source_slice_id           TEXT,
    workload_id                TEXT,
    campaign_run_id            TEXT,
    first_seen                 TEXT    NOT NULL DEFAULT (datetime('now')),
    last_seen                  TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (build_id, vk_hardware_id, vk_signature_id)
);

CREATE INDEX IF NOT EXISTS vk_observation_hot_idx
    ON vk_observation(build_id, vk_hardware_id, calls DESC);

-- ------------------------------------------------------------- vk_measurement
-- One row per (dispatch key, candidate, objective) tuning attempt, mirroring
-- `measurement`. RE30 detailed_solution's tuning-transaction rules (create
-- pipelines outside timing, warm up, submit/wait once per timed sample,
-- fail closed on device-lost/timestamp failure, validate against native
-- output) apply to how a future writer fills this table, not to its shape.

CREATE TABLE IF NOT EXISTS vk_measurement (
    vk_measurement_id     INTEGER PRIMARY KEY,
    build_id               INTEGER NOT NULL REFERENCES build(build_id),
    vk_hardware_id          INTEGER NOT NULL REFERENCES vk_hardware(vk_hardware_id),
    vk_signature_id          INTEGER REFERENCES vk_signature(vk_signature_id),
    dispatch_digest          BLOB,
    vk_candidate_id           INTEGER NOT NULL REFERENCES vk_candidate(vk_candidate_id),
    run_id                    INTEGER REFERENCES tuning_run(run_id),
    objective                 TEXT    NOT NULL DEFAULT 'latency',
    stage                     TEXT    NOT NULL,     -- screen|final
    accepted                  INTEGER NOT NULL,
    reject_reason             TEXT,
    samples                   INTEGER NOT NULL DEFAULT 0,
    median_us                 REAL,
    p95_us                    REAL,
    min_us                    REAL,
    stddev_us                 REAL,
    pipeline_creation_us       REAL,     -- excluded from the timed sample; recorded for diagnosis
    command_buffer_us          REAL,     -- the actual timed quantity (RE30: "timestamp the full recipe")
    workspace_bytes            INTEGER NOT NULL DEFAULT 0,
    nmse                       REAL,
    max_abs_err                 REAL,
    max_rel_err                 REAL,
    samples_json                 TEXT,
    source_slice_id              TEXT,
    build_plan_id                 TEXT,
    effective_build_id            TEXT,
    workload_id                    TEXT,
    campaign_run_id                TEXT,
    measured_at                     TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (build_id, vk_hardware_id, vk_candidate_id, objective, stage, dispatch_digest)
);

CREATE INDEX IF NOT EXISTS vk_measurement_dispatch_idx
    ON vk_measurement(build_id, dispatch_digest, objective);

-- ----------------------------------------------------------------- vk_winner
-- Result of Vulkan winner selection, mirroring `winner`. dispatch_digest is
-- exported into the Vulkan replay artifact, whose header carries its own
-- distinct magic/version (RE30 detailed_solution) so it can never be loaded
-- by a HIP replay reader or vice versa.

CREATE TABLE IF NOT EXISTS vk_winner (
    vk_winner_id           INTEGER PRIMARY KEY,
    build_id                 INTEGER NOT NULL REFERENCES build(build_id),
    vk_hardware_id            INTEGER NOT NULL REFERENCES vk_hardware(vk_hardware_id),
    vk_signature_id            INTEGER REFERENCES vk_signature(vk_signature_id),
    objective                  TEXT    NOT NULL DEFAULT 'latency',
    dispatch_digest             BLOB    NOT NULL,
    vk_candidate_id              INTEGER NOT NULL REFERENCES vk_candidate(vk_candidate_id),
    run_id                        INTEGER REFERENCES tuning_run(run_id),
    stable_name                    TEXT    NOT NULL,
    native_stable_name              TEXT    NOT NULL,
    is_native                        INTEGER NOT NULL,
    improvement_pct                  REAL    NOT NULL DEFAULT 0.0,
    median_us                         REAL    NOT NULL,
    p95_us                             REAL    NOT NULL,
    workspace_bytes                     INTEGER NOT NULL DEFAULT 0,
    reason                               TEXT,
    confidence                           REAL,
    seeded                                INTEGER NOT NULL DEFAULT 0,
    validated                              INTEGER NOT NULL DEFAULT 0,
    promotion_status                        TEXT,
    q_value                                  REAL,
    source_slice_id                           TEXT,
    build_plan_id                              TEXT,
    effective_build_id                          TEXT,
    workload_id                                  TEXT,
    campaign_run_id                               TEXT,
    decided_at                                     TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (build_id, vk_hardware_id, objective, dispatch_digest)
);

CREATE UNIQUE INDEX IF NOT EXISTS vk_winner_dispatch_idx
    ON vk_winner(dispatch_digest, objective);

-- ==========================================================================
-- HI67 slices 2/3 (RV49 contract, RV77 GPT-adjudicated implementation
-- design, 2026-08-20/21): CPU-reference correctness evidence.
--
-- The tuner's existing native-relative acceptance (measurement.nmse/
-- max_abs_err above, compared against a candidate's OWN native run) does not
-- bound end-to-end correctness -- native itself has its own floating-point
-- error relative to the true CPU-reference computation, so a candidate can
-- look "close to native" while both have drifted from the truth (the real
-- q4_1 k=32 RDNA2 intermittent failure this schema exists to close, RV08).
--
-- correctness_evidence is production PROMOTION proof, keyed on the same
-- normalized build/hardware/signature/candidate identities every other
-- table here uses -- deliberately NOT a parallel source_slice_id/
-- architecture/stable-name text-key namespace (RV77 Q2 change 1): one
-- source slice can back multiple build plans/manifests, so build_id is the
-- real namespace boundary, not source_slice_id alone.
--
-- Per-seed observations live in correctness_evidence_seed, not just an
-- aggregated seeds_json array plus worst-of-seeds numbers on the parent row
-- (RV77 Q2 change 2) -- a consumer must be able to verify every declared
-- seed actually ran, native and candidate used the same seed set, and no
-- failed seed was silently dropped from the aggregate. reference_digest
-- (from patches/1222_hi67_deterministic_test_backend_ops_seed.py's
-- BIGCHERRY_REF_DIGEST output) proves the native and candidate runs for one
-- seed actually compared against the same CPU-reference input.
--
-- contract_version/threshold_t/headroom_fraction on the parent row are
-- PRODUCER-RECORDED METADATA ONLY (RV77 Q2 change 4) -- promotion code must
-- independently know the currently-required contract_version/T/headroom and
-- reject evidence whose recorded parameters differ, never trust these
-- columns as authority. A row's own claimed policy is provenance, not
-- permission.
-- ==========================================================================

-- ------------------------------------------------------- correctness_evidence
-- One row per (build, hardware, signature, candidate): the worst-of-seeds
-- verdict a promotion decision actually consumes. e_n_nmse/e_c_nmse/
-- max_abs_native/max_abs_candidate here are DERIVED from the child
-- correctness_evidence_seed rows (the worst value across all of them, per
-- the RV49 contract's "use the WORST result across >=3 deterministic
-- seeds, not an average" rule) -- an index for promotion queries, not a
-- second source of truth; the seed rows are the actual evidence.

CREATE TABLE IF NOT EXISTS correctness_evidence (
    correctness_evidence_id INTEGER PRIMARY KEY,
    build_id             INTEGER NOT NULL REFERENCES build(build_id),
    hardware_id          INTEGER NOT NULL REFERENCES hardware(hardware_id),
    signature_id         INTEGER NOT NULL REFERENCES signature(signature_id),
    candidate_id         INTEGER NOT NULL REFERENCES candidate(candidate_id),
    native_candidate_id  INTEGER NOT NULL REFERENCES candidate(candidate_id),
    contract_version     TEXT    NOT NULL,   -- e.g. 'hi67-rv49-v1'; producer-recorded, see header note
    threshold_t          REAL    NOT NULL,   -- producer-recorded upstream correctness threshold for this op
    headroom_fraction    REAL    NOT NULL,   -- producer-recorded; default 0.5 per RV49
    e_n_nmse             REAL    NOT NULL,   -- worst-of-seeds NMSE(R,N), derived from seed rows
    e_c_nmse             REAL    NOT NULL,   -- worst-of-seeds NMSE(R,C), derived from seed rows
    max_abs_native       REAL    NOT NULL,   -- worst-of-seeds max_abs(N,R), derived from seed rows
    max_abs_candidate    REAL    NOT NULL,   -- worst-of-seeds max_abs(C,R), derived from seed rows
    seed_count           INTEGER NOT NULL,   -- number of correctness_evidence_seed rows this aggregates
    tool_version          TEXT    NOT NULL,
    computed_at            TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (build_id, hardware_id, signature_id, candidate_id, contract_version),
    CHECK (seed_count >= 3)
);

CREATE INDEX IF NOT EXISTS correctness_evidence_lookup_idx
    ON correctness_evidence(build_id, hardware_id, signature_id, candidate_id);

-- -------------------------------------------------------- correctness_evidence_seed
-- One row per deterministic seed a correctness_evidence row aggregates.
-- reference_digest is the BIGCHERRY_REF_DIGEST value the forced-native and
-- forced-candidate test-backend-ops invocations both printed for this seed
-- -- the evidence generator must assert they match before writing this row
-- (RV77 Q1 hard gate: two separate process invocations are only a valid
-- joint R/N/C measurement if both actually saw the same CPU-reference
-- input).

CREATE TABLE IF NOT EXISTS correctness_evidence_seed (
    correctness_evidence_seed_id INTEGER PRIMARY KEY,
    correctness_evidence_id  INTEGER NOT NULL REFERENCES correctness_evidence(correctness_evidence_id),
    seed                      INTEGER NOT NULL,
    reference_digest          TEXT    NOT NULL,
    e_n_nmse                  REAL    NOT NULL,
    e_c_nmse                  REAL    NOT NULL,
    max_abs_native            REAL    NOT NULL,
    max_abs_candidate         REAL    NOT NULL,
    native_execution_status   TEXT    NOT NULL,   -- ok|failed|timeout
    candidate_execution_status TEXT   NOT NULL,   -- ok|failed|timeout
    -- HI67 threshold-authority fix (2026-08-22): the upstream correctness
    -- threshold T as ACTUALLY EMITTED by test-backend-ops for this seed
    -- (BIGCHERRY_CORRECTNESS_METRIC's own threshold=... field), never a
    -- caller-supplied Python float. The parent correctness_evidence row's
    -- own threshold_t is derived FROM these (all seeds of one row must
    -- agree), not the other way around -- this column is the actual source
    -- of truth. Added before this feature ever ran against a real database
    -- (offline-verified only as of this commit), so no migration is needed.
    threshold_t                REAL   NOT NULL,
    UNIQUE (correctness_evidence_id, seed),
    CHECK (native_execution_status IN ('ok', 'failed', 'timeout')),
    CHECK (candidate_execution_status IN ('ok', 'failed', 'timeout'))
);

-- ------------------------------------------------------------- migration 4->5
--
-- Placed at the END of the script, deliberately: schema_version only flips
-- forward once every table it names (the six vk_* tables above) has
-- actually been created by the CREATE TABLE IF NOT EXISTS statements that
-- precede this point, so no reader can ever observe schema_version='5' on a
-- database that is mid-migration and still missing a vk_* table.
--
-- The INSERT OR IGNORE near the top of this file only takes effect on a
-- brand-new database (no schema_meta row yet). A REAL EXISTING database
-- still carrying '4' needs this explicit, unconditional UPDATE to move
-- forward in place -- an INSERT OR IGNORE would silently no-op against an
-- existing '4' row and leave every production dispatch database
-- permanently stuck on the old version the moment
-- inventory.CURRENT_DB_SCHEMA_VERSION becomes '5' in Python, since
-- `_require_current_schema` rejects any exact-mismatch on read.
--
-- No existing row in build/hardware/signature/candidate/observation/
-- measurement/winner is touched, dropped, or reshaped by this migration --
-- schema 4 -> 5 is purely additive (six new vk_* tables, zero changes to
-- any table/column/index that existed at schema 4). This is NOT the
-- "guess at an unlisted intermediate shape" case schema_meta's own comment
-- warns readers against: the delta is fully known and versioned explicitly
-- here. Idempotent: matches zero rows once a database is already at '5'.
UPDATE schema_meta SET value = '5' WHERE key = 'schema_version' AND value = '4';

-- ------------------------------------------------------------- migration 5->6
--
-- Same discipline as 4->5 above: placed after the correctness_evidence /
-- correctness_evidence_seed CREATE TABLE statements, so schema_version never
-- flips to '6' on a database still missing either table. Purely additive --
-- two new tables, zero changes to any table/column/index that existed at
-- schema 5 (including the six vk_* tables 4->5 added). Idempotent: matches
-- zero rows once a database is already at '6'.
UPDATE schema_meta SET value = '6' WHERE key = 'schema_version' AND value = '5';
