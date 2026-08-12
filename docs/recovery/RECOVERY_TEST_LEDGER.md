# BigCherry Recovery Test Ledger

Tracks every subsystem the `BigCherry_Definitive_Recovery_Handover_Condensed.md`
doc named, mapped to its current status: restored / superseded / intentionally
dropped / unresolved. Historical checkpoint (`413 passed, 4 subtests`, commit
`2c2fe7c`) is a semantic-coverage baseline, not a required final count — the
current suite covers a materially different, still-growing set of modules.

Source of truth for "what existed before the reset": commit `2c2fe7c`, still
an intact object in the local `.git` store (`git cat-file -t 2c2fe7c`). Verify
it's still reachable before trusting a "restored from 2c2fe7c" row below; if
pruned, treat this ledger and the individual plan items' `code_samples`
sections as the fallback record.

## C++ runtime (`src/ggml/src/ggml-cuda/`)

| Module | Old status | Current status | Plan item |
|---|---|---|---|
| `hip-autotune-dispatch.{cu,cuh}` | existed | intact, never lost | — |
| `hip-autotune-journal.{cpp,h}` | existed | restored, improved (attempt-tracing extension not in 2c2fe7c) | HI38 |
| `hip-autotune-record.{cpp,h}` | existed | intact; flush sites made crash-safe (atomic write) this session | HI49 (io.cpp reuse) |
| `hip-autotune-replay.{cpp,h}` | existed | intact, never lost | — |
| `hip-autotune-signature.{cpp,h}` | existed | intact, never lost | — |
| `hip-autotune-tuner.{cu,cuh}` | 2,869 lines, full HI34+HI50 design | reconciled: adaptive LPS, schedule-seeded confirmation, promotion_status, policy table, canary_state, SMI hooks all restored | P0, HI41 |
| `hip-autotune-io.{cpp,h}` | existed (HI48 atomic-write) | restored verbatim from 2c2fe7c | HI49 (this session) |
| `hip-autotune-smi.{cpp,h}` | existed (HI52) | restored verbatim from 2c2fe7c, wired into tuner (opt-in, `GGML_HIP_TUNE_SMI`) | (this session) |
| `hip-autotune-transform.{cu,cuh}` | existed (HI27/28) | restored from 2c2fe7c; registry compiles under `GGML_HIP_ROUTING_TRANSFORM` ON/OFF; **not** wired into the tuner's measurement loop | HI27/HI28 (restored), HI30 (unresolved — tuner integration) |

## Python tooling (`tools/bigcherry/`)

| Module | Old status | Current status | Plan item |
|---|---|---|---|
| `recipes.py`, `recipes.toml`, `upstream.py`, `patchset.py` additions | existed (post-2c2fe7c design) | current architecture is *newer* than 2c2fe7c, kept as-is per recovery doc's explicit instruction | SE02 |
| `releases.py` | existed | intact | — |
| `patcher.py`, `source_audit.py` | existed | intact | — |
| `autotune_catalog.py` | existed, simpler | build-descriptor generation (`build_descriptor`/`validate_profile_descriptor`) restored and wired into `emit()`/header | HI49 |
| `inventory.py` | existed, regressed after recovery | HI37 provenance columns wired (hip_version, effective_us, launches_per_sample, promotion_status, q_value, real `native` field instead of name-guessing); `variant_set="tuning"` hardcode and first-row hardware binding **still open**, explicitly flagged in code | HI48 |
| `report.py` | existed, regressed after recovery | SQLite filter bug (`X?` placeholder) and reject-status casing bug fixed this session | (earlier this session) |
| `replay_cache.py` | existed with fail-closed promotion gate | gate restored (`_validate_promotion_gate`), proven on real hardware (raw tune data refused, promoted data exports) | P0 |
| `tune_journal.py` | existed | intact, improved (attempt-tracing) | HI38 |
| `tune_promotion.py` | existed (BH/FDR + bootstrap) | restored, header requirements relaxed to match what the tuner actually emits (variant_set/hardware_key/config not yet in the header — HI37 gap), proven on real hardware: 77 pending_bh → 75 promoted, null-FDR empirical rate 5.05% at 5% target | P0 |
| `ranking_policy.py`, `rank_replay.py` | existed (HI50) | restored verbatim, zero adaptation needed; `--verify-parity` against real hardware: 100/100 matched | HI41 |
| `experiment_bundle.py` | existed (HI47) | restored verbatim, wired as `bigcherry experiment` | HI42 |
| `compare_tunes.py`, `ab_benchmark.py` | existed | restored verbatim, wired as `bigcherry compare-tunes`/`bigcherry ab-benchmark` | HI43 |
| `generalise.py` | existed (HI36) | restored verbatim, standalone | HI44 |
| `drift_check.py` | existed | **reverted**: needs schema-9 columns (`build.hip_version` was one; others may remain) beyond what HI48 wired so far | HI44 (open remainder), HI48 |
| `resource_report.py`, `candidate_binary_size.py` | existed | restored verbatim, wired as `bigcherry resource-report`/`bigcherry candidate-binary-size` | HI45 |
| `release_validate.py` | existed, `PROFILES`-driven build orchestration | **rebuilt, not ported** — thin probe over `bigcherry pull`+`bigcherry build`; deliberately narrower than the doc's full 25-step gate sequence | HI46 |
| `bandit_simulator.py` | existed | restored verbatim, standalone | HI47 |
| `pareto_policy_*.py` (4 files), `pareto_report.py` | existed | **not restored** — the four adapters import `ranking_policy.PROTOTYPE_POLICIES` (now real) and `pareto_report` (never restored); would be dead files without `pareto_report.py` | HI47 (open remainder) |
| `manifest_resolve.py` | doc says "unproven; do not create" | not created | — (correctly not attempted) |
| `presets.py` | doc says superseded | not restored | — (correctly not attempted) |

## Database schema (`sql/dispatch-db.sql`)

Pre-reset surviving database evidence: `campaign.sqlite` and siblings on
brutus (`artifacts/tuning-runs/hi-campaign-*`), dated 2026-08-10, hours
before the reset — real, not reconstructed. Schema there is
**`schema_version=9`**, 17 tables. Full DDL extracted and saved at
`docs/recovery/schema9-recovered-ddl.sql`.

Current schema was `schema_version=1` (10 tables) before this session;
now **`schema_version=2`** after HI48: added `build.hip_version`,
`measurement.launches_per_sample`/`pool_peak_bytes`/`effective_us`,
`winner.promotion_status`/`q_value`, and four new tables (`tuning_run`,
`device_state`, `ranking_decision`, `ranking_decision_candidate`) whose
columns match the schema-9 DDL field-for-field.

**Not carried forward from schema 9**, intentionally deferred:

- `replay_coverage` table — no current producer ingests `coverage.json` into SQLite yet.
- `transform_attempt`/`transform_gap` tables — no current producer; blocked on HI30 (tuner-side routing-transform integration, not yet wired).
- Renumbering: schema-9's version number is not reused verbatim (the intermediate 2–8 DDL between current's prior "1" and schema-9 was never recovered), so this is schema "2" here, not "9" — a deliberate, documented departure, not an inconsistency.

`inventory.py`'s writer populates every new column that has a real data
source today (all of the above except `pool_peak_bytes`, which needs HI40's
requested-size fix first to mean anything, and `q_value`, which is only
present once a measurements file has been through `tune-promote`).

**Still open in `inventory.py`** (flagged in code with `KNOWN GAP` comments,
not silently accepted):
- `build.variant_set` hardcoded to `"tuning"` — the tuner's measurements
  header doesn't carry the real variant set yet.
- Hardware binding uses `SELECT hardware_id FROM hardware LIMIT 1` — wrong
  on any database holding more than one architecture's rows.

## End-to-end hardware validation

Real (not simulated) runs performed this session, all on brutus (dual
gfx1100 + gfx1030 + gfx1201 available):

1. **Full P0 pipeline**: record → tune (adaptive LPS, schedule-seeded
   confirmation) → `tune-promote` (BH/bootstrap) → `tune-null-fdr` → fail-closed
   `replay_cache` export. Raw unpromoted data refused; promoted data exported
   cleanly (100 winners).
2. **HI41 policy table**: live tune run, `rank-replay --verify-parity`:
   100/100 matched against the recorded production policy's decisions.
3. **HI48 schema**: real measurements loaded into a fresh schema-2 database;
   `promotion_status`, `effective_us`, `launches_per_sample`, `hip_version`,
   correct native-candidate resolution all verified present and correct.
4. **Comparative benchmark**: stock vs. properly-tuned bigcherry-replay,
   Qwopus3.6-27B-v2-MTP-Q8_0, dual-gfx1100 `-sm tensor`, 15 reps, 100%
   coverage/0 misses confirmed via `coverage.json` before trusting the
   comparison.

Not yet run this session: a full `bigcherry validate-release`-style gate
sequence against a production-scale workload with the schema-2 database and
HI41's policy table all combined in one pass. Recommended before declaring
the recovery "fully reconciled" per the original doc's section 22 checklist.

## Definition of fully reconciled — checklist against the original doc

Per section 22 of the recovery handover:

- [x] Patches: 17 accounted for, 0100–0820 validated baseline applies.
- [ ] HI40 (pool workspace requested-size fix) — still open, pre-dates this session's work.
- [ ] `1005` anchors — still untested, pre-dates this session's work.
- [x] EX02 remains narrow.
- [x] Tuner: shared adaptive LPS, LPS persisted, fresh confirmation, p-value, `pending_bh`, native state, journal preserves evidence, raw samples sufficient for offline recomputation.
- [x] Promotion/replay: `tune-promote`, BH/FDR, bootstrap/effect gate, `tune-null-fdr`, raw non-native cannot export, promoted can, rejected cannot.
- [ ] Data/provenance: schema reconstructed to a *documented* schema 2 (not schema 5/9 verbatim), current-only enforcement not yet added as a reader check, schema policy fields present but variant-set/hardware-binding gaps still open.
- [x] Experiment/runtime: experiment-bundle core restored.
- [x] SMI restored. IO recovered exactly (not a forensic hold — exact source was available).
- [x] Transforms restored at registry level; ON/OFF compile proven. Tuner integration (HI30) still open.
- [ ] Analysis/release: compare-tunes/ab-benchmark restored but not exercised against a real event-reuse/L2 experiment this session; impact/kernel-fraction/Pareto/generalisation restored at module level but not all run end-to-end; recipe-based release validation rebuilt but narrower than the doc's full gate sequence.
- [x] Tests: current suite green (260 passing at HI48's start; grows with each item). This ledger exists.
