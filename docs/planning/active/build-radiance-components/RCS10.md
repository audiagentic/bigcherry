---
id: RCS10
order: 10
plan: build-radiance-components
state: pending
created-at: '2026-10-19T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Tune with the release build: derive defaults from the shape and flags, calibrate at first start, cache the result

## Description

Owner direction 2026-10-19: we cannot run full lab sweeps for every model, quant and topology, so the release build must tune itself as far as it can. The lab harnesses (replay sweep, stream
bench, simulator) remain how we find and prove the algorithms; the release must carry the small part of them that picks settings for a model it has never seen.

GOVERNING RULE (owner, 2026-10-19): anything that can be derived is derived, and derived values beat hand config. A profile key exists only for what cannot be derived from the model shape,
the serving flags, the devices or a measurement (a policy choice, a hardware preference, a deliberate override), and every profile key is documented with why it cannot be derived. When a hand-set
value turns out to be derivable (the 4/8/16/24 lane list is), the derivation replaces it and the key stays only as an override. A bigcherry check rule lists profile keys that equal their derived value.

Three layers, cheapest first:

1. DERIVE (no measurement). Settings that follow from the model shape and the serving flags are computed, not configured: small-lane sizes from rows per step = streams x (1 + draft depth)
   (--max-num-seqs, speculative tokens) as a few classes covering 1 .. max streams; capacities from the shape and free VRAM; helper pool from the device list. RAD12's R3_MOE_WORKER_SMALL_LANES becomes
   an override of a derived default.
2. CALIBRATE (seconds, once per model x device x build). A built-in `calibrate` phase at first start (or `bigcherry tune-helper`) drives the helper with SYNTHETIC requests built from the model's own
   shape (rows 1..max, routes at the model's top-k, a Zipf-like expert mix, or the heat file when one exists) through the real lanes, times them, and picks: lane sizes that minimise expected cost
   under the derived row distribution, the GEMV / few-rows / grouped crossovers, and a kernel form per shape. Results go to a tune cache keyed by (container hash, GPU arch, helper build id, topology), the
   same idea as the existing tuning cache for llama.cpp, so the second start costs nothing and a changed build or model invalidates it.
3. ADAPT (online, guarded). The helper counts the rows it actually serves per class and can propose a different class list; it applies one only at an idle point, never mid-request, with a
   hysteresis and a fixed budget of changes per run, and logs it. Off unless enabled, because it changes timing (not results).

Everything the calibration chooses is recorded in the run record, so a result is attributable to the settings that produced it; every chosen value can be overridden from the profile.

## Steps

1. Specify the derivation rules and a profile schema for overrides (with RCS09); implement `derive_lane_classes(max_seqs, draft_depth, max_rows)` and unit-test it over a grid of flags.
2. Synthetic request generator from the shape (host-side, deterministic, seeded) shared with the replay harness; replay-based calibration loop inside the helper process with a wall-clock budget
   (default a few seconds, hard cap), no engine and no server needed.
3. Tune cache: key, file format, location under the work root, invalidation, `bigcherry` command to inspect and clear it; the run record states cache hit or fresh calibration.
4. Validate the calibrated choice against the lab result it should reproduce: on Flash-Next the derived/calibrated lane classes must match or beat the hand-chosen 4/8/16/24 in the stream A/B
   (RAD12), and a second, synthetic shape must calibrate without hand help.
5. Online adaptation behind a flag, after 1 to 4 are proven.

## Validation

Calibration time within budget on the dev helpers; chosen settings within noise of the hand-tuned ones in the clean-cell stream A/B; a model the lab has never seen starts, calibrates, and serves
with no profile beyond its container; cache hit on the second start; results (outputs) are unchanged by tuning, only timing.

## Notes

Builds on RAD12 (lane classes), RCS09 (shape from the container), RAD10 (stream-bench for validation), RHP07 (placement policy is a tuning consumer too: ownership budget and heat from the same cache).
Calibration measures the helper side only; the engine-side overlap and scheduler are validated by the stream A/B, not by calibration.

Stream-sensitive settings (owner direction 2026-10-19, after the RAD12 A/B): they derive from the inputs radiance already has, `--max-num-seqs` and the speculative (MTP) draft depth. No new "streams" key.
Each derived value keeps the name of the setting it replaces (`R3_MOE_WORKER_SMALL_ROWS`, `R3_MOE_WORKER_SMALL_LANES`, ...) as an explicit override that wins; the startup log prints `derived` or `override` per value.
A deployment aimed at 1-2 streams sets `--max-num-seqs` in its model profile; the profile lists only what differs.
RAD12 A/B (6 pairs, engine decode tps, rows8 / rows24 / four lane classes): 1 stream 122.8 / 111.7 / 109.6; 2: 178.9 / 173.3 / 174.8; 3: 194.1 / 237.8 / 214.0; 4: 245.2 / 266.6 / 242.9; 6: 318.0 / 336.1 / 305.8.
So one 24-row lane wins at 3+ streams (+7-22%), one 8-row lane at 1-2, and four lane classes lose at every count: the lane size is a function of `max-num-seqs x (1 + draft depth)`.
First step: inventory the stream-sensitive settings (lane rows and slots, ring slots, hardware queues, MTP depth, GEMV/grouped crossover, placement hot-set size, prefill chunk and ubatch, per-sequence KV and compute reservation) and verify each with stream-bench or replay before deriving it.

Live-placement roadmap stage 2 with RHP09: one calibration schema and implementation. RHP09 owns cost-model validity; RCS10 owns default-on start-up integration with opt-out, topology-keyed tune cache, and derived lane/kernel settings. RAD12 lane sizing folds in here: publish request actual_rows before its sequence; pre-queued decode kernels bound work by actual_rows, capacity only controls allocation and launch geometry; no extra polling loop or armed queue (RAD20 power findings). Acceptance: a 4-row request in a 24-row lane executes within ~3% of the 4-row lane helper execution time.

## Ledger-events

- chg_20261102_002943_planning-every-resource-budge_5643

## Change Log

- 2026-11-02T00:29:49.283929+00:00 (updated-by): Updated: section:ledger-events
