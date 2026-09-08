---
id: NRO06
order: 6
plan: nasone-rdna-optimizations
state: pending
created-at: '2026-09-08T09:50:40+10:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Adaptive MTP draft-depth controller

## Description

Re-evaluate and port the adaptive MTP feature represented by nasone commit `10579a7365a3bc86c4f8e41aaab20e73e1571e5e`. BigCherry's older stew675 registry deliberately excluded an earlier adaptive-MTP series because that intake was scoped to RDNA kernel work; the current project now uses MTP in production-shaped workloads, so exclusion-by-scope is no longer a technical rejection.

The source adds a distinct `draft-mtp-adaptive` mode with per-sequence state. Draft depth starts at a configurable floor, climbs after sustained full acceptance, and drops after accumulated miss pressure. This can reduce wasted verify work on unpredictable text while allowing deeper drafts on highly predictable spans. It is algorithm/runtime policy, not a GPU kernel optimization, and should be evaluated independently of kernel patches.

## Steps

1. Freeze current nasone adaptive controller and compare against the earlier stew675 adaptive-MTP commits already recorded as excluded; document semantic differences rather than creating two independent controllers.
2. Add a distinct speculative type, leaving fixed `draft-mtp` unchanged.
3. Add `n_min_adaptive` configuration/CLI with explicit floor/cap validation.
4. Implement per-sequence controller state: current depth, climb streak, drop pressure, reset semantics.
5. Port source climb thresholds and drop-pressure rule initially as candidate policy, not universal truth. Instrument every requested/accepted depth transition.
6. Ensure multi-sequence generation has independent controllers and no cross-sequence state leakage.
7. Define reset boundaries: new request, sequence reset, context rewind, failure/retry, and implementation recreation.
8. Compare against fixed depths 1..n_max and the current production fixed depth. Measure accepted tokens per target eval, drafted-but-rejected work, target/draft latency, and total TPS.
9. Run heterogeneous content sets (prose, code, repetitive text, reasoning-like text) to prevent a controller tuned to one acceptance distribution from overfitting.
10. Evaluate policy constants only after baseline characterization; any retuning becomes a separately recorded experiment identity.

## Detailed Solution & Technical Design

Source state machine:

- `n_cur` current depth in `[floor,n_max]`;
- `n_climb` consecutive full-accept verifies;
- `n_drop` accumulated `n_draft-n_accepted` pressure;
- full accept resets drop pressure and may climb when a depth-specific streak threshold is reached;
- any miss resets climb, accumulates drop pressure, and drops one level when `max(depth*5,20)` is reached;
- floor state does not accumulate pressure below the floor.

The source uses a hardened depth-3->4 barrier because acceptance reportedly collapses in ordinary prose beyond that point. BigCherry must treat those constants as source hypotheses. Instrumentation should permit replaying acceptance traces through the controller offline so alternative constants can be evaluated without rerunning model inference, but runtime policy must remain one fixed pre-registered candidate during qualification.

Correctness is primarily behavioral: adaptive and fixed modes may choose different amounts of speculative work but must preserve target-model semantics. Under deterministic sampling/temperature zero, accepted final token sequence should match nonspeculative/reference generation subject to existing MTP determinism rules.

## Code Samples & Guidance

Keep controller logic pure/testable, e.g. a small struct with `reset()` and `update(n_draft,n_accepted,cap,floor)`. Separate controller policy from model/context plumbing so exhaustive state-machine tests require no GPU.

## Files

- `docs/planning/active/nasone-rdna-optimizations/NRO06.md`
- `patches/1255_nro06_adaptive_mtp_depth/{patch.toml,patch.py,SUMMARY.md,README.md,TESTING.md}`
- shared/static patch tests; future unit tests for controller transitions and request reset semantics.

## Validation

CPU/offline: exhaustive transition tests over depths and acceptance counts, floor/cap edges, repeated reset, multi-sequence independence, malformed configuration.

Model: deterministic output parity vs target/fixed MTP; acceptance trace integrity; long request and request-boundary resets. Performance: interleaved adaptive versus best fixed baseline over multiple content classes and contexts.

## Effort & Risk

Medium-high. Implementation is smaller than a kernel, but controller constants can overfit and shift workload rather than make kernels faster. Aggregate TPS alone can conceal lower acceptance or changed token semantics.

## Standards

Treat source policy constants as experimental. No outcome-conditioned pair deletion. Final token correctness and work accounting are required alongside throughput.

## Acceptance Criteria

- State machine passes exhaustive deterministic unit tests.
- Fixed `draft-mtp` behavior remains byte/behavior unchanged.
- Adaptive deterministic final output agrees with the target reference.
- No request/sequence state leakage.
- Adaptive establishes improvement versus the best relevant fixed-depth control across the pre-registered workload mix, not just one prompt class.

## Notes

This item intentionally revives an idea formerly marked `excluded` only because the old intake scope was kernel-focused. That historical record should be cross-linked, not erased.

## Change Log

- 2026-09-08T09:50:40+10:00 (created-by): Created from current nasone adaptive-MTP implementation; P0.

## Ledger-events

- Pending: ag-ledger MCP unavailable in authoring session.
