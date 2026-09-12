---
id: PRBE103
order: 0
plan: patching-rdna-boost-experiments
state: completed
created-at: '2026-09-12T16:08:58.867574+00:00'
breadth: ''
skill: ''
created-by: agent
---

# Confirm root cause of RD08's uniform sub-threshold bit_identical digest failures

## Description

A real run of run_rd08_contract_qualification() on Brutus (2026-09-13, current pin b10901/28ff0958291c, single gfx1100, tierM-gptoss20b-q6k, 6 paired rounds) found the bit_identical correctness gate FAILS for all 15 rows (5 shapes x 3 seeds), but the subject-vs-control numerical divergence is uniformly tiny (~1e-7 to 1e-6 relative magnitude in every row), far below the gate's own 0.0005 tolerance, and each row's own internal backend1-vs-backend2 comparison independently reports status=ok. This pattern (uniform, sub-threshold, present in every shape/seed rather than isolated) is inconsistent with a real VDR=2-vs-VDR=1 numerical divergence in RD08's patch and consistent with ordinary GPU non-associative floating-point reduction non-determinism between separate process launches -- which the 1222/1223 (HI67) deterministic-seed patches control the RNG inputs for but cannot control the GPU's internal warp/reduction execution order for. This hypothesis was NOT independently confirmed this session (a same-binary-run-twice control was not performed due to time).

## Steps

1. Build a single test-backend-ops binary (either RD08's VDR2-subject or VDR1-control variant, doesn't matter which since this tests binary self-consistency not RD08's correctness) via materialize_rd08_variants().
2. Run collect_all_rd08_correctness_rows() (or the equivalent lower-level row collector) TWICE against the exact same binary, same seeds, same shapes.
3. Compare backend1_digest between the two runs for every (shape, seed) row.
4. If digests differ between two runs of the IDENTICAL binary: this confirms GPU non-determinism as the root cause, not an RD08 defect -- the bit_identical gate's exact-digest-equality design needs to change (either a tolerance-based comparison, or a deterministic-execution mode that also fixes warp/reduction order, not just RNG inputs).
5. If digests are stable across repeated runs of the same binary: the original finding stands as a real, unexplained subject-vs-control divergence and needs further investigation as a genuine correctness concern.
6. Update patches/1204_rd08_q6k_mmvq_vdr2/README.md with the confirmed root cause either way.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

This finding may also be relevant to any other patch using the same bit_identical/exact-digest correctness methodology (e.g. RD39-42/1215's llama-results raw-logit approach used a single-process teacher-forced batch rather than two separate process launches, which may be why it achieved a real byte-exact match -- worth checking whether that structural difference, not luck, is why 1215's bit_identical check succeeded while RD08's did not).

**CLOSED (2026-09-13, GPT-reviewed req_d0be05f943d64d86/req_aabd4a5c2b274763/req_35aec570a6bb4038): nondeterminism hypothesis DISPROVED.**

Ran the identical VDR=1 control binary 4 independent times for every one of 15 (shape, seed) cases, using the real campaign's own environment-preservation runner (a first attempt without it produced spurious nonzero exit codes and was correctly discarded, not counted). Result: all 60 executions exited cleanly (status=ok), and every case produced a bit-identical backend1_digest across all 4 executions -- same-binary execution is fully deterministic on this hardware.

Conclusion: the original VDR=1-vs-VDR=2 digest divergence in RD08's contract-qualification run is real and deterministic, not GPU/process nondeterminism. Cross-process exact digest comparison remains a valid oracle; no tolerance-based redesign is justified. RD08 (patch 1204)'s correctness FAIL stands as a genuine, confirmed finding -- the fork's bit-identical claim is disproven. patch.toml state transitioned to "rejected" (GPT-approved deliberate lifecycle decision, no VDR=2 fix in progress). See patches/1204_rd08_q6k_mmvq_vdr2/README.md's "Final disposition" section for full detail.

## Change Log

- 2026-09-12T16:08:58.867574+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260912_160922_ran-patch-1204-rd08s-full-c_6243
- 2026-09-12T16:09:22.565880+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-12T19:21:43.738642+00:00 (updated-by): Updated: section:notes
- 2026-09-12T19:21:48.669745+00:00 (state-transition): State: pending → completed
