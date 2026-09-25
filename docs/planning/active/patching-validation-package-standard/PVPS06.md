---
id: PVPS06
order: 0
plan: patching-validation-package-standard
state: pending
created-at: '2026-09-25T23:16:32.446729+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Distribution-level correctness criterion for non-bit-exact patches

## Description

GPT review req_4d131e8b7c1c452d: a 5e-4 absolute logprob tolerance over EVERY vocab entry overweights meaningless extreme-tail differences (1207 failed on a ~1e-13-probability token). Replace it for patches that do not claim bit identity with combined gates; keep exact all-vocab comparison only for bit-identity claims.

## Steps

1. full_vocab: compute per step finite check, top-1 identity with margin awareness, KL(control||subject) and JS divergence over the normalized distribution, top-p (0.95) mass agreement, and max |logprob diff| restricted to tokens with p > 1e-4 in either arm.
2. New criterion object with thresholds declared in the contract (e.g. max_kl, max_js, min_topp_overlap, material_logprob_tol); bit_identical contracts keep tolerance 0.
3. Migrate producers using compare_servers (1207 history stays as recorded).
4. Unit tests with synthetic distributions: tail-only noise passes, top-1 flip fails, KL over bound fails.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Offline tests; re-evaluate 1207's recorded comparison offline to show which gate it would fail under the new criterion (its generated tokens diverged at step 55, so top-1 identity still fails).

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

No legacy fallback: the old absolute-all-vocab path remains only as the bit-identity (tolerance 0) mode.

## Change Log

- 2026-09-25T23:16:32.446729+00:00 (created-by): Created by agent

## Ledger-events

- chg_20260925_232233_correctness-checks-for-approxi_1217
- 2026-09-25T23:22:36.566063+00:00 (updated-by): Updated: section:ledger-events
