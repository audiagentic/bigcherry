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

## 2026-10-13 implementation-level audit: normalization integrity

**Status correction.** PVPS06 was implemented in commit `ca39868feb8b` on 2026-09-25. The original Steps 1-4 above are historical, not pending implementation. `tools/bigcherry/experiment/full_vocab.py` already supplies `BIT_IDENTICAL` and `NEAR_LOSSLESS`, streaming full-vocab comparison and control F64 spooling. `tools/tests/patch/test_full_vocab_criterion.py` has seven synthetic tests. The 1207 producer selects `NEAR_LOSSLESS`; 1215 selects `BIT_IDENTICAL`. Keep the current owners and v1 thresholds.

**New bounded defect.** `dense_logprobs()` verifies all token IDs and finite logprobs but never checks that the complete distribution sums to one. `step_metrics()` computes KL from raw logprobs, then clips negative KL to zero. The near-lossless v1 `judge()` can therefore accept invalid distributions: host exact-formula reproduction with control probabilities [0.60, 0.25, 0.10, 0.04, 0.01], subject logprobs uniformly **+0.01**, and matching generated IDs returns PASS (clipped KL=0, material diff=0.01, subject top-p mass=0.959548), although subject probability mass is **1.01005**. Uniform -0.01 returns FAIL (KL=0.01). Four host assertions passed. This is a **conditional malformed-input false accept**, not an observed llama-server or GPU failure.

**Exact fix seam.** Validate normalized mass for **both** control and subject inside `dense_logprobs()` after the complete row is parsed, before spool/hash/score. Use max-shifted logsumexp with `math.fsum` for stable F64 accumulation:

```python
m = max(values)
log_mass = m + math.log(math.fsum(math.exp(v - m) for v in values))
if not math.isfinite(log_mass) or abs(log_mass) > normalization_log_tol:
    raise FullVocabError(f"{arm} step {step}: invalid logprob normalization")
```

The candidate `abs(log_mass) <= 1e-3` is **provisional**, not a measured server contract. Calibrate against a pinned b11552 known-good server row and freeze a versioned threshold before implementation. Never silently renormalize invalid rows into PASS, weaken bit identity, or reclassify historical evidence. Keep `max(kl,0)` only after normalization has been validated; investigate materially negative KL instead of concealing it.

**No new comparator or criterion.** Existing v1 tests use KL<=0.005, material p>1e-4 with |logprob delta|<=0.02, subject mass >=0.94 on the control's 0.95 nucleus, and identical generated IDs. The old plan mentions JS divergence and top-p overlap, but the implemented v1 does neither. Do not silently add a JS threshold: that requires a separate versioned policy and evidence. `compare_servers()` already rejects incomplete row counts and compares only through the first generated-token mismatch, including multirow MTP stream events. Preserve that contract.

**Tests and disposition.** Extend the existing `test_full_vocab_criterion.py` with positive/negative common offsets, normalized extreme-tail perturbations, invalid mass, 151,936-token/underflow fixtures, exact-bit controls, 64/128-step and MTP multirow cases. Verify `FullVocabError` becomes incomplete/invalid evidence, never PASS or ordinary numerical FAIL. Historical 1207 is rejected because generated tokens diverged at step 55; correcting its stale `validation.toml` comment does not reopen the patch. PEC02/PVPS15 own trigger evidence; PVPS14 owns checkpoint reuse; Radiance MTP/quality paths remain independently owned and protected.

**Upstream/fork assessment.** llama.cpp issue #27972 (open, checked 2026-10-12) reports missing/zero logprobs under speculative decoding; continue failing closed on incomplete rows. Issue #29456 (open) reports full-vocab server JSON/sort cost, 115 MB/8.1 s for eight tokens on a CPU build: **external**, not BigCherry. vLLM's `_compute_probabilities` distinguishes true-model normalized logprobs from subset softmax; SGLang's `logsumexp.py` uses stable max-shifted normalization. Reuse the math concept only, not a GPU/Triton implementation.

**Evidence/next gate.** Host-only discriminator 4/4 assertions passed; a separate synthetic 151,936-token sort had median 0.0379 s across five host readings. No repository pytest, C++/HIP build, real server, or GPU benchmark ran. First require zero malformed-row false accepts, unchanged valid-row verdicts and complete producer error propagation in offline tests. Only then consider one pinned gfx1100/gfx1201 server smoke; no hardware campaign is queued and no throughput improvement is claimed. BCOP172 records the thin action/disposition.
