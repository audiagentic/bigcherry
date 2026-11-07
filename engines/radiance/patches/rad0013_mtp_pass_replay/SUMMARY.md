# rad0013_mtp_pass_replay

**Status:** untested
**Plan item:** none

Scope: Qwen4-Exp v1.17.0 recorded-pass regression. Card-free patch and callback tests pass; hardware (four-GPU Flash-Next, two passes): decode step 1.7-2.7% faster on all five accept-bench prompts, generated text and acceptance identical to the unpatched build. Promotion to serving-core waits on a KLD check.

## Summary

Restores recorded-pass replay for **fixed-shape decode MTP draft and speculative-verify passes** in the pinned Radiance v1.17.0 Qwen4-Exp architecture. It replaces the blanket draft/verify refusal with a guarded geometry fingerprint in `arch/qwen4exp_fp8/qwen4exp_fp8.cpp`. Does not change the core tape runtime, kernel selection or any GPU code.

## Outcome

Card-free patch mechanics and compiled callback tests pass; hardware replay and performance are not yet measured.

## Detail

### Geometry contract

* Only pure decode batches are eligible. Head history (negative `draft_pass`), later MTP rounds (`draft_pass > 1`) and speculative verify (`n_spec > 0`) are keyed **only** if the full attention is h256, block-4, GQA6 or GQA12 and its per-sequence query band is at most 10 or 5 respectively.
* All attention launches are traced in the key: dense attention uses the batch's decode band, number of sequences, KV table pitch and the pinned h256 split-KV decode law; QSA sparse attention uses q_len 1, its own query-row chunks and selection-table pitch. The key includes their computed split counts. For these decode rows the splitter is host-deterministic; unlike prefill it never reads a live KV-context maximum.
* The architecture key adds draft round, speculation width, row and sequence counts, decode width, host context bound, KV-group geometry and split counts. The core tape key already includes batch shape, all relevant device pointers, block-table and stream-view identities, arena/epoch and launch-level state.
* Mixed phases, non-decode MTP/verify, widths above the fixed decode limit (which select the live-context prefill splitter), absent/invalid geometry, unsupported layouts, missing MTP head, and ring cases whose sparse-vs-dense stage choice cannot be proven all return zero, forcing live issue. Ordinary non-spec decode and the existing prefill callback path are preserved.

### Verification

`tools/tests/patch/test_radiance_mtp_pass_replay.py` checks the exact v1.17.0 source pin, patch application/idempotence, drift refusal, and invokes the **compiled upstream architecture callback** extracted from the patched source with synthetic C++ batch/model fixtures. No device, GPU allocation or HIP build is used. Hardware tape capture/replay and throughput remain unverified.

No production patch set is changed by this addition; promotion and hardware evaluation must be separate decisions.
