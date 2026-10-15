# Testing — PRBE41 SSM_CONV channels-major (route-corrected 2026-10-14)

## Applicability and source identity

The existing 1263 patch implements CPU/CUDA/HIP channels-major SSM_CONV and changes the recurrent conv-state layout. `src/models/qwen35.cpp`, `qwen35moe.cpp`, and `qwen3next.cpp` select it only when `model.split_mode() != LLAMA_SPLIT_MODE_TENSOR`. The current dual-XTX `-sm tensor` production lane **does not execute channels-major**; it retains stock time-major RESHAPE/TRANSPOSE/CONCAT and recurrent-state views. Do not reuse recurrent state across builds/layout modes. Do not claim the old pre-guard b11233 crash remains active.

## Offline checks

- `PYTHONPATH=tools python -m bigcherry patch-rebase-check --source bigcherry --focal-overlay 1263_prbe41_ssm_conv_channels_major`
- Existing `tools/tests/patch/test_1263_prbe41_ssm_conv_channels_major.py`: apply/idempotence, anchor cardinality, backend-op layout and three Qwen model call sites.
- New gate before any tensor-split proposal: source-shaped test for `channels_major=(split_mode != TENSOR)`, exact `declined_split` marker and unmodified time-major RESHAPE/TRANSPOSE/CONCAT/rollback; graph-only per-rank Meta split-state fixture belongs to PRBE114 only if reopened.

## Activation (route-specific)

- **Eligible single-device/layer-split subject:** `BIGCHERRY_PATCH_HIT patch=1263_prbe41 path=ssm_conv_channels_major`; control must not hit.
- **Tensor-split subject:** require `BIGCHERRY_PATCH_HIT patch=1263_prbe41 path=ssm_conv_channels_major_declined_split` when tracing the guarded graph, and **no positive channels-major hit**. A positive marker in tensor split is a correctness/route regression, not a performance success.
- Marker presence alone does not prove kernel execution or speedup; correlate with exact model, build/patch manifest and graph/dispatch.

## Existing measured evidence (do not rerun blindly)

- `evidence/validation.json`: historical contract-passing single-architecture `tierA-qwen4b-q6k` pp512 sessions: gfx1100 b11233 +1.13%..+1.43%, gfx1201 b11126 +2.37%/+2.58%; full-vocabulary correctness reported zero logprob difference. Non-SSM `tierM-gptoss20b-q6k` tg128 is the control. Incomplete/withdrawn sessions are not promotion evidence.
- `docs/evidence/lab-run-summaries/ab-27b-1263.json`: dual-XTX Qwen3.8-27B Q8_0 `-sm tensor` MTP4 pp4096 +0.14% [0.03%, 0.24%], acceptance 84.29% both. The current fallback means this is **not** measured channels-major tensor-split acceleration.
- `docs/evidence/native-vs-patched-historical/ssm1263-activation/RESULT.md`: b11233 pre-fallback tensor-split startup assert; historical failure only.

## Future current-pin qualification

Before any promotion, confirm exact b11552 patch composition, HIP build, backend-op and full-vocabulary/logits/greedy correctness, same-process repeated requests, multi-ubatch, MTP rollback/state lifetime and graph replay. Positive lane: single-device Qwen hybrid GDN pp512 (or separately justified model/shape); negative: non-SSM tg128 and tensor-split fallback. Four independent order-balanced sessions per declared architecture with >=10 paired rounds, appropriate CI and no >1% control regression. PRBE114 owns any future tensor-split expansion and is closed absent a >=3% exposed E2E opportunity; do not modify generic Meta reshape to qualify this package by assertion suppression.


## 2026-10-15 BCOP234 — prefix-marker qualification blocker

`validation/producer.py::_MARKER_REGEX` and `validation.toml::activation.marker-regex` currently match the positive marker as a prefix of `ssm_conv_channels_major_declined_split`. A tensor-split fallback can therefore pass as `executed`. Bound both patterns with `(?![A-Za-z0-9_])`, parse the decline separately, and reject mixed markers. Host-test the actual producer/TOML on positive, declined, absent, both, control-only, wrong-patch and same-process cases. The process-once positive server marker precedes HIP completion and does not prove the separate `llama-bench` route. Require a benchmark-side HIP `ssm_conv_f32<...,channels_major=true,...>` kernel/graph witness, no CPU fallback and per-ubatch CONCAT/CONT accounting. PRBE41 owns the full correctness/performance gate; no GPU lane is queued.
