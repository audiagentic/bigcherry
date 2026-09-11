# 0840 (GP03): size-adaptive internal/RCCL AllReduce provider dispatch

## Scope

Adds `GGML_CUDA_ALLREDUCE=hybrid`: a 4th AllReduce provider that brings up
both RCCL and BigCherry's own internal AllReduce pipeline (patch 1001,
already validated) simultaneously, then dispatches per call based on
`ggml_nbytes(tensors[0])` against the internal pipeline's own real
copy-engine threshold -- below it, tries internal first (falling through to
RCCL on failure); at or above it, goes straight to RCCL.
`comm_ctx->provider_name` is set per call right before each sub-provider
runs, so patch 0830's existing telemetry seam attributes every call
correctly with no separate labeling change needed. Init also forces the
internal pipeline to exact F32 rather than trusting
`GGML_CUDA_AR_BF16_THRESHOLD`'s own lossy default.

## Why

Patch 1001 (validated) alone is a large decode win (+17.33% TPS, MTP
completion-bench) but a severe prefill regression (-32% to -34%, real
llama-bench pp512/pp2048/pp4096) -- `GGML_CUDA_ALLREDUCE` can only pick one
provider for a whole server session, so neither `internal` nor `rccl` alone
is safe to ship as a blanket default. Real telemetry (0830's
`reduction_bytes` field, captured on real traffic) found a clean, 10x,
zero-overlap separation between decode and prefill message sizes -- decode
tops out almost exactly at the internal pipeline's own default copy-engine
threshold, so dispatching on that real threshold (not a second,
independently-tunable constant) keeps the policy from ever routing into the
exact regime the prefill regression implicates.

## Upstream / provenance

Local, BigCherry-authored (not an upstream port) -- plan item GP03, opened
following `dev-gpt-agent` design guidance after the prefill-regression and
reduction-byte-histogram evidence in `1001_hip_internal_allreduce`'s own
SUMMARY.md. Consolidates and **replaces** an independently-written
competing implementation, `1243_gp03_size_adaptive_allreduce_dispatch`
(deleted 2026-09-02 after this patch reproduced and exceeded its real
hardware numbers under a proper baseline comparison) -- this is now the
sole implementation for size-adaptive AllReduce dispatch, per explicit
operator direction ("let's consolidate our patches so we have one
implementation").

## Real hardware evidence (2026-09-02, Brutus)

Built 3 independent binaries from a clean pinned llama.cpp checkout in
isolated scratch clones (not the shared tree, no risk to concurrent
sessions): true-upstream-native baseline, and `bc-hybrid`
(0100+0830+1001+this patch, all verified to apply cleanly and idempotently
against the true pinned source).

**Synthetic pp/tg** (llama-bench, `-b 2048 -ub 512`, r=3, `{0,1}` 2x RX
7900 XTX): hybrid matches or slightly exceeds native RCCL at pp512 (1478.29
vs 1464.80) and tg128 (37.25 vs 34.62), roughly ties native RCCL at
pp2048/pp4096. Layer-split beats both native RCCL and hybrid at
pp2048/pp4096 by 10-19% -- confirmed NOT a defect in this patch (native
RCCL shows the identical shortfall against layer-split at the same sizes;
a real upstream tensor-split-prefill characteristic of the hardware, not
introduced by this patch).

**Real MTP completion-bench** (the metric that resolves the pp-synthetic
concern above -- production server flags, real `mtp-27b-v1` corpus, 48
requests/arm, `predicted_tps` mean, against the explicit required baseline
per operator direction: "baseline is native llamacpp rccl"):

| arm | predicted_tps | vs layer-split | vs native RCCL |
|---|---|---|---|
| layer-split | 44.99 | -- | -31% |
| native-meta | 53.04 | +18% | -19% |
| native-rccl (baseline) | 65.50 | +46% | -- |
| **hybrid (this patch)** | **68.03** | **+51%** | **+3.9%** |
| internal-only | 68.02 | +51% | +3.9% |

Draft acceptance (0.487-0.491) and mean accepted length (2.93-2.95) are
consistent across every arm -- no correctness/behavioral regression in
speculative decoding from any provider choice. Real MTP serving is
decode-dominated (many small verification-reduction calls, not
prefill-sized ones), which is why layer-split -- the pp-synthetic winner at
large sizes -- is actually the WORST arm under the workload that matters.
Under real serving, this patch beats every baseline including layer-split
by the largest margin of any arm, and beats the explicit required baseline
(native RCCL) by 3.9%.

**Reproduced across 2 RCCL versions** (ROCm 7.2.4/RCCL 1.0.70204 and ROCm
7.14/RCCL 2.30.4) and **confirmed on both 2-GPU `{0,1}` and 3-GPU
`{0,1,2}` topologies.**

**Heterogeneous-topology safety (the final blocker, now closed)**: `{0,3}`
(device-3-inclusive, RCCL-unsafe per HI138's PCIe-atomics findings) was
found to hard-crash before GP02's admission predicate was wired in -- this
patch's own secondary `ncclCommInitAll()` had the identical device-3 crash
gap the competing 1243 implementation had. Fixed by wiring in patch 1225's
(GP02's) shared `ggml_backend_cuda_comm_rccl_admission_ok()` predicate
before this patch's `ncclCommInitAll()` call (confirmed present in the
real patch source, line ~273). Real hardware re-confirmation: `{0,3}` now
fails closed cleanly (falls back to the internal pipeline, no crash) and
`{0,2}`/`{1,2}` remain correctly admitted through RCCL.

Full evidence: `artifacts/gp03-validation/` on Brutus (`bc-native`,
`bc-hybrid` scratch builds; `mtp-results/*.completion.jsonl` + server/bench
logs for all 5 arms).

## Lifecycle: promoted to `validated` (2026-09-11)

Requested an explicit GPT solution-approval decision on whether the
evidence above supports promotion now that its dependency 1225/GP02 is
itself validated. **Verdict: APPROVE** (dev-gpt-agent, session
`ses_ba0d0982ae4b472d`, `req_d11e4b5149324f20`): "Evidence closes the
material qualification gaps: real production workload beats the required
native-RCCL baseline, results reproduce across RCCL versions/topologies,
and the former heterogeneous-topology crash blocker is now fail-closed via
validated dependency 1225. Missing README.md is documentation debt, not a
validation blocker given the recorded GP03/SUMMARY evidence." `patch.py`'s
`STATE`, `patch.toml`'s `state`, and `SUMMARY.md`'s `Status` corrected to
`"validated"` together; this README authored to close the documentation
gap GPT flagged.

## Known limitations

- No `validation.toml` adapter exists (no bound Experiment Contract).
- **Resource-cost accounting is explicitly deferred, per the patch's own
  docstring**: the internal pipeline's init is not yet split so hybrid
  mode skips paying for its large-message (32MB-class host/device staging)
  buffers it will never route through below the copy threshold -- accepted
  for this validation slice per GPT's own guidance at the time, with the
  resource cost (VRAM, pinned host allocation, clean teardown) explicitly
  flagged as "to be measured, not assumed, before wider adoption."
- The `GGML_HIP_REDUCE_RCCL_THRESHOLD`-adjacent dispatch boundary is
  derived from the internal pipeline's own real copy-engine threshold
  rather than independently swept/tuned.
