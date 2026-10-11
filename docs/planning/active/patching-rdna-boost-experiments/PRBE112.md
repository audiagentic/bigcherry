---
id: PRBE112
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-25T23:16:25.986252+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Disposition of rejected RD39-42/1215: isolate MoE overlap from generic QKV scratch

## Decision (2026-10-09; authoritative)

**Keep 1215 rejected. Do not restore the four-part port or schedule another broad 1215 A/B.** Current pinned-source evidence reduces its unresolved performance question to a narrow, falsifiable **RD41 generic QKV scratch versus RD42 MoE-only overlap** boundary. Upstream has already absorbed the per-stream handle/workspace mechanism; a new RD39/RD40 implementation would duplicate it. PRBE112 is the only owner of any bounded rework/disposition. PRBE33/34 are completed/deprecated historical owners, not additional implementation queues. PRBE35/1216 is independently eligible as a join-fusion correctness guard on native concurrent regions and must not be changed under this item.

**Measured BigCherry evidence, not a hypothesis:** eight first-party b11126 records in `engines/llamacpp/patches/1215_rd394041_amd_stream_moe_overlap/evidence/validation.json`, four sessions per architecture, ten paired rounds each, `GGML_CUDA_GRAPH_OPT=1`:
- gfx1100 MoE tg128 subject geometric effects **-2.84% to -3.30%**, dense negative-control **-1.59% to -2.21%**.
- gfx1201 MoE **-8.05% to -8.18%**, dense **-6.83% to -6.86%**.
- The raw full-vocabulary correctness check passed for the recorded lane; the performance contract did not. Dense regression cannot be caused solely by RD42's shared-expert fork, because its name/pattern predicate does not activate on the dense control. The specific cause of the dense regression is **not yet measured**.
- Older gfx1100 +2.38% ten-round ON result and fork +7.4% tg128 are historical/different qualification contexts. PRBE108's later interleaved -5.93% check already disproved an ordering-bias-only explanation. Neither historical gain supersedes the eight rejected contract records. Never pool these runs across pins/compositions.

## Current code path and lifetime audit

At pinned b11474, `ggml/src/ggml-cuda/common.cuh::ggml_backend_cuda_context::cublas_handle()` and per-stream pools are native. Upstream llama.cpp [#26574](https://github.com/ggml-org/llama.cpp/pull/26574) introduced per-(device,stream) handles and static workspaces, with stream bound at handle creation. In `engines/llamacpp/patches/1215_rd394041_amd_stream_moe_overlap/patch.py`, the old handle-widening/method edits are **removed**. No active Edit in this package rewrites `ggml_cuda_op_mul_mat` to select a stream. The patch's RD39/RD40 title and the frozen contract rationale describe historical provenance, not current incremental work.

Remaining live edits and ownership:
1. `rd3942-scratch-fields` (`common.cuh`) plus `rd3942-destructor` (`ggml-cuda.cu`) own one `concurrent_scratch` backend buffer and its teardown.
2. `rd39-stream-fallback-join-branch` and `rd39-stream-fallback-post-fusion` change `ggml_cuda_graph_evaluate_and_capture` to route nodes absent from a concurrent event's `stream_mapping` to stream zero; this is needed for RD42's main-stream routed branch, but is not a general replacement for native QKV scheduling.
3. `rd3942-concurrent-groups-decl`, `rd3941-drop-interleave-decls` and `rd3941-scratch-buffer-and-rd42-moe-overlap` change `ggml_backend_cuda_graph_optimize`. **Crucially, the patch replaces native branch interleaving for ALL existing QKV concurrent groups**, before it scans for a MoE shared-expert diamond. It then adds a MoE-specific auxiliary group only when a single-token `GGML_OP_ADD` joins named `ffn_moe_out` and `ffn_shexp` branches with a proven common ancestor.
4. For each group, scratch footprint is `sum(GGML_PAD(ggml_nbytes(non-view/non-noop), 128))`; one buffer of the **maximum** group footprint is allocated and the same base reused across groups. That is correct only when the corresponding regions do not overlap in lifetime and every prior graph executable referencing the old allocation is quiesced/recaptured before buffer growth/free. Neither condition is established by a host-only footprint calculation. Pointer/graph replay safety is an explicit future correctness gate, **not an observed memory error**.
5. `1216_rd43_concurrent_join_fusion_guard` protects the join from fusion. Its `patch.toml` currently has `requires=[]`, so its standalone native-QKV path is separate from 1215. A hypothetical RD42-only variant must explicitly include 1216 as a companion in the subject composition; do not revive the historical 1215↔1216 dependency cycle.

**Causal inference, not measured proof:** RD41's generic QKV branch-order/allocation rewrite is the smallest current candidate that can explain the dense-control regression; RD42's MoE-only detector cannot explain that lane. An upstream-native QKV path with MoE-only scheduling is the least duplicative alternative. Do not attribute the regression to cuBLAS handle creation, register pressure or memory copies without a trace.

## Cheapest discriminator and implementation boundary

**Stage 0: no hardware reservation.** Use existing patch-local producer and validation artifacts. Static-check `patch.py` edit inventory against b11474: native handles and per-stream pools exist, all QKV groups enter `concurrent_groups`, MoE predicate is name/shape-gated, and resize frees the old scratch. Run a disposable host fixture for 128-byte padding, group offsets and overlapping-group counterexample; this proves capacity arithmetic only, not GPU safety. If current upstream already exposes a safe native MoE two-way fan-out with equivalent lifetime handling, close PRBE112 as upstream-superseded without a new patch.

**Stage 1 (only if Stage 0 leaves a real gap):** disposable offline patch composition, NOT production or a second scheduler. Retain native `ggml_backend_cuda_graph_optimize` QKV interleaving and existing `ggml_cuda_concurrent_event` / `stream_mapping` / fork/join machinery. Add only a bounded MoE detector for the exact `GGML_OP_ADD(ffn_moe_out, ffn_shexp*)` single-token diamond, with explicit reachability and output ownership checks. Keep routed nodes on stream zero, shared branch on one existing auxiliary stream, and join at the original ADD. Do not globally redirect all QKV tensors into a new scratch buffer. If MoE auxiliary temporaries require dedicated storage, prove disjointness with native per-stream pools first; otherwise allocate only MoE-scoped storage before capture, with stable address through replay and synchronized graph destruction before resize. Avoid a second dispatch table, allocator, queue or runtime flag. Reject if this cannot be expressed as a narrow patch against native concurrency.

**Required host fixture/selector cases:** (a) dense/QKV-only graph -> no new group and native node order unchanged; (b) named shared-expert decode diamond -> exactly one auxiliary event and original join; (c) prefill/multirow -> no fork; (d) missing names, extra consumer, shared-source alias, overlapping concurrent regions, changed graph topology -> fail closed; (e) two groups with different padded footprints -> no alias while live; (f) grow/shrink across repeated same-process requests -> graph/pointer lifetime invariant. Static selector pass is necessary, not sufficient.

## Hardware qualification, only after offline discriminator passes

Use the existing `validation/producer.py`, `full_vocab.compare_servers`, `run_paired_llama_benchmark`, `config/experiment-contracts.toml` and `rocprofv3`; do not create another benchmark framework. Do not collide with QFP36/41, Meta-cache, patch-ablation, MTP, Radiance or queued lab work.

Frozen controls for any **new, separately identified** candidate: A = b11474 native graph-opt + standalone 1216; B = A + minimal RD42-only mechanism; C = rejected 1215+1216 solely as a diagnostic reference (never a promotion arm). Compare `GGML_CUDA_GRAPH_OPT=1` in A/B, with `=0` as negative activation control. MoE positive = `tierM-qwen35b-a3b-moe-mtp` single GPU, decode tg128; dense negative = `tierA-qwen4b-q6k`. Qualify gfx1100 first, gfx1201 independently; gfx1030 is correctness/negative-only unless its own memory and benefit gates pass. Tensor split and layer split are out of admission scope (historical layer-split -4.4%; no normal P2P). Do not extrapolate to dual-XTX+R9700 or Flash-Next without model-specific activation.

Telemetry: same-run marker subject-hit/control-miss, actual fork/join count, stream id, QKV concurrent-event count, scratch bytes/allocations, graph capture/update/recapture count, and rocprof queue intervals/overlap. For correctness require raw full-vocabulary bit identity on the frozen greedy lane, no dropped joins, graph on/off, multi-ubatch, 100K+ context, repeated same-process requests and growth/replay without stale pointers. Attribute performance only when **timed** A/B requests demonstrate activation; an independent server probe does not prove the timed kernel ran.

Promotion: frozen `improvement_no_regression_v1` (four independent sessions per architecture, >=10 paired rounds/session, session-bootstrap CI95-low >0% positive gain, dense-control regression CI95-high <=1%, exact correctness and activation). Fail any control, lifetime or architecture gate -> reject/retire; do not queue an open-ended investigation. If a pre-profile shows eligible MoE-overlap critical-path fraction too small to beat the measurement floor, close without a hardware campaign.

## Ownership / upstream / external

- PRBE112: sole rework-versus-retire decision for rejected 1215. PRBE33 (native per-stream pool) and PRBE34 (historical RD42) remain completed/deprecated; do not reopen duplicate patches. PRBE35/1216 owns standalone join correctness; no change to its plan, package or contract. The rejected 1215 package and evidence remain immutable.
- llama.cpp [#16991](https://github.com/ggml-org/llama.cpp/pull/16991) supplies native concurrent QKV regions; [#26574](https://github.com/ggml-org/llama.cpp/pull/26574) absorbed the handle/workspace component. [AMD fork #36](https://github.com/AMD-Ecosystem/llama.cpp/pull/36) remains provenance, not a promotion baseline.
- vLLM's [ROCm dual-stream request #48111](https://github.com/vllm-project/vllm/issues/48111) explicitly reported HIP-graph integration was not solved by a one-line enable; its gfx950/ATOM measurements are not RDNA3/4 evidence. [vLLM MoE runner](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/runner/moe_runner.py) launches the shared expert before routed dispatch and joins before consuming the result: useful ordering reference, not a portable kernel or proven BigCherry gain.

## Validation record (this documentation audit)

12/12 source-static/host assertions passed against the current 1215 patch: removed cuBLAS edit, QKV/MoE group coverage, 128-byte footprint model (groups 1152/256/896 bytes; max allocation 1152 bytes), stream-fallback and MoE gate checks, and buffer-growth/free identification. The overlapping-group case is a **negative lifetime model**, not a reproduced GPU alias. No repository pytest, compilation, HIP run, profiler capture or new hardware measurement was performed.

## Historical change log / ledger

- 2026-09-25: PRBE112 created to diagnose the rejected 1215 regression.
- 2026-10-08: triage retained PRBE112 as sole rework owner.
- 2026-10-09: implementation-level reconciliation supersedes the former broad "profile everything then rework RD39/40/41/42" steps; only the narrow discriminator above remains.
- chg_20260925_232219_first-patch-promoted-to-the-pr_2756
- 2026-09-25T23:22:33.660056+00:00 (updated-by): Updated: section:ledger-events
