---
id: PRBE41
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:18.783942+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P3
---

# AMD-SSM-001: Channels-major SSM_CONV input mode

## Description (historical; superseded by authoritative BCOP234 gate)

**2026-10-11 authoritative disposition (b11474):** Patch `1263_prbe41_ssm_conv_channels_major` exists, remains `untested` in patch.toml, and is **single-device HIP only for positive activation**. It is not an unimplemented API proposal. Its model selector uses `model.split_mode() != LLAMA_SPLIT_MODE_TENSOR`; tensor split deliberately builds the pristine time-major graph and emits `ssm_conv_channels_major_declined_split`. Historical text in Notes describes superseded design states. Do not promote tensor split or assume a new channels-major path is active there. No implementation is authorised until the current-pin gate below passes.

**Eligibility/novelty:** Last substantive independent PRBE41 plan commit was 2026-10-08 06:47 UTC; the 1263 patch's 2026-10-09 04:20 UTC move was mechanical. No independent PRBE41/1263 code, PR, submission, queue or evidence update appeared within 2026-10-10 15:10 to 2026-10-11 03:10 UTC on the inspected active branches. Recent BCOP124-128 covered other capabilities. Active QFP49/1360/1361 Meta scheduler and cross-card work, QFP50/1359 prefill, QFP17/1330 attention and QFP41/1356 dispatch are protected; do not modify their implementation, plans or queued lanes.

## Steps (historical sketch)

1. **Use the existing 1263 implementation**, not a new SSM_CONV API/dispatcher. Confirm clean composition on pinned b11474 and inspect the actual graph/markers on one physical GPU (gfx1100 first, gfx1201 second).
2. **Close the invalid tensor-split positive lane:** assert `ssm_conv_channels_major_declined_split` and absence of the positive `ssm_conv_channels_major` kernel marker under `-sm tensor`. A tensor-split positive campaign is not an eligible 1263 experiment. PRBE114 owns any future Meta axis/split propagation; do not change it during active QFP49 work.
3. **Admit only observed removable cost:** capture actual `GGML_OP_CONT`, `GGML_OP_CONCAT`, SSM_CONV launch count, exposed wall time, graph/replay behavior and copy bytes per layer/ubatch. Retire the single-GPU candidate if exposed removable cost is below 2.9126% of E2E or a native/upstream mechanism already eliminates it.
4. **Run a bounded correctness gate before timing:** direct CPU-vs-HIP SSM_CONV (time-major and channels-major) with d_conv 3/4/9, short/long tokens, multiple sequences, F32 reference, saved conv-state and speculative K rollback; repeated same-process requests, graph recapture/replay, logits/KLD, greedy and MTP acceptance. Never reuse state buffers across different layouts.
5. **Qualify only if the cost gate survives:** compare b11474 control vs b11474+1263 (only variable) at pp512/1024/4096 with ubatch explicitly recorded; tg128 and non-SSM controls. Four independent sessions per architecture, >=10 paired order-balanced ABBA rounds/session; promote only CI95-low >=3% E2E and <=1% control regression with complete correctness. Otherwise terminally reject that shape/architecture; do not queue work already running.

## Detailed Solution & Technical Design (historical sketch)

**Current code/data flow, not proposed code:**
- `src/models/delta-net-base.cpp::llm_build_delta_net_base::build_conv_state` takes a `channels_major` boolean. `src/models/qwen35.cpp`, `qwen35moe.cpp`, `qwen3next.cpp` select it from `model.split_mode()`.
- In single-device mode, recurrent state is reshaped `[C,K-1,S]`; `qkv_mixed` is already `[C,T,S]`; `ggml_concat(...,1)` creates `[C,K-1+T,S]`. `ggml_ssm_conv_channels_major` preserves output `[C,T,S]`. The stock tensor-split branch instead reshapes `[K-1,C,S]`, transposes QKV, concatenates on axis 0 and calls stock `ggml_ssm_conv`.
- `ggml/src/ggml.c::ggml_ssm_conv_impl` stores layout in `op_params[0]`. CPU `ggml_compute_forward_ssm_conv_f32` and HIP `ggml/src/ggml-cuda/ssm-conv.cu::{ssm_conv_f32,ssm_conv_long_token_f32}` select input channel/time strides. CUDA support predicate gates channel dimension to multiples of 128. No separate allocator, scheduler or cache is needed.
- The channels-major kernel reads input `src[(seq*(T+K-1)+time)*C+channel]`; time-major reads `src[(seq*C+channel)*(T+K-1)+time]`. For each output `(seq,t,c)`, sum `src[seq,t+k,c] * weight[c,k]` over `k=0..K-1` with the existing F32 arithmetic and optional SiLU. State update for accepted token slot `t` must copy exactly the last `K-1` time steps from that slot's logical prefix, in the chosen layout, before reuse.
- `1263/patch.py::_TIME_MAJOR_FALLBACK` explicitly preserves the original recurrent-state view/CPY and MTP rollback slots under tensor split. Its single-GPU state layout is **not interchangeable** with the stock layout.
- CANN/Hexagon/OpenCL/SYCL/Vulkan/WebGPU reject unsupported channels-major op mode. **Metal support rejection from AMD #52 is absent from 1263**, so a build that can schedule Metal must be gated out until its `supports_op` checks the layout; do not claim generic non-HIP correctness or silent safe fallback.
- `1263/validation/producer.py` already owns full-vocabulary backend-reference comparison, marker check, pp512 positive and non-SSM tg128 control. Reuse it; ensure build/source/model/physical-device identities and subject marker correspond to the measured process. A once-per-process marker alone is not proof every benchmark lane executed the kernel.

**Evidence classification:** The existing `1263/evidence/validation.json` has 24 historical b11126/b11233 records, some withdrawn in `evidence/withdrawn.json`. At b11233, four validated gfx1100 pp512 sessions show +1.1268% to +1.4341% geometric throughput, with CI95-low +0.8969% to +1.1974%. The best historical gfx1201 CI95-low across all 24 records is +2.5859%; **none reaches the 3% promotion floor**. These are measured historical outcomes, not b11474 qualification or a new speedup. A prior fork's gfx1151 pp4096 roofline (CONCAT 1166->34 ms; E2E 3825->2717 ms) and pp512 +5.3% cannot be extrapolated to gfx1100/gfx1201. The fork's CONT line was 10->10 ms; avoid calling its improvement a measured 221 ms transpose saving.

**Upstream/fork decision:** [AMD-Ecosystem #52](https://github.com/AMD-Ecosystem/llama.cpp/pull/52) is the implemented layout reference (merged 2026-07-16); no equivalent `ggml_ssm_conv_channels_major` was found in current upstream code search. Open [llama.cpp #30149](https://github.com/ggml-org/llama.cpp/pull/30149) tiles **Vulkan** CONCAT of a transposed source; it is a different baseline and does not replace HIP 1263. [vLLM causal_conv1d](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py) uses explicit token/channel strides and speculative state offsets; [SGLang KDA](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kda_backend.py) passes a transposed QKV view into a fused causal conv. Reuse their layout/state-invariant test ideas, not their CUDA/Triton performance claims.

## Code Samples & Guidance (historical sketch)

```text
if split_mode == TENSOR:
    build_stock_time_major_graph()       # existing 1263 fallback, not positive lane
else:
    state = reshape(state_bytes, [C,K-1,S])
    inp   = concat(state, qkv[C,T,S], axis=1)
    out   = ssm_conv_channels_major(inp, weight)
    publish_saved_state_from_logical_prefix(inp, accepted_slot)
# no mixed-layout state reuse; no unsupported-backend dispatch
```

**Cheapest discriminator completed in this audit:** source-static checks 10/10; deterministic host layout/reference convolution + state-tail equivalence 144 cases, zero mismatches. These test the addressing contract only, not the actual patched C++/HIP binary. No new hardware run, build, repo pytest or GPU benchmark was performed.

## Files

Authoritative: `docs/planning/active/patching-rdna-boost-experiments/PRBE41.md`. Implementation: `engines/llamacpp/patches/1263_prbe41_ssm_conv_channels_major/{patch.py,patch.toml,validation.toml,validation/producer.py,TESTING.md,SUMMARY.md,evidence/validation.json,evidence/withdrawn.json}`. Code anchors: `ggml/src/ggml.c`, `ggml/src/ggml-cpu/ops.cpp`, `ggml/src/ggml-cuda/{ssm-conv.cu,ggml-cuda.cu}`, `src/models/{delta-net-base.cpp,qwen35.cpp,qwen35moe.cpp,qwen3next.cpp}`, `ggml/src/ggml-backend-meta.cpp`, `tests/test-backend-ops.cpp`. Meta split investigation remains with PRBE114 (dependency only).

## Validation

**Preflight:** `PYTHONPATH=tools python -m bigcherry patch-rebase-check --source bigcherry --focal-overlay 1263_prbe41_ssm_conv_channels_major`; patch lint and existing backend-op unit tests. The command is a proposed gate, **not executed in this audit**. Require separate `single GPU HIP positive`, `tensor-split decline`, `unsupported backend negative`, `multi-request/MTP saved-state`, `pp512/1024/4096` and `tg128 control` evidence, with physical-device and model fingerprints. Run no overlapping QFP49 hardware campaign.

**Measured decision:** Existing pp512 evidence is insufficient for promotion under >=3% CI95-low. For larger ubatches first prove materialized CONCAT/CONT exposed time and a plausible >=3% E2E ceiling; if absent, close PRBE41 without new implementation or benchmarking. Any future tensor-split work is PRBE114-owned and is not an excuse to bypass the 12-hour exclusion.

## Effort & Risk

Existing patch; incremental work is evidence reconciliation, source-pin composition and single-device correctness/performance qualification, not a new op/kernel family. High risk: recurrent-state layout/rollback, unsupported Metal scheduling, stale graph/capture and Meta split-state semantics.

## Standards

Fail closed on unsupported layouts/backends. Separate measured historical evidence, source facts and untested hypotheses. No source-level patch promotion without b11474 build, numeric/state tests and reproducible hardware A/B. Do not interfere with protected QFP49 Meta scheduler work.

## Acceptance Criteria

Single-device positive marker and no control marker; exact expected tensor/transfer work; full-vocabulary near-lossless logits and greedy/MTP acceptance parity; saved-state and rollback parity; no cross-layout state reuse; graph and same-process repeat safety; CI95-low >=3% pp E2E and <=1% tg/control regression on four independent sessions per architecture. Explicit negative `-sm tensor` decline and unsupported-backend checks. Otherwise record **REJECTED_AT_CURRENT_SHAPE** or **BLOCKED_ON_CURRENT_PIN_TRACE**, not an invented speedup.

## Notes

**Historical log only.** Earlier proposals below predate the 1263 implementation, the explicit tensor-split decline and the 2026-10-11 evidence rebaseline above; do not treat them as current implementation instructions.

Supersedes: RD49
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd49

Supersedes RD49. PRBE21 is a distinct SSM conv_input concat optimization; cross-reference but do not merge.

2026-09-24 relevance at b11126: no existing patch (RD49 grep = no hits). Not upstream-absorbed (ggml_ssm_conv signature at ggml/include/ggml.h:2524 and CUDA kernel addressing at ggml/src/ggml-cuda/ssm-conv.cu:6-56 both read directly and confirmed time-major-only, no layout parameter exists). GPT design requests for this batch (req_6a45d0917b114112, req_3710dbae73fb49be) were unusable (failed/unresponsive) -- this plan was authored directly by the planning agent from verified source. PRBE21 (SSM conv_input concat) is a distinct, separate optimization per the item's own notes -- do not merge.

2026-09-24 GPT review req_e17e0bf5a68c48d5 applied: replaced the implicit-default layout-parameter approach (which violated this project's no-legacy-shim doctrine) with a separate ggml_ssm_conv_ext() function; defined exact channels-major shape/index algebra (d_inner=sx->ne[0], n_t=sx->ne[1]-d_conv+1) for CPU and CUDA/HIP; required either full backend coverage or capability-aware graph-selection before switching callers.

2026-09-25 (4d7117f6): implemented as patch 1263_prbe41_ssm_conv_channels_major, a port of nasone/AMD 33611a98 generated with bigcherry.patch.port_diff (17 files: ggml.h/ggml.c op mode, CPU + CUDA/HIP ssm-conv kernels, other backends' supports_op rejection, qwen35/qwen35moe/qwen3next graph, delta-net-base build_conv_state hand-merged onto b11126, fork test-backend-ops cases). Metal rejection not ported; conv-state layout changes (saved states not interchangeable). Activation marker patch=1263_prbe41 path=ssm_conv_channels_major. Contract PRBE41-SSM-CONV-CHANNELS-MAJOR (owner policy, 4 sessions/arch): full-vocab backend reference on qwen4b, pp512 positive, gpt-oss tg128 control. Rebase check CLEAN on bigcherry and bigcherry-tuning. Hardware sessions queued after the current batch.

## 2026-10-08 b11474 experiment scope

**Rank 7, qualify existing 1263 rather than re-port.** `1263_prbe41_ssm_conv_channels_major` state=untested (patch.toml); b11474 composed targets `ggml/include/ggml.h::ggml_ssm_conv_ext`, `ggml/src/ggml-cuda/ssm-conv.cu`, `src/models/delta-net-base.cpp` and Meta axis0 split/reshape propagation. Input channels-major shape `[d_inner,n_t+d_conv-1,n_seq]`, output `[d_inner,n_t,n_seq]`. Current Meta/tensor-split test failed `GGML_ASSERT(src_ss[0].nr[0] == 1)` before activation (PRBE114 evidence), so qualify single-GPU GDN first and treat multi-GPU as blocked until split-state instrumentation fixes actual axis ratios. Env `BIGCHERRY_SSM_CHANNELS_MAJOR=0|1` proposed opt-in for qualification if existing 1263 has no reliable selector; no implicit legacy default args. Expected gain uncertain (memory coalescing), only on GDN/27B SSM_CONV, not Flash-Next attention path. `queue-env-ab.sh`: 27B GDN 8K/24K/98K and single-gpu/tensor-split arms, output+recurrent-state parity, PPL, capture/replay, marker, VRAM, separate source-split traces. Do not claim multi-GPU validated.

Recheck every historical b11126 anchor on the composed b11474 source before coding; no GPU run or patch implementation is claimed by this plans-only triage.

## Change Log

- 2026-10-08 (triage): Kept experiment pending (P3); source/shape, coverage against #29901/1202/1253, env gate, expected effect and separated hardware test defined above.

- 2026-09-09T10:56:18.783942+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:34.146919+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.314431+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.077374+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:04:24.733208+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- chg_20260910_030451_carried-forward-the-detailed-s_2071
- 2026-09-10T03:04:51.318883+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:05:49.720099+00:00 (updated-by): Updated: section:acceptance_criteria
- chg_20260910_030619_removed-migration-placeholder_7703
- 2026-09-10T03:06:19.327420+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T04:45:40.709189+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- 2026-09-24T05:07:59.505020+00:00 (updated-by): Updated: section:description, section:steps, section:notes
- 2026-09-25T10:49:09.867182+00:00 (updated-by): Updated: section:notes
- 2026-09-25T10:49:12.783952+00:00 (state-transition): State: pending → in_progress

## 2026-10-14 evidence/route reconciliation (authoritative over earlier historical wording)

**1263 remains `untested` in `patch.toml` for the current production pin; historical validated receipts do not automatically promote it.** The current `patch.py` explicitly gates all three Qwen hybrid model call sites on `model.split_mode() != LLAMA_SPLIT_MODE_TENSOR`. Tensor split builds the original time-major graph, emits `ssm_conv_channels_major_declined_split` when tracing, and must **not** be required to emit the positive channels-major marker. The b11233 pre-guard tensor-split crash is historical, not the expected behavior of the guarded patch.

The existing `evidence/validation.json` includes contract-passing, single-device `tierA-qwen4b-q6k` pp512 sessions: gfx1100 b11233 approximately +1.13% to +1.43%; gfx1201 b11126 +2.37% and +2.58%. Correctness receipts report zero full-vocabulary logprob difference; `tierM-gptoss20b-q6k` tg128 is a non-SSM control. Do not conflate these with dual-XTX Qwen3.8-27B `-sm tensor`: `docs/evidence/lab-run-summaries/ab-27b-1263.json` reports only +0.14% pp4096, and the current guarded route does not remove its transpose. All are historical measurements on older pins, not new b11552 GPU qualification.

**Ownership:** PRBE41/1263 retains single-device channels-major kernel/layout and contract ownership. PRBE114/BCOP195 closes the unqualified generic Meta RESHAPE/VIEW tensor-split rewrite; no second scheduler or split-state cache. Preserve the mode-specific conv-state lifetime, explicit fallback, full-vocab/MTP correctness, and current-pin composition/activation checks before any future promotion.


## 2026-10-15 BCOP234: route-bound activation and prefill qualification (authoritative)

**Disposition:** Patch 1263 remains `untested` on b11552. This section supersedes the historical `ggml_ssm_conv_ext` and default-argument sketches below: the actual implementation already uses `ggml_ssm_conv_channels_major`. Do not add another layout selector, scheduler, cache, allocator, or telemetry system. No new GPU campaign is authorized until activation evidence is repaired.

### Code-level finding: false-positive activation contract

`engines/llamacpp/patches/1263_prbe41_ssm_conv_channels_major/patch.py::_CALL_NEW` sets `channels_major = model.split_mode() != LLAMA_SPLIT_MODE_TENSOR` in `qwen35.cpp`, `qwen35moe.cpp`, and `qwen3next.cpp`. `_TIME_MAJOR_FALLBACK` preserves original RESHAPE/TRANSPOSE/CONCAT(axis 0), state views and copy, and emits `path=ssm_conv_channels_major_declined_split`. However `validation/producer.py::_MARKER_REGEX` and `validation.toml::activation.marker-regex` both match the **unbounded prefix** `path=ssm_conv_channels_major`. The producer's `re.search` accepts the decline marker; with no control hit, `ActivationEvidence(status="executed")` and the promotion trigger can pass on a time-major fallback. This is a demonstrated source/host **prospective evidence defect**, not a reproduced GPU failure or retroactive invalidation of older correctly routed single-device receipts.

The positive marker is emitted in `ggml/src/ggml-cuda/ssm-conv.cu::ggml_cuda_op_ssm_conv` before successful kernel completion, guarded by `std::once_flag` (process-once). The correctness server and separately measured `llama-bench` processes do not share an execution witness. The graph's channels-major `sx=[d_inner,n_t+d_conv-1,n_seq]`, recurrent state `[d_inner,d_conv-1,n_seq]`, output `[d_inner,n_t,n_seq]` and CONCAT(axis 1) use CPU `chan_nb/time_nb` and HIP short/long `stride_xc/stride_xt` addressing. Tensor-split remains time-major with CONCAT(axis 0). Layout must be fixed for the model/context/graph/state lifetime, including MTP rollback and saved-state restore.

### Minimal implementation-ready corrective gate (PRBE41/1263 owner)

1. Change **both** existing regexes to `BIGCHERRY_PATCH_HIT patch=1263_prbe41 path=ssm_conv_channels_major(?![A-Za-z0-9_])` (preserve TOML escaping). Parse `ssm_conv_channels_major_declined_split` separately. Positive admission: `subject_positive && !control_positive && !subject_declined`. Tensor-split negative control: `subject_declined && !subject_positive`. Mixed routes fail closed as `AMBIGUOUS_ROUTE`; never substring-match. No new telemetry.
2. Extend `tools/tests/patch/test_1263_prbe41_ssm_conv_channels_major.py` or patch-local producer tests to exercise the actual TOML and Python patterns on positive, declined, absent, wrong-patch, suffix, both-markers, control-only and repeated same-process logs. The process-once marker cannot prove every request; correlate subsequent requests with existing graph/dispatch traces.
3. Before performance attribution, establish that the **benchmarked** HIP subject launches `ssm_conv_f32<...,channels_major=true,...>` on the declared GPU UUID and has no CPU SSM_CONV fallback. Use existing rocprofv3/graph evidence for per-ubatch CONCAT/CONT nodes, launches, bytes, duration, stream synchronization, capture/replay and removed transpose. A server marker without a benchmark-side witness is `UNPROVEN_EXECUTION`.
4. Direct oracle: CPU time-major vs channels-major vs HIP, `d_conv=3/4/9`, `d_inner=128/256` and actual model width, `n_t=1/2/128/512/2048`, `n_seq=1/>1`, short/long kernels, contiguous strides, same-process multi-request/multi-ubatch, 8K/24K/98K, full-vocab logits/KLD, greedy/MTP acceptance, state rollback, memory safety and graph replay. Tensor-split must decline and preserve the original graph; non-SSM gpt-oss tg128 and gfx1030 are negative controls.
5. Only if source/route/correctness gates pass, compare composed b11552 baseline vs +1263 on single gfx1100/gfx1201, identical quant, batch/ubatch, model, context, clocks and device. Start with pp512; pp2048/4096 only when traces show exposed CONCAT/CONT headroom. Four independent order-balanced sessions/architecture, >=10 paired ABBA rounds/session; CI95-low >=3% E2E prefill, <=1% decode/control regression, exact work/bytes and numerical/state parity. Reject/default-off if removable E2E ceiling <3%, execution unproven or any state failure.

### Existing evidence, upstream and ownership

Historical `tierA-qwen4b-q6k` pp512 receipts: +1.13%..+1.43% gfx1100 (b11233), +2.37%/+2.58% gfx1201 (b11126), zero reported full-vocab logprob difference; not b11552 results. Dual-XTX Qwen3.8-27B tensor-split pp4096 +0.14% is fallback, not channels-major acceleration. [AMD-Ecosystem #52](https://github.com/AMD-Ecosystem/llama.cpp/pull/52) merged 2026-07-16 with external gfx1151 ubatch-dependent gains. New upstream [llama.cpp #30591](https://github.com/ggml-org/llama.cpp/pull/30591), open 2026-10-15, instead fuses a **Vulkan** transposed-input CONCAT via a padded 32x33 shared-memory tile; external R9700 Vulkan results do not justify a HIP port. [vLLM ROCm GDN](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py) distinguishes dimension-first/sequence-first conv-state and uses AITER fused decode, a state-lifetime reference rather than GGML replacement.

PRBE41/1263 owns single-device layout and validation. PRBE114's generic Meta tensor-split rewrite remains closed; PRBE18/PRBE21 own distinct qkv/CONCAT fusion. Do not touch active MEN10/RHP/Radiance.
