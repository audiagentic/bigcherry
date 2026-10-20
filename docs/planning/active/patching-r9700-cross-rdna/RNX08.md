---
id: RNX08
order: 8
plan: patching-r9700-cross-rdna
state: pending
created-at: '2026-10-03T01:33:25.968209+00:00'
breadth: ''
skill: advanced
created-by: codex
work: M
priority: P2
---

# R9X08 — PLE, norm+RoPE, and transpose microfusion portfolio

## 2026-10-20 kernel audit — HIP Q/K norm+RoPE→FWHT in-kernel rotation (BCOP331)

**Disposition: HOLD, no patch or hardware queue.** New narrow HIP gfx1100/gfx1201 capability: fold an eligible 64/128/256-wide Hadamard transform into the already-native RMS_NORM→MUL→ROPE kernel. This is not a generic norm+RoPE re-port, a Vulkan wide FWHT change, or Q8_0 SET_ROWS. The important first-party source fact is **route exclusion**: Qwen3.5/3.6/3.8 and Flash-Next use multi-axis ROPE, which the candidate NORMAL/NEOX matcher does not support. No BigCherry activating graph or incremental performance gain was measured.

### Pinned implementation, dispatch, and physical work

- Pinned llama.cpp b11552 / 23b0202a189c44a54625aadcb37a946dd1d6278d: ggml/src/ggml-cuda/ggml-cuda.cu functions ggml_cuda_should_fuse_rms_norm_mul_rope, ggml_cuda_can_fuse and ggml_cuda_try_fuse already admit F32 contiguous-row RMS_NORM→MUL→ROPE, NORMAL/NEOX only, n_offs=0, with existing graph edge and ggml_cuda_check_fusion_memory_ranges checks. ggml/src/ggml-cuda/rope.cu functions rms_norm_mul_rope_f32, rms_norm_mul_rope_cuda and ggml_cuda_op_rms_norm_mul_rope_fused launch a 256-thread CTA per row (ncols<1024), use block_reduce with dynamic s_sum and write roped F32. Rejected patch 1004 is already upstream-absorbed; do not revive it.
- ggml/src/ggml-cuda/fwht.cu functions ggml_cuda_op_mul_mat_use_fwht, ggml_cuda_op_fwht_impl and fwht_cuda implement Hadamard-hinted MUL_MAT with F32/F16 contiguous input and F32 output. N64/128/256/512 uses one physical warp per row and four rows per CTA; N1024–8192 uses a different 256-thread LDS implementation. This slice only considers N64/128/256, not wide FWHT.
- [RDNA fork #172 patch 0001](https://github.com/stew675/llama-cpp-rdna-boosts/pull/172/files) adds a five-node matcher RMS_NORM→MUL→ROPE→RESHAPE→Hadamard MUL_MAT, and fwht_n specializations of rms_norm_mul_rope_f32<256,...>. Roped F32 values go into a shared N-float tile; after one block barrier the first physical warp runs the standalone fwht_cuda butterfly code, moved into a shared fwht.cuh helper, then writes the final F32 Hadamard output. Other CTA lanes idle during the butterfly. The fork edits ggml-cuda-try-fuse.cu after an upstream file split; pinned b11552 uses ggml-cuda.cu, so do not apply the patch verbatim. Fork #172 was accepted into downstream rdna-boosts r78/block 10 on **17 Oct 2026**, not merged into ggml-org llama.cpp.
- Baseline R head rows of width N: R norm+RoPE CTAs and ceil(R/4) FWHT CTAs, two launches, intermediate roped F32 write plus FWHT read. Candidate: R CTAs and one launch, same R*N normalization, RoPE, butterfly and final F32 outputs, with logical 8*N*R bytes of intermediate global traffic replaced by R LDS tiles of 4*N bytes and one block barrier. These are **logical** bytes/CTA counts, not measured HBM, PCIe or occupancy. For N256 the tile is 1024 B plus existing reduction scratch. Profile wave32/64, VGPR/SGPR, spills, LDS, occupancy, idle lanes, synchronization, ISA, physical bytes, launches and output counts; no normal P2P is assumed on dual gfx1100 + gfx1201 + gfx1030.

### Exact safety, route and ownership gates

- Admit only consecutive five-node graph with exact edges, one ROPE/RESHAPE consumer, non-output RESHAPE, supported Hadamard hint, F32 same-device/stream, contiguous rows, matching width N in {64,128,256}, NORMAL/NEOX and native zero-offset rotary eligibility. Prove full-window graph reachability and all output/source allocation spans with the existing ggml fusion and memory-range helpers, not only the fork's custom overlap loop.
- The fork permits exact output==norm-input alias after each CTA reads its own full row before the barrier. Accept this **only** for identical byte spans and non-overlapping row partitions with no other live consumer; reject shifted/partial overlap, different type/stride, alias to MUL/position/frequency inputs, and graph-output/retained-view lifetime conflicts. Keep native two-launch fallback. All K/V cache writes and other consumers must still execute; never infer a speedup from missing work or skipped reshape users.
- Pinned src/models/qwen4exp.cpp::build_layer_attn uses ggml_rope_multi/IMRoPE for Qwen3.8 Flash-Next; Hadamard Q/K/V rotation exists on some KV paths but is insufficient to activate this NORMAL/NEOX matcher. The fork maintainer explicitly reports Qwen3.5/3.6/3.8 as ineligible. Do **not** widen IMRoPE under this item: PRBE15 owns multi-axis position/scatter semantics. An actual Gemma4 BigCherry graph may qualify, but prove its deployed checkpoint, KV dtype, physical rank, chain and consumer shape first.
- RNX08 remains the norm+RoPE specialization owner. Existing 1004 owns generic ancestral fusion provenance, fwht.cu native transform, PRBE15 IMRoPE, PRBE05/1307 Q8_1 cache (fork #172 patch 0002 is a *separate* producer question), PRVP06 Vulkan wide FWHT and QFP13 launch census. No new dispatch table, scheduler, allocator, cache, telemetry, persistent knob or second technical owner.

### Measurements and external comparisons

- Fork author's **two-patch combined** r71 results: gfx1100 Qwen3-30B-A3B 130.4→136.3 tok/s (+4.5%); gfx1201 133.1→138.8 (+4.3%). They do **not** isolate patch 0001, and do not prove a Qwen3.8/Flash-Next route.
- Fork maintainer r77→r78 gfx1201 **Gemma4 12B QAT / Q8_0 KV**: tg256 60.15→61.05 tok/s (+1.5%) with both patches, ROPE_FWHT disabled 60.41, FWHT_Q8_1 disabled 60.80; pp512 2975.8→2970.3 (flat). Reported greedy hashes, 450/450 ROPE and 2444/2444 MUL_MAT backend cases, full-vocab KLD and width probes passed externally. The kill-switch arms suggest contributions but provide no isolated confidence interval. All are **external** and below BigCherry's ≥3% E2E promotion bar; no first-party BigCherry gain is established.
- Upstream [#26767](https://github.com/ggml-org/llama.cpp/pull/26767) is the already-present generic RMS_NORM→MUL→ROPE baseline. [#29772](https://github.com/ggml-org/llama.cpp/pull/29772) concerns Vulkan *wide* FWHT, not this HIP candidate. [SGLang grouped QK norm+RoPE](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/minimax_qknorm_rope.py) is a head-layout comparison without ggml Hadamard/alias semantics; do not transplant or extrapolate its performance.

### Completed host discriminator and next terminal gate

Disposable host-only Python oracle: **842 assertions passed** for exact integer FWHT vs lane-decomposed warp32/64, N64/128/256, 256-thread row coverage, shifted/exact output alias and NORMAL/NEOX vs IMROPE/MROPE selector; Python compilation passed. This does not prove F32 contraction, GPU barriers, graph lifetimes, GPU numerics or performance. No HIP build, backend-op test or BigCherry hardware benchmark was run.

1. **Read-only first:** existing current-pin graph/profile for a representative Gemma4 route (only if actually deployed), count exact five-node patterns, widths, output consumers/aliases, physical rank, two native launches, and *non-overlapped exposed* E2E wall for decode n=1, MTP verify 2–8, pp512, multi-ubatch, repeated same-process requests, graph on/off and 8K/98K. No chain → NO_ELIGIBLE_FWHT_CHAIN; optimistic removable exposed wall <**2.9126%** E2E → BELOW_CEILING; pin already absorbed → PIN_ABSORBED. Terminalize without patch/queue.
2. **Only after gate 1 passes:** one default-off native-fusion-path specialization inside rope.cu and ggml-cuda.cu, reusing the exact fwht.cuh butterfly and existing graph/stream/allocator. Pseudocode: if native_norm_rope_eligible AND consecutive_reshape_hadamard AND one_use AND width_supported AND safe_full_window_output_alias then fused_one_CTA_per_head else native_norm_rope_then_fwht. No IMRoPE expansion or new generic fusion framework.
3. **Qualify:** host and HIP N64/128/256 with zero/subnormal/NaN/Inf, nonzero RoPE positions, partial rotary dims, tail heads, alias/nonalias, logits/KLD/ULP, deterministic greedy, MTP acceptance, KV slot bytes, graph replay/reallocation, same-process multi-request, multi-ubatch and complete physical work; gfx1100/gfx1201 ISA/LDS/VGPR/spill. A/A then ≥4 sessions per arch with ≥10 paired ABBA rounds; CI95-low ≥3% E2E and ≤1% unaffected control regression. Otherwise NO_GAIN, NUMERIC_FAIL, ALIAS_UNSAFE or REGRESSION.

**12-hour/novelty:** checked selected and other active BigCherry branch commits and PRs through 2026-10-19 17:11 UTC. No independent same-capability work or recorded queue in preceding 12h. RNX08/BCOP327 (Q8_0 SET_ROWS) and PRVP06/BCOP329 (Vulkan wide FWHT) are scheduled audits of distinct kernel paths. Protected RAD13/PR #330 (latest independent b0973b 17:08 UTC), RCS09/RCS05, RAD11/12/10, WSL and release work were not modified; off-repository queues cannot be verified.


## Description

Harvest independently gated PLE gather/dequant, Qwen-specific norm+RoPE, and transpose/short-conv layout wins without creating a monolithic or duplicate generic fusion.

## Steps

- Reject generic norm+RoPE re-port because 1004 and pinned llama.cpp already cover it; proceed only for a measured Qwen/Gemma-specific dataflow/layout specialization.
- Map PLE rows through Qwen4Exp GET_ROWS/dequant/loading and verify actual GGUF storage before proposing int6.
- Trace short-conv/attention layout round trips and prefer producer/consumer layout fusion over a standalone transpose.
- Give every surviving subfeature its own trace marker, disable path, and acceptance gate; split patch allocation if multiple unrelated features survive.
- Check row IDs before gather and test PLE host/VRAM placement.

## Detailed Solution & Technical Design

Perform tensor/callsite H2D attribution before implementing any PLE optimization. The relevant chain is src/models/qwen4exp.cpp::{llm_graph_input_qwen4exp_ple::set_input,build_inp_ple} -> host hash -> optional llama_prefetch_rows() in src/llama-impl.cpp -> ggml_backend_tensor_set(rows,...) -> ggml_get_rows(per_layer_tok_embd,rows). Instrument bytes, tensor identity, and caller, separating tiny PLE I32 row-index uploads, host-backed per-layer-token-embedding row movement, meta-backend placement/split copies, and unrelated H2D.

## Code Samples & Guidance



## Files

- kernels/r9k_ple.hip
- kernels/r9k_norm_rope.hip
- kernels/r9k_transpose.hip
- r9700_vllm/kernels/ple.py
- r9700_vllm/ple/int6.py
- r9700_vllm/ple/short_conv.py
- src/models/qwen4exp.cpp
- ggml/src/ggml-cuda/rope.cu
- ggml/src/ggml-cuda/rope.cuh

## Validation

Do not infer PLE ownership from aggregate 4.7 s H2D time. Produce a per-tensor/per-callsite attribution and only then compare direct mapped-host/device gather, unique-row packing, or device-side hash/index generation. Keep gfx103x fallback unchanged until gfx1100/gfx1201 evidence proves a portable route.

## Effort & Risk



## Standards

Original source alias is R9X08. Existing owner 1004 remains authoritative for generic RMS_NORM+MUL+ROPE.

## Acceptance Criteria

No PLE cache or gather implementation is added from aggregate traffic alone. The measured H2D attribution identifies the dominant tensor/callsite and distinguishes indices from row payloads before a code proposal is selected; unrelated norm/RoPE traffic is not folded into this item.

## Notes

Proposed slot 1309, split if needed. Current Flash-Next evidence flags PLE H2D traffic (~23/token) for targeted mapping.

Verbatim legacy source retained during R9X→RNX migration:

# R9X08 — PLE, norm+RoPE, and transpose microfusion portfolio

Status: planned
Proposed patch: `1309_r9x_decode_microfusions` (split if more than one subfeature survives)
Depends on: R9X01
External source: `kernels/r9k_ple.hip`, `kernels/r9k_norm_rope.hip`, `kernels/r9k_transpose.hip`, `r9700_vllm/kernels/ple.py`, `r9700_vllm/ple/int6.py`, `r9700_vllm/ple/short_conv.py`
Existing owner: `1004_rms_norm_mul_rope_fusion` plus Qwen4Exp model path



GPT design pass (req_83e7cdc000be4b9e): PLE row indices are host-generated and uploaded, proving some H2D but not proving that PLE dominates the observed traffic. This item is attribution-gated and remains analysis-only.

## Goal

Harvest small memory/launch wins that do not justify their own campaign: Qwen4Exp per-layer embedding (PLE) gather/dequant, model-specific norm+RoPE specialization, and transpose elimination/fusion around short conv/attention. Each subfeature has an independent go/no-go gate.

## Important non-duplication rule

`patches/1004_rms_norm_mul_rope_fusion/patch.py` targets `ggml/src/ggml-cuda/rope.cuh`, `ggml/src/ggml-cuda/rope.cu`, and `ggml/src/ggml-cuda/ggml-cuda.cu`, but is rejected because the generic RMS_NORM+MUL+ROPE fusion already existed in the pinned llama.cpp base. Therefore **do not** re-port r9k norm+RoPE generically. Only proceed if r9k contains a Qwen/Gemma-specific dataflow or output-layout specialization not present upstream and it is a measured hotspot.

## PLE mapping

r9k PLE is an int6 row gather+dequant, including a pinned-host/UVA use case. Map Qwen4Exp PLE/per-layer token embeddings in `src/models/qwen4exp.cpp` to current GGML GET_ROWS/dequant/backend path and model tensor loading. Determine actual GGUF storage format before proposing int6. If BigCherry's deployed model stores PLE in another quant type, port the fused gather+dequant concept to that representation rather than forcing checkpoint conversion.

Security/correctness note from source review: validate all row IDs before gather; do not reproduce a source path that assumes IDs are always in bounds.

## Transpose/short-conv mapping

Use `r9700_vllm/ple/short_conv.py` and `r9k_transpose.hip` only to identify avoidable layout round trips. In llama.cpp, trace the PLE/conv graph in `src/models/qwen4exp.cpp` and backend transpose/contiguous ops. Prefer changing producer/consumer layout or fusing a transpose into an existing kernel over adding a standalone faster transpose.

## Packaging

Do not create a monolithic 1309 if multiple unrelated subfeatures survive. Reserve 1309 for the first validated microfusion; allocate later free patch numbers for additional survivors and update this plan. Every subfeature must have its own trace marker and disable path during evaluation.

## RDNA adaptation

PLE gather/dequant and transpose reduction are strongly portable (`common`) with per-arch vector width/coalescing tuning. Norm+RoPE specialization is also primarily memory-bound, but any source ISA assumptions must be isolated. RDNA2 is a high-value test for these changes because they should not require gfx12 matrix hardware; a failure to port is a warning that the design is over-specialized.

## Change Log

- 2026-10-03T01:33:25.968209+00:00 (created-by): Created by codex
- 2026-10-03T01:39:08.818575+00:00 (updated-by): Updated: section:notes
- 2026-10-03T02:18:35.135369+00:00 (updated-by): Updated: section:detailed_solution, section:validation, section:acceptance_criteria, section:notes

## Reviews

- RV4207

## 2026-10-12 deep audit — PLE cold-row acquisition (one narrow RNX08 slice)

**Scope and evidence:** This section owns *only* Qwen4Exp per-layer n-gram embedding (PLE) row acquisition and its prefill/TTFT consequences. RNX08's norm+RoPE and transpose proposals remain separately gated, not approved for implementation here. Source-static review is against BigCherry's pinned llama.cpp **b11552 / 23b0202a**, not a new GPU measurement. The independent RNX08 plan last changed 2026-10-03 15:24 UTC; no matching independent PLE patch/PR/benchmark/queued lane appeared on inspected active branches in the 12 hours before this run. Recent BCOP136–144 concern other paths. Protected PA49/b11552 patch reconciliation, QFP41/1356, QFP48, QFP49/1360/1361, PGC17 and MTP/1346/1359 were not edited.

### Code-level correction: PLE H2D count is not 23 copies/token

- `src/models/qwen4exp.cpp::llm_graph_input_qwen4exp_ple::set_input` (~1224–1295) hashes n-gram tokens and their attention-KV predecessors on the CPU into `idx[n_heads*n_tokens]`, checks one sequence per token, conditionally calls `llama_prefetch_rows`, then issues **one** `ggml_backend_tensor_set(rows, idx, 0, idx.size()*sizeof(int32_t))` per input update. The historical "~23/token" is a row-index count if `ple_n_heads=23`, **not 23 H2D calls/token**. Its nominal index payload is `4*ple_n_heads*n_tokens` bytes (92 B/token at 23 heads). Actual hparams, bytes, transfers, and GPU assignment must be read from the deployed GGUF/trace; the old aggregate 4.7 s H2D figure does not attribute time to PLE.
- `build_inp_ple` (~1353–1371) creates the I32 input, calls `ggml_get_rows(model.per_layer_tok_embd, rows)`, then reshapes the gathered output. `ggml/src/ggml-cuda/getrows.cu::ggml_cuda_op_get_rows` (~442–459) dispatches the existing typed GPU gather/dequant, including `GGML_TYPE_IQ4_NL` (~395). Do not add an int6 PLE kernel without proving the actual GGUF type; do not create a duplicate gather or dispatcher.
- `src/llama-model.cpp::load_tensors` (~1848–1852) enables PLE prefetch only when the tensor is lazy. `src/llama-impl.cpp::llama_prefetch_rows` (~23–41) requires a host-backed tensor and validates every row index. `src/llama-mmap.cpp::llama_prefetch` (~837–914) page-aligns, sorts/merges ranges and advises `MADV_WILLNEED` on Linux or `PrefetchVirtualMemory` on Windows; this is a hint, **not** proof that cold page faults or serialized I/O disappeared. `src/llama-model-loader.cpp::lazy_read::add` (~1088–1114) uses lazy mode `on` or `auto` for tensors >4 GiB when mmap is supported. Existing prefetch already implements the upstream #29599 mechanism.
- No `--lazy-mode on-direct` or direct-row reader is present in b11552. Upstream [#28136](https://github.com/ggml-org/llama.cpp/pull/28136) closed unmerged, superseded by [#29030](https://github.com/ggml-org/llama.cpp/pull/29030), also closed unmerged after [#29599](https://github.com/ggml-org/llama.cpp/pull/29599) merged on **2026-09-30**. #29030's *distinct* experiment stages **packed** rows into a per-context input and performs `GET_ROWS` over identity indices, keeping GPU dequant; #28136's earlier design staged F32 rows on CPU. Neither should be blindly re-ported on top of #29599.
- Other-engine correctness reference: vLLM [#58489](https://github.com/vllm-project/vllm/pull/58489), merged 2026-09-25, fixed a PLE side-stream use-after-reuse of graph-pool row IDs (reported in [#58441](https://github.com/vllm-project/vllm/issues/58441)). BigCherry's current host-index synchronous input path is **not shown** to have that bug; any future async prefetch must pin ID/storage ownership until the last GPU/IO consumer, including graph replay, and prove repeated-request greedy/logit stability.

### Cheapest discriminator: row/page arithmetic, not a kernel port

A deterministic **synthetic host fixture** (not a BigCherry GGUF/SSD measurement) used the external #28136 example geometry: 320,001,536 rows, 90 packed bytes/row, 160 F32 elements/row, 4 KiB pages, 23 heads and 512 tokens (11,776 index slots). Uniform random rows touched 11,776 unique rows and 12,048 distinct pages: 49,348,608 B of potentially faulted pages versus 1,059,840 B useful packed rows (46.56x *theoretical page footprint*), with only 47,104 B I32 input. A 128-row hot set touched just **3 pages** for the same index count. An F32 CPU-stage alternative would upload 7,536,640 B per ubatch; a packed-row stage would upload up to 1,059,840 B plus identity indices, before dedup. These are traffic models, **not measured PCIe, NVMe or runtime bytes**. Six synthetic page-interval/overlap/boundary checks passed. Source-static checks of the exact b11552 files: 14/14. No C++/HIP build, repository pytest or GPU test ran.

### Decision and exact implementation/measurement gate

**Decision: retain upstream #29599 prefetch and native GET_ROWS; no 1309 patch, new cache, runtime knob, or hardware queue.** RNX08 PLE owns only the following bounded gate. Do not modify QFP17's active prefill definitions, QFP23's profile admission, or PA49's pin composition. RNX09 consumes a result only after admission.

1. **Admit the actual workload first.** Record GGUF PLE tensor name/type/`ne[0:2]`/row bytes/file offset, lazy mode, backing buffer, `can_prefetch`, number of heads, PLE input/backend owner and graph split on each XTX/R9700. Verify the 6900 XT drafter's PLE participation rather than assuming it. Distinguish model file `--load-mode dio` from the lazy mmap backing; if PLE is resident or not executed, terminate this slice.
2. **Use existing tooling, not a new telemetry system.** Extend the *disposable* `tools/lab/flash-next/page-cache-fraction.py` concept to sample **only the PLE file-offset range**, not the entire GGUF; capture before/after `mincore` residency, per-thread major/minor faults (`getrusage`), prefetch-call wall, PLE `GET_ROWS` kernel time, `ggml_backend_tensor_set` bytes/count, host read bytes/IO latency, GPU critical-rank time, and unique rows/pages per ubatch. Use real diverse code/prose prompts and a repeated-token hot-set control at 8K/24K/80K; cold is a fresh process with verified selective eviction, warm is same-process reevaluation with prompt cache disabled. Keep MTP off for attribution, then MTP-on as a no-regression control. Record all three GPU ranks; no P2P assumption.
3. **Fail-fast gate:** If PLE lazy prefetch is inactive, cold major faults are negligible, PLE page acquisition is <5% of prefill/TTFT critical wall, or the best possible end-to-end Amdahl saving is <3%, **close PLE acquisition without implementing direct reads**. Also close if #29599 already eliminates the observed cold/warm gap. Do not infer a PLE issue from aggregate H2D or from the 10K-to-80K prefill decay alone.
4. **Only if gate 3 passes**, make one *disposable, default-off* comparison of the existing #29599 prefetch against a minimal adaptation of #29030's packed-row per-context reader. Preserve original quant layout and existing `ggml_get_rows` dequant, row bounds, EOS/image/mixed-batch semantics, per-sequence predecessors, graph shape/reuse and fixed-rank tensor placement. On each request: hash IDs -> dedup/sort packed file rows -> bounded workers read into request-owned staging -> copy staged packed tensor plus identity indices -> existing `GET_ROWS` -> release staging only after input-copy completion. No side-stream reading graph-pool IDs; no new persistent placement/cache/allocator. A failed read/short read must fail visibly, never silently skip work. Compare isolated mmap-prefetch, cold direct-read and warm/resident controls with identical model/pin/flags; don't use NVIDIA gains as AMD expectations.
5. **Promotion/rejection:** Exact gathered F32 rows versus baseline for actual quant types, full-vocabulary logits/KLD, greedy IDs, MTP acceptance, EOS/multimodal/mixed ubatches, repeated same-process requests, graph capture/replay, file truncation/short-read, and no missing rows. Verify expected unique row reads, input bytes, major faults, PCIe plausibility and memory lifetimes. Only then perform >=4 independent sessions per relevant gfx1100/gfx1201 configuration, >=10 paired ABBA rounds/session at cold and warm 8K/24K/80K. Consider a *user-opt-in* path only if cold TTFT CI95-low >=3% and warm/decode regression CI95-high <=1%; otherwise reject and retain #29599. No production-default change without the same gate and proof of real cold workload frequency.

**Measured BigCherry context:** Historical Flash-Next prefill ~1,100 t/s at ~10K and ~910 t/s at ~80K (QFP17) is **not** a controlled cold/warm PLE comparison. External #29599 Strix Halo ROCm/Vulkan pp512 177.19±6.99 -> 382.93±32.07 t/s is *upstream*, not BigCherry. External #28136 direct-read cold gains and warm penalties vary with cache pressure and architecture; no transfer of those percentages is justified.

**Consolidation:** RNX08 owns the PLE admission/decision; upstream #29599 owns existing page prefetch; `ggml_get_rows` owns dequant; QFP17 owns prefill profiling, RNX09 owns integration, 1004/merged #26767 owns generic norm+RoPE. RNX08's legacy proposed `1309` slot is not reserved. The shared RNX README received recent independent RNX04 work and was deliberately left unchanged; reconcile its PLE row only when that documentation is no longer protected. Do not create a second PLE cache, host hash, or gather dispatch.


## 2026-10-19 kernel audit — Q8_0 SET_ROWS warp-group quantization (BCOP327)

**Disposition: HOLD, source/route gate only.** This is a narrow HIP GPU cache-write kernel slice, not RNX08's PLE or norm+RoPE work. Do not port or queue a GPU run from external microseconds alone: the external end-to-end results do not meet BigCherry's >=3% promotion threshold. Reopen only after a real BigCherry physical-rank Q8_0 write trace proves >=2.9126% optimistically removable *exposed* E2E latency. No patch, dispatcher, allocator, runtime knob or experiment was created here.

### Exact pinned implementation and source-level finding

- BigCherry llama.cpp b11552 (`23b0202a189c44a54625aadcb37a946dd1d6278d`): `ggml/src/ggml-cuda/set-rows.cu::ggml_cuda_op_set_rows` -> `set_rows_cuda<float,int32_t/int64_t>` -> `set_rows_cuda_quant<idx_t,block_q8_0,QK8_0,quantize_f32_q8_0_block>` -> `k_set_rows_quant`. The Q8_0 destination is a dedicated branch; other quants and F16/BF16 writes remain untouched. The kernel maps one HIP thread to one 32-element Q8 block; `cpy-utils.cuh::quantize_f32_q8_0_block` serially does 32 F32 reads/max comparisons, d=amax/127, id=d?1/d:0, 32 roundf byte stores and one half-scale store. It maps slot indices via `fast_div_modulo` and calls `ggml_cuda_pdl_sync` before reading indices. `set_rows_cuda_quant` asserts `ne00%32==0` and launches 256 threads per CTA, `ceil(B/256)` CTAs, where `B=ne00*ne01*ne02*ne03/32` is the full logical Q8 block count.
- [RDNA fork #171](https://github.com/stew675/llama-cpp-rdna-boosts/pull/171) replaces only the Q8_0 arm with `k_set_rows_q8_0_group`: one 32-lane group per quant block, five XOR-shuffle maximum stages, identical scale/inverse/roundf/bytes, eight groups per 256-thread CTA, `ceil(B/8)` CTAs. **At large B the CTA grid is 32x larger**, not 32x less logical work. Per-lane source and destination accesses become coalesced; index loads are replicated across 32 lanes unless compiler/cache broadcasts them. Five shuffle stages replace serial max; no extra LDS or inter-group barrier is required. Real VGPR/spills, wave32/64 lowering, CTA residency, cache behavior and occupancy remain unmeasured on BigCherry.
- Example D256/four KV heads: tokens 1/2/4/8 -> B=32/64/128/256; old CTAs=1/1/1/1, candidate CTAs=4/8/16/32. Both logically read `128*B` F32 bytes and write `34*B` Q8_0 bytes plus slot-index traffic; **these are not physical HBM or PCIe bytes**. On dual XTX + R9700 without ordinary P2P, measure only the rank actually executing each write; gfx1030 is fallback until qualified. Never infer full K/V writes were eliminated.

### Measured evidence, novelty and ownership

- Fork #171 (16–17 October): gfx1100 WSL2 Q8_0 SET_ROWS 5.1 -> 3.7 us/call across ~44,500 calls. Its 400-token request GPU time 4,111 -> 4,048 ms includes **two** patches (also `rms_norm_q8_1`); the isolated SET_ROWS per-call saving times calls is 62.3 ms, only **1.52% of that GPU-time sum** even if entirely exposed. This is not a BigCherry E2E wall-clock bound, and the per-call kernel values came from one trace per arm. Combined decode +1.7% was within noise. gfx1201 native Linux Q8_0 SET_ROWS 1.51 -> 1.33 us; combined MoE decode +0.6%, dense flat. No BigCherry gain was measured. The fork's base is newer than b11552, so recheck pin absorption before porting.
- [llama.cpp #26731](https://github.com/ggml-org/llama.cpp/pull/26731) repairs underfilled quantized CPY launches, not Q8 SET_ROWS. No equivalent merged upstream Q8 SET_ROWS group writer was verified. vLLM's [ROCm paged cache writer](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/rocm_attn.py) uses vectorized slot-aware writes but FP8/F16 cache layouts, not GGUF Q8_0: comparison only, no direct port.
- RNX08 owns **only this Q8_0 cache-write kernel disposition**. PRBE15 owns ROPE/IMROPE->VIEW->SET_ROWS scatter fusion, RNX02 owns quantized KV *attention reads*, 1307 owns Q8_1 activation caching. Fork #171's independent `norm.cu::rms_norm_q8_1_f32` register-reuse change is out of scope. Reuse native `set_rows.cu` launch/slot/stream ownership; do not introduce another dispatcher or cache.
- Repository-visible 12-hour check (2026-10-19 01:16–13:16 UTC) found no independent Q8_0 SET_ROWS kernel patch/PR/plan/benchmark/queue. Protected RAD13 FP8 worker PR #330, RCS09 runtime-shape work, RAD11/RAD12/RAD10 helper lanes, WSL and release changes were excluded and untouched. BCOP322–326 cover different kernels. Off-repository queues could not be verified.

### Cheapest discriminator and implementation-ready terminal gates

Host-only disposable oracle (outside repository) passed **10,005 finite Q8_0 block byte comparisons** between serial and XOR-reduced reference, **13,260 mapped block-address cases** across tokens/heads/dimensions, and Python compilation. It models F32 arithmetic, F16 scale storage, round-away-from-zero and permuted slots; it does not prove GPU NaN/Inf behavior, FP contraction, memory coalescing or graph replay. No HIP build, backend test or GPU benchmark ran.

1. **Admission without a hardware job:** inspect the production cgraph and per-rank profile for actual F32 -> Q8_0 `GGML_OP_SET_ROWS` K/V writes, I32/I64 slot indices, `ne00%32==0`, actual B, launch counts and exposed critical-rank time on plain decode and MTP verify. If no route, close `NO_Q8_WRITE_ROUTE`; if optimistic exposed E2E ceiling <2.9126%, close `BELOW_CEILING`; if upstream pin already absorbed the mechanism, close `PIN_ABSORBED`.
2. **If admitted**, implement only a default-off Q8_0 branch inside the existing `set_rows_cuda_quant` selection: eligible F32/Q8_0, valid index/stride/row bounds, supported gfx1100/gfx1201 -> group kernel with `ceil(B/8)` CTAs; otherwise native `k_set_rows_quant`. Preserve PDL sync, slot indices, stream, destination allocation, exact `B` work and all other types. No persistent buffers, new config surface or host sync. Capture compiler ISA, VGPR/LDS/spills, wave size, occupancy, grid and actual traffic.
3. **Correctness first:** `test-backend-ops -o SET_ROWS` for Q8_0 and I32/I64, D64/128/256/512, 1/2/4/8/16 tokens, permuted/duplicate/edge slots, noncontiguous outer strides, zero/halfway/saturation/NaN/Inf, canaries and exact destination bytes. Require logits/KLD/greedy/MTP acceptance, same-process repeated requests, multi-ubatch, long context, graph capture/replay, reset and physical work/write accounting. Reject any apparent gain from skipped work or changed quant bytes.
4. **Performance:** same pin/model/flags A/A then ABBA; pp512/2048, tg128, MTP1/3/5, 8K/32K/98K, each XTX, R9700 and mixed no-P2P controls. Promotion only with CI95-low >=3% E2E gain, no >1% material control regression and all correctness/resource gates. Otherwise close `NO_GAIN`, `REGRESSION` or `NUMERIC_FAIL`.
## 2026-10-19 kernel audit: upstream F32 RMS_NORM float4/register reuse (BCOP315)
**Disposition: WAIT_PIN / NO_LOCAL_KERNEL_PORT.** This is the dormant **norm** subfeature of RNX08, not a repeat of RNX08's 2026-10-12 PLE page-acquisition audit. Last independent RNX08 plan commit found: `a50af2b6`, 2026-10-11 19:15:57 UTC; no matching independent norm/RNX08 patch, benchmark, PR or recorded queue in the 12 hours before this audit. Active RAD12 helper lane classes, RAD10 streaming/counters, RAD06 helper kernels, RHP07 placement, RCS/MEN releases and workspace work were excluded. This kernel is not an authorized replacement for the existing 1004/merged generic norm+RoPE fusion.
### Implementation and graph route, pinned b11552 vs merged upstream
- **Pinned baseline** `ggml/src/ggml-cuda/norm.cu::rms_norm_f32<block_size,...>` (llama.cpp `23b0202a`, ~85-180): one CTA per logical row, F32 sum-of-squares, `block_reduce<SUM>`, `rsqrtf`, then **a second global read of x** to write normalized output. Dispatch uses 256 threads for `ncols<1024`, otherwise 1024; optional RMS_NORM+MUL, +ADD and SCALE fusion reuse the same kernel. No per-row cross-CTA state or new scratch allocation.
- **Merged upstream** [llama.cpp #29720](https://github.com/ggml-org/llama.cpp/pull/29720), commit [`623c1639`](https://github.com/ggml-org/llama.cpp/commit/623c16393c7d53a37df8bb69c6ea8ed6e5171f62) on 2026-10-16: `norm.cu::rms_norm_f32<block_size,vpt,do_vec,...>` loads aligned `float4 xs[vpt]`, keeps x in per-thread registers across reduction and output, and vector-loads full-width MUL/ADD operands. It preserves one reduction and the trailing `__syncthreads()` for shared reduction storage reuse; **it removes a possible x reread, not a launch, synchronization barrier, or any required output work**. Vector admission in `rms_norm_f32_dispatch` requires `ncols>0 && ncols%4==0`, 16-byte output base, `ggml_cuda_is_aligned(x_t,16)` (base and `nb[1..3]`), and full-width/aligned MUL and ADD if present. Broadcast, misalignment and `ncols>16384` use scalar fallback. Vector CTA/vpt: `ncols/4<=32:32/1`, `<=128:128/1`, `<=256:256/1`, `<=512:512/1`, `<=1024:1024/1`, `<=4096:1024/4`. On gfx1100/gfx1201 this changes active wave32 counts, VGPR live ranges and possible spills; actual occupancy/ISA was **not measured**.
- **Qwen4Exp call graph:** `src/models/qwen4exp.cpp::build_layer_attn` (~950-990) builds interleaved Q/gate views, Q and K `build_norm(...LLM_NORM_RMS)`, then `ggml_rope_multi` IMRoPE. `ggml-cuda.cu::ggml_cuda_should_fuse_rms_norm_mul_rope` (~2785) admits only NORMAL/NEOX, so do **not** assume Q/K IMRoPE executes that native 3-op fused kernel. `ggml_cuda_try_fuse` (~4344-4365) can instead fuse RMS_NORM+MUL (or +ADD/SCALE) into `norm.cu` when graph edges, F32 types, row contiguity and memory-range guards pass. An interleaved Q view and broadcast gamma do **not** by themselves prove vector admission: verify actual `ne/nb`, aligned weight/output pointers, fused op and physical backend per layer/rank. The attention output gate is a **separate** post-attention sigmoid/MUL; vLLM's QKV/gate fused kernel is not a source-compatible transplant.
- **Ownership/lifetime:** upstream `norm.cu` and native `ggml_cuda_try_fuse` own the kernel/dispatch and fusion, with graph/output buffers unchanged. No new `1309` package, RMS_NORM selector, scheduler, allocator, cache, telemetry or runtime flag. RNX08 owns only the qualification/disposition of this norm subfeature; 1004 stays rejected/superseded for generic norm+RoPE; PRBE15 owns the distinct IMRoPE→VIEW→SET_ROWS question; QFP35/1355 owns HC gate POST; RNX10 owns shared-expert MUL+ADD; RNX09 consumes only a validated combined-stack result. The PLE portion above remains unchanged.
### Work, correctness and cheapest discriminator
- **Theoretical bound, not measured traffic:** if eligible, baseline's second `x` load is `4*ncols` bytes per F32 row (e.g. 1 KiB at D256, 8 KiB at D2048). The upstream variant still reads x once, performs all squares/reduction, reads any MUL/ADD, and writes all `ncols` outputs. A cached baseline reread may cause little HBM traffic; do not equate source-level bytes with PCIe/HBM savings. The kernel uses the same number of launches and required rows. Larger `float4 xs[4]` can increase VGPR pressure/spills; fewer waves can also reduce latency hiding. Multi-rank critical-path impact is unknown.
- **Numerical boundary:** vector per-thread accumulation groups F32 squares differently from scalar stride-`block_size` and can change reduction rounding and downstream MTP acceptance. This is not a bit-identity claim. Preserve all Q/K/gate outputs, padding, IMRoPE positions, and attention KV writes; verify full-vocabulary logits/KLD/greedy, MTP acceptance, repeated same-process requests, ubatch changes, graph replay and long context.
- **Host-only completed:** a source-shaped selector/coverage fixture modeled `ncols`, alignment, full-width MUL/ADD, CTA and vpt; **4,734/4,734 cases**, 44,700,957 logical elements, 4,478 fallback controls, zero omitted/duplicated output elements; Python compilation passed. A simplified F32 per-thread accumulation surrogate produced **5/168 grouping-difference witnesses**; it does **not** emulate HIP block-reduce/ISA or prove a GPU mismatch. No BigCherry repository pytest, HIP compile, rocprof, hardware correctness or benchmark ran.
- **External measured control (not BigCherry):** #29720's RX 7900 XTX Qwen3.5-27B Q5_K_M pp128/ub1 32.67→32.68 t/s (+0.03%), pp1024/ub512 925.33→936.21 (+1.18%); R9700 pp128/ub1 27.08→27.23 (+0.55%), pp1024/ub512 937.73→949.99 (+1.31%). These are external benchmark tables without BigCherry confidence intervals; CDNA/MI100's larger gains must not be extrapolated to RDNA. No observed BigCherry norm-only bottleneck is established.
### Exact decision/benchmark gates
1. **Gate 0 — source/route census before GPU campaign:** keep production pin b11552 unchanged. On a future approved pin including #29720, use existing `tools/bigcherry/profiling/rocprof.py` and `tools/lab/flash-next/prefill-kernel-table.py` plus a disposable graph/dispatch receipt to identify per physical gfx1100/gfx1201 rank: model/quant, layer, op, `ncols/nrows/ne/nb`, F32 type, aligned pointers, full-width gamma, native RMS_NORM+MUL(+ADD/SCALE) fusion, vector vs scalar selector, kernel calls/time, exposed critical-rank wall and IMRoPE route. Do not infer activation from model architecture or upstream throughput. If Q/K norms are absorbed by a different fusion, never count their x rereads again.
2. **Gate 1 — opportunity rejection:** if no eligible executed F32 RMS_NORM or if the *optimistically removable* exposed E2E time is <2.9126% (the minimum for +3% throughput even at zero replacement cost), close `NO_ROUTE`/`BELOW_CEILING` without a kernel fork. At a plausible isolated 20% kernel speedup, the original kernel would need about **17.48%** of serialized E2E time to support +3% throughput. Treat these as ceilings, not forecasts.
3. **Gate 2 — only after Gate 1:** compare same approved-pin upstream vector selector vs a local **test-only scalar dispatch control**, not a second production implementation. Run gfx1100 and gfx1201 separately; F32 D256/2048/3584/4096/5120/8192, odd/misaligned and broadcast negative controls; pp512/2048 and tg128, ub1/8/128/512, 8K/80K and long-context/MTP3. Verify x read/write and row counts, exact output coverage, `rocprofv3` VGPR/SGPR/LDS/scratch, wave32/CTA occupancy, spills, barrier count, clocks, graph recapture and per-rank critical path. Compare deterministic greedy/logits/KLD and multi-request/ubatch/graph; test unaffected gfx1030 fallback without making it a primary tuning target. No P2P/RCCL inference from single-rank kernels.
4. **Terminal:** require >=4 independent sessions/architecture, >=10 paired ABBA rounds, CI95-low >=3% E2E for an admitted workload, <=1% regression in unaffected decode/prefill/MTP controls, complete physical work and no correctness failure. Otherwise `UPSTREAM_ABSORBED` (no patch needed), `NO_GAIN`, `CORRECTNESS_FAIL` or `BELOW_CEILING`. The only allowed production change is an approved upstream pin absorption with normal pin/rebase ownership, not a local norm kernel fork.
**Other-engine mechanism:** [vLLM #51406](https://github.com/vllm-project/vllm/pull/51406), open/updated 2026-10-19, enables an existing Triton `fused_qk_rmsnorm_rope_gate` on ROCm by changing `is_cuda()` to `is_cuda_alike()`. Its MI325X BF16 measurements show +10.3% eager decode but only +2.1% graph-on batch1 and ~parity graph-on prefill; neither is gfx1100/gfx1201 evidence. It fuses a **different** Qwen3Next QKV split/GemmaRMSNorm/NeoX RoPE/gate-copy contract; BigCherry's interleaved Q view, IMRoPE, ggml graph fusions and F32 norm path require separate semantic/shape proof. Reject direct transplant and avoid double-counting with #29720.
**Sources:** [pinned norm.cu](https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278d/ggml/src/ggml-cuda/norm.cu), [merged norm.cu](https://github.com/ggml-org/llama.cpp/blob/623c16393c7d53a37df8bb69c6ea8ed6e5171f62/ggml/src/ggml-cuda/norm.cu), [pinned ggml-cuda.cu](https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278d/ggml/src/ggml-cuda/ggml-cuda.cu), [pinned Qwen4Exp graph](https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278d/src/models/qwen4exp.cpp), [#29720](https://github.com/ggml-org/llama.cpp/pull/29720), [vLLM #51406](https://github.com/vllm-project/vllm/pull/51406).
