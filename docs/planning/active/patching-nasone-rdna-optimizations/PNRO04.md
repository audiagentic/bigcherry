---
id: PNRO04
order: 0
plan: patching-nasone-rdna-optimizations
state: done
created-at: '2026-09-09T10:52:17.766702+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: L
priority: P0
---

# gfx1100/gfx1201 BF16-WMMA chunked GatedDeltaNet — validated historical owner

## Disposition (2026-10-10, b11474 source rebaseline)

**Original implementation complete; patch 1253 is validated and in the production `validated-enhancements` set.** This supersedes the obsolete 2026-09 predicate-only TODO at the top of the previous plan. `patch.toml` declares `state="validated"`, `requires=[]`, and **`conflicts=["1221_rd50_gdn_chunked_recurrence"]`**. The package README previously said `untested` and *requires 1221*; those statements were stale and are corrected in the same audit. Do not create another GDN kernel or combine 1253 with conflicting 1221.

**Measured fact (historical, not current-pin):** eight bound `evidence/validation.json` records at llama.cpp **b11126** include four sessions on each of gfx1100 and gfx1201. The final record for each architecture has `eligible_for_validated_state=true`, `final_eligibility=true`, and `validation_disposition="validated"`; earlier one-/two-/three-session receipts explicitly say `incomplete`. `config/recipes.toml` records gfx1100 pp512 effects **+8.2/+8.8/+7.7/+9.3%**, gfx1201 **+10.4/+11.8/+10.6/+11.0%**, tg128 controls approximately flat, and GATED_DELTA_NET backend-reference passing both arms. These are per-session results for a hybrid-GDN workload, not b11474, long-context, MTP, tensor-split or general-model gains. No new GPU measurement was performed in this audit.

## Source-level implementation and ownership

- `engines/llamacpp/patches/1253_nro04_gfx1100_bf16_chunked_gdn/patch.py` creates `ggml/src/ggml-cuda/gated_delta_net_chunked.cuh`, `gated_delta_net_chunked_bf16_gfx11.cu`, and `gated_delta_net_chunked_bf16.cu`; it edits `gated_delta_net.cu` dispatch, CUDA build registration and backend test cases. This is a real implementation, not a predicate.
- `ggml_cuda_op_gated_delta_net_impl` selects chunked BF16 for **K=1, non-KDA, n_tokens>1, S_v=128**, a runtime RDNA3/RDNA4 device, and `GGML_CUDA_GDN_CHUNKED!=0` plus `GGML_CUDA_GDN_CHUNKED_BF16!=0`. Both flags are **default-on** when unset. gfx1030 remains sequential; RDNA3.5 is admitted by the broad RDNA3 predicate but has **no first-party qualification**. The `hardware=["gfx1100"]` metadata is narrower than the runtime gate and the validated gfx1201 receipt; do not infer qualification for other RDNA variants.
- gfx11 and gfx12 are separate WMMA translation units. `ggml_cuda_op_gated_delta_net_chunked_bf16_gfx11` / `..._bf16` allocate pool-owned `A_sc`, launch `launch_gdn_bf16_kkt` then `launch_gdn_bf16_scan` on `ctx.stream()`. The first pass constructs chunk-local inverse/decay scratch; the second reads scratch, produces attention output and publishes either the K=1 state tail or the fused-cache `state_d_ext` slot. `A_sc` is a stream-ordered temporary, not a persistent cache. The two launches must retain dependency order; graph capture/replay and pool reuse need current-pin controls.
- The temporary has **`ceil(n_tokens/64)*H*n_seqs*64*64*sizeof(uint16_t)` bytes**. Source-derived examples for H=32,n_seqs=1: 512 tokens **2 MiB**, 1024 **4 MiB**, 2048 **8 MiB**; an unchunked 245760-token call would request **960 MiB**. These are logical allocation sizes, not observed peaks. Qualify the actual ubatch rather than extrapolating long-context total tokens.
- `ggml_cuda_kernel_launch_try` checks synchronous `cudaGetLastError` after each launch. A rejected first KKT launch has no destination publication; if scan launch is rejected, only KKT scratch was enqueued. The fallback is stock sequential. **An asynchronous execution fault is not caught**, and launch acceptance alone does not establish successful output. The `BIGCHERRY_PATCH_HIT` log is once-per-process and proves selector entry, not per-request successful execution.

## Newly identified source-level admission/precision boundaries

**Unsupported GQA ratios can abort rather than fall back.** Both BF16 wrappers assert `H % H_k == 0` and instantiate KKT only for `H/H_k in {1,2,3,4,6,8}`; their default switch arm is `GGML_ABORT`. The outer dispatcher does **not** check this set before selecting BF16. A synthetically eligible S_v=128, K=1, RDNA3/4, H=80,H_k=16 (ratio 5) or H=256,H_k=16 (ratio 16) reaches the abort arm. This is a **source-proven admission gap**, not a reproduced production crash. Do not run unsupported GQA models through this route. A future expansion must check divisibility and the explicit supported ratio set **before** allocation/launch, then select the existing sequential fallback; no new dispatcher is needed.

**BF16 conversion is not strict ties-to-even.** Both gfx11/gfx12 helper families use `(float_bits + 0x8000u) >> 16` in `gdn_f2bf` and `gdn_f2bf2`. For positive normal F32 exact halfway cases, this rounds up even when the retained BF16 LSB is even. A disposable integer-bit host fixture tested **32,512** such halfway patterns and found **16,256** differing BF16 outputs versus round-to-nearest-even; e.g. `0x3f808000` produces `0x3f81` instead of `0x3f80`. This proves conversion-policy difference only; no observed logits/KLD error or GPU regression follows. Preserve the historical near-lossless tolerance, but do not claim native BF16 RNE or bit identity. A future precision change must first prove actual input incidence and end-to-end quality, not silently alter a validated kernel.

## Consolidation / adjacent owners

1253 alone owns **K=1 BF16 chunked GDN prefill**. `1221_rd50_gdn_chunked_recurrence` owns the alternative FP32 chunked implementation and **conflicts** with 1253. `PNRO05/1254` owns K>1 MTP prefix/tail snapshot publication and remains `untested`; its eligibility, per-snapshot correctness, accepted-token state, and rollback cannot inherit 1253's K=1 receipt. Do not edit the separately active Flash-Next/MTP/router/DFlash or Radiance capability paths to implement this item. Stock sequential GDN remains the fallback/control; there is no reason to create another allocator, selector table, cache, scheduler, or profiling framework.

## Current upstream / fork comparison (checked 2026-10-10)

- [llama.cpp PR #29353](https://github.com/ggml-org/llama.cpp/pull/29353), **open** (updated 2026-10-09), adds a single-CTA HIP/NVIDIA GDN MMA kernel using **high + residual BF16** operands to approximate F32. Actual HIP selector requires `GGML_CUDA_CC_IS_RDNA3_5`, `n_tokens>=2048`, H_k=16 and supported H. **It does not select gfx1100 or gfx1201**, so is a numerical/architecture design reference, not a replacement for 1253 or transferable performance claim. Do not port its extra BF16 residual scratch without a separate critical-path/quality gate.
- [llama.cpp PR #30087](https://github.com/ggml-org/llama.cpp/pull/30087), merged **2026-10-08**, changes the stock recurrent GDN warp-column geometry; pinned b11474 predates this change. Use an explicitly pinned stock baseline when comparing sequential controls. [#26001](https://github.com/ggml-org/llama.cpp/pull/26001) closed unmerged; [#30207](https://github.com/ggml-org/llama.cpp/pull/30207) remains draft Vulkan work requiring newer RADV maintenance1 support.
- [vLLM #56307](https://github.com/vllm-project/vllm/pull/56307) proposes a single-block FP32-state GDN for **gfx1150–1153**; [SGLang #39873](https://github.com/sgl-project/sglang/pull/39873) consumes an AITER fused GDN chain on **gfx950**. Their register/LDS residency and explicit fallback gates are useful design controls, but neither is qualified on the BigCherry dual-XTX + R9700 no-P2P topology. No external throughput figure is a BigCherry measurement.

## Bounded next gate / terminal outcomes

1. **No new kernel now.** First run a host-only selector fixture for S_v={64,128}, K={1,2,5}, n_tokens={1,64,65,512}, H/H_k={1,2,3,4,5,6,8,16}, cc={gfx1030,gfx1100,gfx1201,gfx1151}, with default/0 environment settings. Check **fallback not abort** for unsupported ratios before any future wider deployment; do not reinterpret historical production receipts as proof of this case.
2. Recompose 1253 on a pristine **b11474** pin, verify the exact dispatcher and two created translation units, and run package mechanics/compile checks. If an upstream change invalidates anchors, reconcile in 1253's owner path; do not introduce a second patch.
3. Only if the currently deployed workload actually exercises the route, compare b11474 **1253-on vs `GGML_CUDA_GDN_CHUNKED_BF16=0`**, same build, on isolated gfx1100 and gfx1201. Collect per-request route hits, KKT/scan launches, pool allocation and high-water, graph nodes/replay, output/state publication, and time attribution. Use pp512/1024/2048 with ubatch and context explicitly pinned; tg128, K>1/MTP, non-GDN dense, gfx1030 and unsupported ratios are negative controls. No parallel experiments overlapping protected active work.
4. Correctness: backend-reference GDN including odd 64-token tails, GQA ratios, fused-cache state, multi-request same-process, repeated graph replay and long-context greedy/full-vocabulary/KLD comparisons. Test exact-halfway BF16 bit patterns only as a conversion-policy fixture; retain the documented near-lossless numerical contract. Do not treat fewer launched kernels or missing state writes as a speedup.
5. **Terminal:** if stock/current-pin already supplies equivalent qualified performance or the converter/selector is not on the non-overlapped critical path, retain 1253 as historical validated with no further optimisation. Any *new* kernel/precision/dispatch change must be default-off, pass all safety/quality gates, and show >=3% CI95-low end-to-end gain with <=1% controls over >=4 independent sessions/architecture and >=10 paired rounds/session. No b11474 benchmark or new GPU correctness run is claimed here.

## Historical notes

The original 2026-09-24 predicate-only plan was overtaken by independent 2026-09-25 implementation `d423644d` and the 2026-09-26 four-session promotion. Earlier comments saying `no BF16 kernel exists`, `1253 requires 1221`, or `1253 untested` are **superseded**; the original commit history and immutable evidence remain authoritative for their respective dates. This audit updates documentation only, not patch state, recipe, implementation, experiment queues or evidence.

## Change Log

- 2026-09-09T10:52:17.766702+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:08:46.573893+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events


- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.067141+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:42.708292+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T02:26:15.401708+00:00 (updated-by): Updated: section:description, section:steps, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes
- chg_20260910_022800_five-nasone-successor-plans-no_4030
- 2026-09-10T02:28:00.297184+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:26:11.680244+00:00 (updated-by): Updated: section:validation, section:notes
- 2026-09-24T04:48:19.490128+00:00 (updated-by): Updated: section:description, section:steps, section:notes
- 2026-09-25T04:23:36.714622+00:00 (updated-by): Updated: section:notes
- 2026-09-25T04:23:39.602014+00:00 (state-transition): State: pending → in_progress
- chg_20260926_012534_qwen3536-hybrid-model-promp_1365
- 2026-09-26T01:25:37.283467+00:00 (updated-by): Updated: section:ledger-events
