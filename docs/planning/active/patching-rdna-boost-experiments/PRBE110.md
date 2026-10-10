---
id: PRBE110
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-23T09:19:09.142442+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# RD07 Q6_K MMQ scale fold: incremental gfx1201 qualification (1267)

## Authoritative disposition (2026-10-10)

**Pending, default-off. No new kernel or campaign is authorised yet.** The RD07-only patch `1267_rd07_q6k_mmq_scale_fold` already exists and is `untested`. Its b11126 gfx1201 four-session pp512 result (+2.4675%, CI95 [2.2776%, 2.6508%]; tg128 -0.0331%, CI95 [-0.0592%, -0.0071%]) is first-party, but the stored baseline composition **did not contain 1006**. Validated `1006_rdna4_mmq_q6k_codegen_fix` is now in `config/recipes.toml`'s `validated-enhancements` production set and independently measured about +18% Q6_K pp512 on gfx1201. **Do not add these effects or treat the old 1267 result as incremental to production.** gfx1100/gfx1030 1267 results failed the gain gate. Rejected 1203 is not a fallback; rejected/retired 1266 RD05 is not part of this work.

The 2026-10-09 merged upstream MMQ fixes #29953 and #30168 change the correctness baseline and dispatch/padding assumptions after pinned llama.cpp **b11474** (`b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea`). Do not time/promote 1267 on an unverified input-allocation boundary.

## Exact implementation and dispatch

- `ggml/src/ggml-cuda/mmq.cu::ggml_cuda_mul_mat_q`: quantizes/stages Q8_1 activations, allocates staging, builds `mmq_args`, and dispatches `ggml_cuda_mul_mat_q_switch_type`. `GGML_TYPE_Q6_K` calls `mul_mat_q_case` through the existing `0300_mmq_forced_j` composition.
- `ggml/src/ggml-cuda/mmq.cuh::mul_mat_q_switch_J`: selects tile J and config by architecture, shape, fallback, and shared-memory budget. `ggml_cuda_mmq_get_config` uses the existing RDNA3/RDNA4/RDNA2 tables. Preserve this dispatch; no new selector/table.
- `ggml/src/ggml-cuda/mmq.cuh`: Q6_K chooses `ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma` on AMD MFMA/WMMA, but DP4A on unsupported architectures. 1267 changes **only the MMA** implementation in `ggml/src/ggml-cuda/mmq-vec-dot.cuh`: row base scale is loaded before k01; each k01 subscale is multiplied by that base before the j0 loop; the inner accumulation uses `(float)C.x[l] * x_s2_reg[n][l] * dB`. It does not change DP4A, quantization, tensor allocation, scheduling, graph lifetime, or transport.
- The patch's once-per-process `BIGCHERRY_PATCH_HIT` marker is in the **host Q6_K dispatch**. It proves a dispatch call, not that the modified MMA path ran, that every request used it, or that fallback was absent. gfx1030 DP4A is an explicit negative route. Verify actual compiled kernel and shape with existing HIP profiling/trace infrastructure, not another telemetry framework.
- `1267/patch.py` has exactly three ordered `mmq-vec-dot.cuh` edits, an `mmq.cu` marker, a `mmq.cuh` J_MAX qualification knob, and backend-op perf cases. `1267/patch.toml` requires 0300 and conflicts with rejected 1203. The 1006 explicit F32 cast and 1267 scale-fold target the **same Q6_K MMA accumulation line**; 1267's dual anchor accepts both forms, but composability is not numerical/performance qualification. Do not change the validated 1006 implementation.

## Correctness-first blockers and cheapest discriminators

**B1: source-proven capacity/selection mismatch on the pinned base.** b11474 `ggml_cuda_mmq_get_J_max` rounds `ne11` down to a multiple of eight when reserving Q8_1 staging padding, whereas `mul_mat_q_switch_J` may choose a larger J to cover the same columns. Pinned RDNA4 Q6_K configs include J=16/32/64/128 (fallback=true) and J=16/32/48/64/80/96/112/128 (fallback=false). A host selector fixture with eligible shared-memory configurations gives `ne11=17`: allocation J_max=16, selected J=32, for both fallback values. This is a source-derived underpadding witness, **not a reproduced GPU OOB**. Upstream [#29953](https://github.com/ggml-org/llama.cpp/pull/29953) (merged 2026-10-08) moves tile selection before allocation and derives padding from the selected J/thread geometry; [#30168](https://github.com/ggml-org/llama.cpp/pull/30168) (merged 2026-10-09) threads src1 precision through host config helpers. Confirm the exact current-pin fix/backport status with the pin/upstream-fix owner and prove allocation >= worst actual read for J/shape/fallback/precision boundaries before any 1267 benchmark. Do not modify this generic MMQ correctness path as part of 1267's optimisation slice.

**B2: arithmetic and route parity.** The source 1006 accumulation is `((float)C.x[l]) * sc[k01/4] * x_df[i*sram_stride] * dB`; 1267 uses `((float)C.x[l]) * ((float)sc * x_df) * dB`. A deterministic 200,000-case host F32 fixture with binary16-representable scales, int tile values in [-65536,65535] and int8 subscales found zero nonzero numerical differences, but 391 signed-zero bit-pattern differences. This is host-only arithmetic evidence, not a HIP FMA/codegen or model-parity result. Preserve the 1006 cast and require full-vocabulary/logit, greedy and MTP acceptance comparisons; test finite extremes, zero/sign, K/J boundaries, and the non-MMA route. Never treat a Q6_K dispatch marker alone as MMA-path activation.

**B3: production composition and materiality.** Compare (A) current validated production composition **with 1006** and no 1267, versus (B) identical composition plus 1267; same GGUF, exact Q6_K dense MMQ shapes, architecture, driver, quantization, graph settings, and allocator. Run offline mechanics and source diff first; prove 0300/1006/1267 order-independent application, correct baseline hashes, and PEF01 memory-safety/HI71 dense-shape eligibility (PRBE04). If existing kernel/rocprof timing cannot support a theoretical >=3% end-to-end prefill improvement, **retire 1267 without another hardware campaign**. Historical +2.47% pp512 does not itself meet a >=3% deployment threshold.

## Bounded implementation / benchmark gate

1. Host-only: patch lint, rebase/apply/idempotence and negative anchors against b11474 with 0300+1006; inspect composed Q6_K line and verify the patch is not already upstream-absorbed. Static selector fixture for Q6_K on gfx1100/gfx1201/gfx1030: ne11=8/9/15/16/17/23/24/31/32/33/47/48/49/63/64/65/80/81/127/128, both fallback values, src1 precision and shared-memory gate. Check padding, launch J, and zero-work/unsupported cases. If #29953 parity is missing, block GPU timing and hand only that correctness dependency to the existing upstream-fix owner.
2. On one idle gfx1201, after the safety gate: capture **actual** Q6_K MMA kernel symbol/launch counts, tile J, active shape and time with 1267 off/on, including pp512/pp2048, ub16/128/512, one long-context run and a multi-ubatch/tail case. Negative controls: gfx1030 DP4A, Q8_0, and decode tg128. No competing Radiance or Flash-Next job. Do not add new configuration, cache, allocator, or scheduler.
3. Only if non-overlapped E2E ceiling >=3% and exact numerical/graph/memory gates pass: run four independent gfx1201 sessions, >=10 paired ABBA rounds each, on the same production model. Require subject-only **MMA** activation, full-vocab/logit/KLD and greedy/MTP acceptance parity, multi-request same-process, graph capture/replay and memory safety. Promotion: CI95-low >=3% **end-to-end** pp512 improvement, <=1% regression on pp2048/tg128/other controls, no missing work/bytes or unsupported architecture activation. Otherwise retire/default-off. Existing `PRBE110-RD07-Q6K-MMQ-SCALE-FOLD` producer is the owner; strengthen its activation evidence rather than building a second producer.

## Ownership and external alternatives

PRBE110/1267 owns the scale-fold mechanism and terminal decision; PRBE04 owns PEF01/HI71 downstream safety/eligibility requirements, not a second patch. PA35/1006 owns the validated Q6_K float-promotion/codegen fix; 0300 owns forced-J; the pin/upstream-fix owner owns #29953/#30168 parity. Rejected 1203 and 1266 stay terminal. Existing MMQ config/dispatch and telemetry remain authoritative.

Upstream `ggml-org/llama.cpp` master inspected 2026-10-10 still has the unfused Q6_K MMA scale chain, so 1267 is **not** source-absorbed; master differs in config-helper precision and staging correctness. [#30021](https://github.com/ggml-org/llama.cpp/pull/30021) and [#30022](https://github.com/ggml-org/llama.cpp/pull/30022) are open GCN MMQ tuning/stream-K work, not a transferable gfx1201 Q6_K scale-fold result. [#25940](https://github.com/ggml-org/llama.cpp/pull/25940) remains open and is the source for 1006. [vLLM GGUF plugin](https://github.com/vllm-project/vllm-gguf-plugin) has a separate legacy Q6_K MMQ and Triton Q6_K GEMM; no demonstrated gfx1201 replacement or additive gain. Do not import another Q6_K dispatcher.

## Evidence classification

**Measured BigCherry (historical b11126):** 1267 gfx1201 +2.4675% pp512 (CI95 2.2776–2.6508%), tg128 -0.0331%; 1006 independent gfx1201 approximately +18% pp512, current production-selected. **Repository/source:** 1267 patch, producer, stored validation, 1006 patch/recipe, pinned/current upstream. **Host fixture (this audit):** 200,000 scale-chain cases and RDNA4 J capacity model. **Not performed:** repository pytest, patch composition, HIP build, GPU execution, fresh benchmark. No new performance claim.
