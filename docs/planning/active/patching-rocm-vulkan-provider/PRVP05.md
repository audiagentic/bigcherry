---
id: PRVP05
order: 5
plan: patching-rocm-vulkan-provider
state: pending
created-at: '2026-10-19T08:13:00+00:00'
created-by: agent
priority: P2
work: M
---

# Vulkan F32 RMSNorm wave-local reduction on RDNA3/RDNA4 — upstream-first admission

## Decision and boundary (BCOP322; 2026-10-19)

**Pending, default-off; no local shader fork, benchmark queue or implementation while RRVP02's Vulkan implementation pause remains active.** Upstream [llama.cpp #29882](https://github.com/ggml-org/llama.cpp/pull/29882) is OPEN (inspected 2026-10-19). It changes the **Vulkan F32 RMSNorm shader reduction**, not RNX08's HIP vector-load/register-reuse mechanism. No existing PRVP item owns this shader: PRVP02 owns CM1 A-prefetch safety, PRVP03 owns Vulkan AllReduce, and the historical PRVP04 branch concerns Q4 KV transpose. This new item owns only the narrow RMSNorm shader admission/disposition. Do not change those items or RNX08.

**Eligibility:** selected independently active branch `feat/rcs09-04-expert-index` at Audumla `d54690b0` (2026-10-19 08:09:30 UTC; one commit ahead of main `504282d0`). No repository-visible independent RMSNorm/Vulkan-subgroup implementation, patch, benchmark, PR or queue was found in the preceding 12 hours; last text-matching RMSNorm commits were in August. Protected active work includes RCS09-04/v14 shape, RAD11 FP8, RAD12 small lanes, RAD10 counters, RCS05 helper/backend and WSL/release changes. No protected file was edited. Off-repository queues are unknown. Novelty: a **distinct Vulkan wave64 subgroup reduction** with AMD benchmark evidence and an inspected shader diff, not the recent RNX08 HIP RMSNorm audit.

## Pinned implementation and actual work

Pin: llama.cpp b11552 `23b0202a189c44a54625aadcb37a946dd1d6278d`.
- `ggml/src/ggml-vulkan/vulkan-shaders/rms_norm.comp::rms_norm`: fixed `BLOCK_SIZE=512`, each lane reads `col=tid+t*512`, accumulates `xi*xi`, stores `sumsh[tid]`, then performs a 512->1 shared-memory binary tree with offsets 256,128,64,32,16,8,4,2,1 and a workgroup `barrier()` after each level. A barrier also precedes the reduction; the outer channel/sample loop and fused ROPE path have separate barriers. Each logical input is read for the norm and again for output; this candidate does **not** remove that reread or any required output work.
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp::ggml_vk_load_shaders` creates the F32 RMSNorm, RMSNorm+MUL, +ADD, +ADD+MUL, and RMSNorm+RoPE variants. `ggml_vk_op_get_pipeline` and `ggml_vk_rms_norm` select/submit them; partials and SET_ROWS variants have their own paths. Preserve the existing fusion matcher, descriptor ownership, strides, queue/command-buffer lifetime and native fallback. Do not add a dispatcher/allocator/telemetry framework.
- #29882 replaces the per-lane tree with `subgroupAdd(sum)`, one elected lane writing `sumsh[gl_SubgroupID]`, then a workgroup barrier and either direct sum of <=4 subgroup partials or a second `subgroupAdd`. It adds specialization-controlled 128/256/512-thread pipelines, selecting 128 for `ncols<=128`, 256 for `<=256`, otherwise 512. The shader's shared array scales with block size; fused ROPE uses `2*BLOCK_SIZE` scratch. It gates subgroup compilation on vendor AMD/Intel/NVIDIA **and** `subgroup_arithmetic && subgroup_require_full_support`.
- On an actual wave64 AMD driver the variants have 2/4/8 subgroups. The original 512-thread tree uses 9 shared reduction barriers; the proposed subgroup reduction uses one inter-subgroup barrier for the norm (not counting outer/fused barriers). **This is an instruction/synchronization opportunity, not a measured time or bandwidth saving.** Old F32 `sumsh` is 2048 B/workgroup; new 128/256/512 variants allocate 512/1024/2048 B for that array, subject to compiler resource accounting. No wave32/64 ISA, VGPR, spills, LDS occupancy or physical memory transactions were measured here.

## Upstream and other-engine disposition

- #29882 source/diff: https://github.com/ggml-org/llama.cpp/pull/29882/files ; status: OPEN 2026-10-19. Prefer pin absorption after the Vulkan pause, **not** a BigCherry duplicate shader. If merged/absorbed, close `PIN_ABSORBED` after compatibility checks. If rejected, do not port without new BigCherry evidence.
- External AMD RX 7900 XT (proprietary Vulkan driver, reported wave64) Qwen3-0.6B Q4_K_M: pp1024/FA-off 18,923.05 -> 20,291.91 tok/s (~+7.2%), while Qwen3.5-9B Q6_K pp512/FA-off 2,041.75 -> 2,058.44 (~+0.8%) and tg128 86.62 -> 86.32 (~-0.35%). These are **upstream cross-build, five-repeat reports**, not matched BigCherry gfx1100/gfx1201 evidence or confidence-qualified promotion. The small-model prefill gain cannot be transferred to a multi-rank Flash-Next bottleneck.
- vLLM `vllm/model_executor/layers/layernorm.py::RMSNorm.forward_native` delegates to its custom RMSNorm operation, with a separate batch-invariant numerical path; its runtime/kernel contract is not a drop-in Vulkan SPIR-V shader. Use only as a numerical/control comparison, not an AMD speed claim: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/layernorm.py .

## Cheapest discriminator — completed host-only, not GPU validation

A disposable Python integer-sum/coverage oracle modeled wave64 with block sizes 128/256/512, 17 row widths (1..8192 including 63/64/65, 127/128/129, 255/256/257, 511/512/513), and four deterministic signed patterns. **204/204 cases passed**: exactly one read of each logical norm element, no padded tail element included, correct subgroup partition and exact integer reduction. This tests address/work accounting only. It does **not** establish F32 bit identity, shader compilation, numerical equivalence, Vulkan synchronization correctness, GPU occupancy, or performance. No Vulkan/HIP build, repository test suite or hardware benchmark ran.

## Implementation-ready gate after the Vulkan pause

1. **Source/pin admission:** resolve #29882 merged/rejected/head SHA and exact pin; inspect shader and `ggml_vk_rms_norm` host dispatch. Record actual device subgroup size, subgroup arithmetic/full-support features, driver (RADV/AMDVLK/proprietary), pipeline variant, `ncols`, fused mode, row count, physical rank and shader SHA. Keep 512-thread tree as stock control and existing partials/fused paths untouched.
2. **Correctness:** backend-ops F32 RMSNorm and fused MUL/ADD/ROPE/SET_ROWS controls; ncols {1,63,64,65,127,128,129,255,256,257,511,512,513,1024,4096,8192}; strided/broadcast rows, channel/sample loop, zero/NaN/Inf/near-zero variance, epsilon, signed zero, and both wave32/64 if actually supported. Test multi-request same-process, multi-ubatch, graph/descriptor reuse, long-context logits/KLD and MTP acceptance. Floating reduction order changes: **do not require bit identity where impossible**, but report differences and enforce independent reference tolerances; reject any missing output or changed physical work.
3. **Resource/physical accounting:** pipeline executable register count, LDS bytes, wave occupancy, spills, shader dispatch counts, all input/output bytes and workgroups. Verify no regression in fused ROPE barriers or partials. No extra materialization, hidden CPU fallback, duplicated launch, or graph allocation.
4. **Cost admission:** obtain current-pin per-rank norm kernel time and overlap-adjusted critical-path contribution on dual gfx1100, gfx1201 and mixed 2xXTX+R9700; separately decode width1, MTP3 verify, pp512/ub512 and long prefill. Reject `NO_ROUTE` if no eligible Vulkan path, `BELOW_CEILING` if optimistically removable exposed end-to-end time is <2.9126%, or `NO_GAIN` if paired A/B is neutral. Do not infer E2E ceiling by summing concurrent GPU times.
5. **Promotion:** only after pause lifted and admission positive, compare native pinned vs absorbed upstream on matched model/build/driver, A/A and ABBA >=10 paired rounds, CI95-low >=3% end-to-end throughput improvement, no material regression in pp/tg/MTP controls, and complete correctness. Otherwise leave native fallback and terminate. No new configuration surface; use existing Vulkan pipeline capability gate.

## Dependencies and ownership

RRVP02 Vulkan implementation pause is a hard dependency. RNX08 HIP RMSNorm and PRBE15 HIP IMRoPE scatter remain separate; PRVP02 CM1 and PRVP03 transport are untouched. Existing Vulkan pipeline selection and shader generation own any eventual adoption. BCOP322 is only the action ledger.
