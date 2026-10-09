---
id: PRBE47
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:56:45.120291+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: M
priority: P3
---

# UP-HIP-002: AMD DPP/native shuffle path


## 2026-10-09 authoritative audit: wave32 DPP reduction, not a blanket shuffle replacement

**Disposition:** dormant P3; source/host discriminator completed, no patch or hardware lane queued. Only consider an opt-in Q6_K MMVQ reduction experiment if a current-pin ISA and critical-path census shows a genuine opportunity. The old DPP code sketch and its row-shift assumption are superseded by this section. This is NOT a demonstrated speedup.

### Verified source and ownership

Pinned llama.cpp b11474 (b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea) and upstream master inspected 2026-10-09:

- ggml/src/ggml-cuda/common.cuh::warp_reduce_sum<float> (~468-474): butterfly offsets **16,8,4,2,1** through __shfl_xor_sync, float additions after each exchange. The float2, half2 and integer overloads are separate and are **not** in scope.
- ggml/src/ggml-cuda/mmvq.cu::mul_mat_vec_q (~600-835): generic type-templated Q8_1-activation MMVQ. At ~795 and ~798, normal and optional fused-gate sums are reduced **after** cross-warp shared-memory accumulation and __syncthreads(). Preserve that order and every barrier. All quant types instantiate these sites.
- ggml/src/ggml-cuda/mmvq.cu::mul_mat_vec_q_moe (~843-990): a distinct multi-token MUL_MAT_ID path, reached when has_ids && ncols_dst > 1. It has no inter-warp shared reduction; at ~948/951 it reduces normal and optional fused-gate sums independently. Previous PRBE47 text omitted this second pair of call sites. Both paths require separate eligibility/activation evidence.
- ggml/src/ggml-cuda/mmvq.cu::get_device_table_id (~106-122) already selects RDNA3_0 / RDNA4 at compile time. The existing 0600/0650 MMVQ geometry and 1273 IQ tuning packages own geometry/dispatch; do not add a table, scheduler, allocator, runtime selector or global warp_reduce_sum override.
- No DPP MMVQ patch is present in the BigCherry package inventory. Current upstream master still uses the same float shuffle reductions. This audit did not inspect generated HIP ISA, so the stock compiler might already lower exchanges efficiently.

**Critical lane constraint:** AMD DPP16 row_xmask only addresses XOR partners within each 16-lane row. In wave32, XOR 16 crosses the row boundary and cannot be replaced by row_xmask:16. The historical row_shr:1 sketch is not an XOR butterfly and is also invalid for this reduction. Use the existing full-wave __shfl_xor_sync for offset 16, then consider DPP row_xmask for offsets 8,4,2,1. Do not copy the old update_dpp float sketch or its incorrectly grouped preprocessor condition. AMD ROCm's wavefront guide and LLVM's DPP16 syntax confirm the row boundary; rocPRIM warp_reduce is a reference for warp/participation constraints, **not** a drop-in bit-identical replacement.

### Cheapest discriminator (completed, host only)

A deterministic NumPy float32 lane simulator compared (A) the stock XOR-16/8/4/2/1 butterfly, (B) XOR-16 + four row_xmask stages with identical addition order, and (C) an invalid row-only reduction. **8,195/8,195** full-wave inputs produced bit-identical 32-lane outputs for A/B; **8,195/8,195** exposed a difference for C. Inputs included 8,192 seeded random vectors plus ramp, unequal half-wave and cancellation vectors. This verifies lane-index algebra and float32 addition order only; it does **not** prove AMD DPP builtin codegen, masked-lane behavior, GPU correctness or speed.

### Implementation-ready decision sequence

1. **No patch until ISA Gate 0.** On the installed ROCm compiler, build the existing composed b11474 baseline and inspect gfx1100/gfx1201 disassembly for Q6_K mul_mat_vec_q and mul_mat_vec_q_moe, including fused and unfused instantiations. Record actual v_dpp/ds_bpermute/v_permlane instructions, VGPR/SGPR, scratch, LDS, wave32, and kernel timing/call counts with existing rocprofv3 tooling. If the compiler already emits an equivalent DPP sequence or the targeted reduction's *measured* upper bound is <3% of E2E decode, close without patch or queued A/B. No GPU result is asserted here.
2. **If Gate 0 survives, prototype one additive helper, never replace common.cuh's generic overload.** Only HIP + (RDNA3_0 or RDNA4) + physical warp_size==32 + type==GGML_TYPE_Q6_K; preserve other architectures/types, including gfx1030, and keep normal/fused-gate reduction paths in both kernels aligned. A type/architecture compile-time gate must resolve before codegen. Use bit-preserving float-to-i32 reinterpretation for mov_dpp and reinterpreted result, not numeric conversion. DPP control encodings and availability must be checked against installed ROCm headers/ISA; use row_mask=bank_mask=0xf and require a full participating wave.
3. **Pseudocode, not compile-ready code:** for offset in [16,8,4,2,1], partner = offset==16 ? native_shfl_xor(x,16,32) : bitcast_f32(mov_dpp(bitcast_i32(x), verified_ROW_XMASK(offset),0xf,0xf,false)); x = float32(x + partner). Keep each addition and its order. Do not use row_shr, row_bcast, an unverified 0x111 control word or a wave64 fallback.
4. **Bounded host/device tests before timing:** a host lane-permutation fixture (all 32 lanes; nonuniform halves; signed zero, finite extremes, NaN handling documented); a tiny HIP 32-lane kernel comparing every lane bitwise against native shuffles on gfx1100/gfx1201; reject partial-wave/unsupported-width configurations. Test both normal and fused gate, Q6_K ncols 1..8, MUL_MAT_ID ncols 1 vs >1, multi-ubatch, repeated same-process requests, graph capture/replay, long context, full-vocabulary logits, greedy outputs and MTP acceptance. Include Q4_K/IQ4_XS/Q8_0/gfx1030 negative controls with unchanged ISA and outputs.
5. **Only then performance qualification:** baseline composed b11474 versus baseline + Q6_K-only DPP package, identical model/quant/graph/ubatch, single gfx1100 and gfx1201 first; optional dual-XTX tensor-split control later to isolate communication from per-device kernel effects. Measure tg128/tg512, 8K/80K context, per-kernel critical-path share, LDS/VALU instructions, register pressure, clocks and E2E throughput. Four independent sessions per architecture, >=10 paired ABBA rounds/session, CI95-low >=3% E2E decode improvement, <=1% prefill/negative-control regression, exact work and output parity. If no causal kernel win or codegen regresses, reject and remove the candidate. Only consider Q8_0 in a **separate** follow-up after verifying the F32-activation 1241/1301 routes and MTP acceptance; do not widen this slice.
6. **Fallback/rollback:** baseline native shuffle, no new state, no memory lifetime changes, no graph topology changes, no GPU-to-GPU transfers. Do not modify active QFP/MMQ/Meta/MTP plans or queue experiments against occupied hardware.

### Consolidation, measured context and sources

PRBE47 owns only Q6_K reduction-lowering proof. PRBE111 owns IQ4_XS/IQ3_XXS vec-dot/VDR metadata and hoisting; PRBE21 owns small-K geometry through 0600/0650; 1273 owns IQ tuning; 1204's rejected Q6_K VDR2 campaign is a **different** mechanism (gfx1201 decode +0.204%, CI95 [+0.094,+0.330] versus required +0.3% low, prefill -0.089%). These historical numbers warn against assuming microkernel gains; they are not DPP measurements. Theoretical E2E gain cannot exceed measured time attributable to the targeted reduction.

Source references:
- https://github.com/ggml-org/llama.cpp/blob/b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea/ggml/src/ggml-cuda/mmvq.cu
- https://github.com/ggml-org/llama.cpp/blob/b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea/ggml/src/ggml-cuda/common.cuh
- https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/cross-arch/wavefront-ref/dpp-builtins.html
- https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUModifierSyntax.html
- https://rocm.docs.amd.com/projects/rocPRIM/en/latest/warp_ops/reduce.html

## Historical provenance (superseded design notes)

Supersedes: RD56
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd56

Supersedes RD56.

2026-09-24 relevance at b11126: no existing patch (RD56 grep = no hits). Not upstream-absorbed -- common.cuh's warp_reduce_sum family read directly at lines 459-495, confirmed __shfl_xor_sync-only, no DPP path exists anywhere in ggml-cuda (grep for amdgcn_*dpp across ggml-cuda = zero hits). GPT design requests this batch were unusable; plan authored directly from verified source.

2026-09-24 GPT review req_e17e0bf5a68c48d5 applied: corrected the target call sites from vecdotq.cuh's vec_dot_q6_K_q8_1* (which does not call warp_reduce_sum) to the real generic, type-templated reduction sites in mmvq.cu (confirmed at mmvq.cu:795,798,937,940); specified an if-constexpr type-gated specialization at those sites rather than a blanket redefinition.

## Change Log

- 2026-10-09 (BCOP84): replaced invalid DPP row-shift sketch and stale four-site inventory with source/host-qualified hybrid wave32 decision gate; no patch/hardware result.

- 2026-10-08 (triage): Kept pending at P3. No RD56 DPP MMVQ specialized patch or measured gfx1100/gfx1201 ISA/perf evidence. Native warp_reduce_sum at mmvq.cu's type-shared sites must remain for unqualified quant types; profile then isolate. Keep pending.

- 2026-09-09T10:56:45.120291+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:13:58.680914+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.341721+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.115748+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:08:59.693148+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- 2026-09-10T03:09:19.555246+00:00 (updated-by): Updated: section:acceptance_criteria
- chg_20260910_030930_repaired-two-more-active-patch_7368
- 2026-09-10T03:09:30.744289+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T04:46:57.296925+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:standards, section:notes
- 2026-09-24T05:08:29.772808+00:00 (updated-by): Updated: section:description, section:steps, section:notes
