---
id: PRBE111
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-10-04T04:04:36+11:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# IQ MMVQ decode: source-qualified 1273 correctness and descriptor disposition

## Authoritative audit (2026-10-09; pinned llama.cpp b11474)

**Disposition: pending, no implementation or hardware queue.** The previously proposed "hoist block-invariant IQ scale/metadata from the innermost loop" is largely **already implemented** in the pinned `vecdotq.cuh` for IQ4_XS/IQ3_XXS. A more immediate correctness constraint exists in evaluated patch 1273: lowering VDR changes the integer/float reduction partition, so its claim of unchanged quant math must not be interpreted as stock-bit-identical output. This audit supersedes the older speculative metadata-hoist and unconditional descriptor-cleanup steps. It is source/host evidence, not a GPU correctness failure or measured speedup.

### Exact source and caller/ownership trace

- `ggml/src/ggml-cuda/mmvq.cu::get_vec_dot_q_cuda(type)` and `get_vdr_mmvq(type)` are two compile-time switch functions over `ggml_type`. They are used by `mul_mat_vec_q<type,...>` to bind the function and loop stride; `mul_mat_vec_q_switch_fusion` chooses fused/plain launches. `get_device_table_id`, `calc_nwarps`, `calc_rows_per_block` already own geometry. `mul_mat_vec_q_moe` is a distinct multi-token MUL_MAT_ID path and is NOT covered by 1273's `ncols_dst==1` tuning.
- `ggml/src/ggml-cuda/vecdotq.cuh::vec_dot_iq4_xs_q8_1` loads the IQ4 nibble/table pairs inside its four-iteration loop, then reads block scale `scales_l/scales_h`, multiplies the **integer** `sumi`, and applies F32 `d` once outside the loop. The scale is not redundantly decoded per inner iteration.
- `vec_dot_iq3_xxs_q8_1` reads `aux32` once, unpacks **different sign fields for each l0**, then applies `ls=aux32>>28` and integer `(ls*sumi+sumi/2)/2` once after the four two-entry loop iterations. Moving sign extraction wholesale outside the loop would require materializing distinct l0-dependent values and proving VGPR/ISA benefit; moving scale out is already done.
- Patch `1273_iq_mmvq_rdna_tuning/patch.py` introduces IQ4 VDR2 (two iterations) and IQ3 VDR1 (two sign-pair iterations) and a third template function selector `bigcherry_iq_vec_dot_q_cuda<type,vdr>`. Its two independent env gates are cached in function-local `static const bool`; the once-per-process marker proves that *a* variant was selected, not the number or share of timed dispatches. The patch has no per-invocation allocation or new transfer, but template variants affect captured HIP graph kernels; separate processes per A/B arm are mandatory.
- `1273` requires 0600 geometry and 1241 F32 activation; it is `evaluated`, not validated. Its README and SUMMARY contain no auditable hardware result or current-pin correctness receipt. `releases/evidence/bpb01-four-evaluated.md` explicitly records that absence. No 1273-specific run or independent edit occurred within the preceding 12 hours.
- Inspected upstream master `mmvq.cu` is text-identical to pinned b11474, so no newer upstream IQ descriptor or hoist was found. Upstream HIP PR #27962 (merged 2026-10-07) already replaces `__vsub4`/`__vcmpne4` with SWAR operations in `vendors/hip.h`, present in b11474: do not re-port that IQ2/IQ3 instruction change.

### Quant-math correctness discriminator (completed, host only)

**IQ3_XXS:** Stock VDR2 returns `d * T(s0+s1,ls)` for two four-term halves, with C++ integer truncation toward zero and `T(s,ls)=(ls*s+trunc(s/2))/2` (final division also truncates). VDR1 returns separate `d*T(s0,ls)` and `d*T(s1,ls)`, later added as F32. The integer map is not additive: for `s0=s1=1, ls=1`, stock `T(2,1)=1` while split `T(1,1)+T(1,1)=0`. Exhaustive deterministic signed host arithmetic for `s0,s1 in [-32,32]`, `ls in [0,15]` found **25,344 / 67,600** unequal integer results (37.49%, differences ±1). These are synthetic partial sums, not a measured distribution of real quantized blocks or a GPU output mismatch. The non-additivity is sufficient to reject any assertion that VDR1 is algebraically bit-identical for all inputs.

**IQ4_XS:** Stock VDR4 applies the integer scale to a full four-term sum, then one F32 multiply; VDR2 applies it separately to two partial sums and adds two F32 products. Integer math is distributive, but F32 multiplication/addition may round differently. A seeded NumPy float32 host model over 131,072 synthetic pairs observed **51,459** different F32 bit patterns (39.26%). This establishes a possible reassociation difference, not its frequency on hardware. GPU FMA, compiler contraction, warp accumulation and actual IQ block distributions were not simulated.

**Immediate implication:** Keep both VDR candidates default-off and unpromoted. The IQ3 VDR1 arm requires either an explicit approximate-numerics acceptance policy (not currently approved) or a redesigned full-integer accumulation before the single `T` operation; the latter changes the vec-dot interface/partition and is NOT a cheap hoist. Do not claim exact greedy or MTP acceptance without device evidence. IQ4 VDR2 must independently prove F32/logit and greedy parity.

### Implementation-ready decision sequence (one narrow owner)

1. **Gate 0, cheap and mandatory:** Reuse the existing 1273 package and `iq-mmvq` experiment, not a new patch. Add a package-local host fixture that decodes real packed IQ3_XXS/IQ4_XS + Q8_1 blocks with the *stock and 1273 split integer order*, including negative sums, odd signs, scale extremes and exact output bit patterns. Explicitly compare raw integer partials before F32. If IQ3 VDR1 changes values, mark that arm numerically non-equivalent and **terminate exact-parity promotion**; do not silently relax MTP correctness.
2. **Gate 1, activation and cost:** On a fresh b11474 composition, use existing rocprofv3 + per-device timing to count actual `mul_mat_vec_q` IQ4_XS/IQ3_XXS single-token launches separately from MMQ, `mul_mat_vec_q_moe`, 1347 thin MMVF, 1350 MMQ, graph replay and AllReduce. Record `type, arch, ncols_dst, vdr, nwarps, graph mode, kernel count/duration, VGPR/SGPR, scratch, ISA`; require marker *and* matching timed kernel. Single-card gfx1100/gfx1201 first; gfx1030 negative control; no P2P assumption. If attributable hot-loop time caps theoretical E2E improvement below **3%**, close PRBE111 without descriptor edits or A/B.
3. **Descriptor-only candidate, conditional:** Only if maintenance/codegen benefit remains after 1273 reconciliation, implement a **single compile-time traits template** in `mmvq.cu` providing `vec_dot` and `vdr` together for every supported type (including default-VDR cases), while retaining the existing geometry/arch tables. `template<ggml_type type,int vdr_override=0> struct mmvq_dot_traits` resolves direct call and stride. Make 1273's override an optional specialization of this *same* trait; no second registry, dynamic function pointer, new env flag, allocator or kernel dispatch. Diff generated ISA and symbol/instantiation counts for baseline vs descriptor-only; reject if function-pointer indirect calls, code size, VGPR or kernel time regress. No descriptor refactor is a prerequisite to testing existing 1273.
4. **Only if Gate 1 survives:** Prototype one **IQ4_XS-only** compile-time-gated register-local nibble/table decode variant, based on a measured redundant instruction/gather. Do not redo the already-hoisted scale, the merged SWAR HIP operations, or 1273 VDR/nwarps tuning. Inspect gfx1100/gfx1201 ISA before adding a `v_perm_b32` equivalent: the external NVIDIA `__byte_perm` PRMT mechanism is not a portable HIP primitive. If a device-native byte permute and identical lookup mapping cannot be demonstrated, terminate. IQ3_XXS is a separate future gate after its 1273 numerical semantics are resolved.
5. **Correctness before performance:** Pinned full-vocabulary backend-op/reference, F32 logits and KLD, deterministic greedy tokens, MTP draft acceptance, IQ3 negative/scale-edge cases, same-process multi-request, multi-ubatch, ctx 8K/80K, graph capture/replay/reallocation, and controls Q4_K/Q6_K/Q8_0 and gfx1030. Prove expected vs observed kernel counts/work; a faster arm with skipped work fails. Do not interfere with active Flash-Next accuracy/1357/1358/1347 lanes or queued hardware.
6. **Promotion:** Isolated subject/control binaries or process-restarted A/B arms; gfx1100 XTX and gfx1201 R9700, tg128/tg512 plus pp512 control, 4 independent sessions x >=10 paired ABBA rounds per architecture. Require bitwise backend parity for any arm claiming exact quant math, greedy identity and unchanged MTP acceptance, CI95-low >=3% attributable E2E decode improvement, <=1% prefill/control regression, stable registers/scratch, and no extra H2D/D2H. Otherwise retire the candidate; retain stock and evaluated 1273 as historical evidence.

### Consolidation and external mechanism disposition

PRBE111 owns only *future* IQ descriptor/hot-loop experiment. Existing 1273 owns VDR/nwarps experimental variants; 0600/0650 own generic MMVQ geometry; PRBE47/BCOP84 own Q6_K DPP reduction; PRBE21 owns small-K MMVQ/SSM candidates. Do not create a new MMVQ selector or parallel experiment. Active QFP36/1357, QFP41/1358, QFP35, MTP/Flash-Next 1334/1347/1350, Radiance and engine work are protected by the 12-hour exclusion and untouched.

The external `localweights/vllm-gguf-plugin` fork describes register-only IQ4_XS nibble PRMT, a packed block-header load and multi-column reuse, reporting RTX 3090 Ti results. Its README references `csrc/gguf`, but that source path was not present in the inspected GitHub root listing; treat these as **unverified fork mechanism claims**, not code-reviewed or AMD measurements. NVIDIA PRMT/occupancy and vLLM graph layouts do not transfer without RDNA ISA/shape qualification. Upstream SYCL #30226 (open 2026-10-09) pairs Q4_K multi-column rows on Intel Xe2, not the single-column HIP IQ path; do not port it. Upstream HIP #27962 is already absorbed by the pin.

**Measured evidence:** 1273's evaluated status records historical hardware exercise but its package/evidence review supplies **no auditable per-arm performance numbers or full-vocab correctness receipt**. No BigCherry IQ4_XS/IQ3_XXS speedup is established by this audit. The 25,344/67,600 and 51,459/131,072 figures above are synthetic host arithmetic, not hardware measurements.

Sources: pinned [mmvq.cu](https://github.com/ggml-org/llama.cpp/blob/b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea/ggml/src/ggml-cuda/mmvq.cu), [vecdotq.cuh](https://github.com/ggml-org/llama.cpp/blob/b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea/ggml/src/ggml-cuda/vecdotq.cuh), [HIP #27962](https://github.com/ggml-org/llama.cpp/pull/27962), [SYCL #30226](https://github.com/ggml-org/llama.cpp/pull/30226), [vLLM fork](https://github.com/localweights/vllm-gguf-plugin), BigCherry 1273 patch.py/README/SUMMARY, `releases/evidence/bpb01-four-evaluated.md`.

## Files

- Authoritative: `docs/planning/active/patching-rdna-boost-experiments/PRBE111.md`.
- Existing experimental package: `engines/llamacpp/patches/1273_iq_mmvq_rdna_tuning/{patch.py,patch.toml,README.md,SUMMARY.md}`; no implementation change in this audit.
- Pinned source: `ggml/src/ggml-cuda/mmvq.cu`, `vecdotq.cuh`, `vendors/hip.h`; future tests belong to existing `tools/tests/patch/` and `tools/lab/flash-next/` conventions.

## Validation

Completed this audit: 14/14 source-static assertions against pinned `mmvq.cu`, `vecdotq.cuh` and 1273 patch.py; 67,600 exhaustive signed IQ3 integer partitions; 131,072 seeded float32 IQ4 reassociation cases. No repository pytest, HIP build, GPU execution, new benchmark or new experiment queue. Tests demonstrate arithmetic *possibility* and source structure, not actual GPU divergence.

## Acceptance Criteria

A current-pin, independently activated IQ MMVQ kernel must contribute >=3% theoretical E2E headroom before a new implementation. Exact-numerics candidates must preserve bitwise kernel/backend outputs, greedy and MTP acceptance. New descriptor must not change generated direct-call code or resource use. Every future experiment has a bounded reject/close outcome and remains subordinate to existing 1273/0600/0650 ownership.

## Notes

Historical 2026-10-04/08 proposals to hoist scale and automatically merge type selectors are superseded by the source-level evidence above. No production or active experimental implementation was changed.

## Change Log

- 2026-10-08 (triage): Kept pending at P1. Decode shortlist rank #2 after PRBE113 Gate 0. IQ4_XS then IQ3_XXS MMVQ inner-loop metadata hoist, existing vector dispatch descriptor compile-time only, gfx1100/gfx1201; no new registry. Compare kernel fraction and resource use vs PRBE113 baseline; don't start without hot-loop evidence.

- 2026-10-04T04:04:36+11:00 (agent): Original side-branch plan created.
- 2026-10-05: Transplanted to `patch-refactor` as free sequence slot PRBE111; removed stale branch-history assumptions and aligned ownership with current IQ/HIP tuning plans.
