---
id: PRVP04
order: 0
plan: patching-rocm-vulkan-provider
state: deferred
created-at: '2026-10-18T05:12:00+00:00'
created-by: agent
work: S
priority: P2
---

# Vulkan Q4_0 KV fused dequant-transpose: prefill-only admission (upstream #30738)

## Description



## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

2026-11-22 owner handoff from decision-only BCOP295: Unsectioned source gate. Source-bounded action/stop rule: **Discovery:** Open upstream llama.cpp [#30738](https://github.com/ggml-org/llama.cpp/pull/30738) (2026-10-17) adds a Q4_0 dequant+transpose SPIR-V variant and pipeline registration for the existing Vulkan FA scratch path. BigCherry b11552 only has Q8_0. The pinned path requires **Q rows >=64**, quantized dense K and V, both transpose pipelines, buffer-range capacity and !coopmat2. Thus the PR's GCN1 tg128 improvement is not proven to arise from this shader for ordinary n=1 decode. External pp512 gain is not BigCherry gfx1100/gfx1201 evidence. **Owner/consolidation:** PRVP04 owns this one Vulkan FA shader admission; PKC04's completed lifecycle ownership is unchanged, PRVP02 retains CM1 safety, TRVP12/13/14/15 provide shared identity/route/replay, RRVP02 preserves the user-directed Vulkan implementation pause. HIP RNX02, PRBE54 and QFP04 are separate kernels. No new selector, allocator, cache, dispatch table, local patch or hardware job. **Branch/exclusion:** Selected independently active `feat/rad06-profile2` at `a80aa61e45fcb7b87dc1e68306064169d18ead5e` (Audumla, 2026-10-18 04:10:39 UTC), later independent than main's `c06dd0a9` (04:08:34 UTC); main's later BCOP294 merge was sched See docs/planning/completed/patching-bc-optimizations/BCOP295.md for the complete source/shape/negative case and evidence. PRVP04 is the only active implementation owner; this handoff does not qualify a kernel, change a source pin or authorize extra dispatch code.

DEFERRED VULKAN PORTFOLIO 2026-11-22: Explicit user-directed Vulkan implementation pause from 2026-09-10 remains in force. Vulkan Q4_0 KV prefill-only fused dequant/transpose is paused, and its external GCN1 decode improvement cannot be attributed to an n>=64-query shader on BigCherry RDNA devices. This item is parked as deferred, NOT completed, superseded, deprecated, implemented or promoted; its original source/negative/hardware acceptance remains mandatory. Resume only via Planning API state->pending after a new owner review confirming: Resume after RRVP02 unpause and upstream/pin reconciliation of #30738. Require real Q rows>=64 with quantized K and V, both transpose pipelines, buffer-range/temporary F16 scratch and no-coopmat2 eligibility, then gfx1100/gfx1201 SPIR-V/full-logit/long-context correctness and measured prefill benefit. Do not invent a decode gain. Shared HIP/ROCm identity, SDK, provider capabilities and benchmark controls remain active under their separate owners; this deferral does not block them.

## Decision and ownership (2026-10-18)

**Plan only; no implementation or Vulkan hardware queue while RRVP02's 2026-09-10 pause applies.** PRVP04 owns only the Q4_0 K/V dequant+transpose shader and its activation/performance disposition. This is a newly identified native Vulkan attention implementation boundary, not a second Vulkan dispatcher, cache, scheduler, allocator, or tuning registry. PKC04 is a completed lifecycle *ownership* decision; PRVP02 owns unrelated CM1 MMQ A-prefetch bounds; TRVP12/13/14/15 and RRVP02 own shared provider identity, executed-route attestation, replay, and future qualification. HIP RNX02/1300, PRBE54/1271 and QFP04/1295 are separate dense Q8 vector, HIP Q5 staging, and sparse-QSA paths: do not borrow their measurements or edit them.

**Novelty:** upstream [llama.cpp #30738](https://github.com/ggml-org/llama.cpp/pull/30738), opened 2026-10-17 23:18 UTC, is open at head `b1b775acaa372053417b53361c5e945f0d6b1aea` (checked 2026-10-18). Its three-file diff adds a Q4_0 `DEQUANT_TRANSPOSE` shader variant, one generator registration and one Vulkan pipeline registration. No local patch exists or is authorized. This is not an independent RAD06 or recent BCOP kernel slice.

## Exact pinned source, route, lifetime

- Current BigCherry llama.cpp b11552 (`23b0202a189c44a54625aadcb37a946dd1d6278d`): `ggml/src/ggml-vulkan/ggml-vulkan.cpp::ggml_vk_load_shaders` registers `pipeline_dequant_transpose[GGML_TYPE_Q8_0]` only. `vulkan-shaders-gen.cpp::process_shaders` likewise generates a transpose variant only for `q8_0`. `dequant_q4_0.comp` has a standard 256-thread (4x64) decode but no transposed output path. The proposed PR changes only these seams.
- `ggml_vk_flash_attn` sets `use_dequant_kv` only for quantized K **and** V, **Q query rows `neq1 >= 64`**, dense K/V layouts, individual F16 scratch buffers within `maxStorageBufferRange`, both per-type transpose pipelines present, `!coopmat2`, and the existing Intel Xe1 exclusion. Thus Q4_0/Q4_0, Q4_0/Q8_0 and Q8_0/Q4_0 can become eligible; F16, mixed F16/Q4, non-dense, coopmat2, or n=1..63 cannot. **The shader cannot explain the upstream PR's reported tg128 improvement for ordinary single-query decode.** Treat that result as an unattributed external A/B observation until a per-dispatch trace proves otherwise; do not advertise a decode win.
- When selected, `ggml_vk_flash_attn` sizes `ctx->prealloc_x` for **both** K and V full F16 tensors, reuses the Vulkan context-owned scratch, dispatches one transpose for K and one for V, synchronizes before FA consumes the buffers, rewrites effective strides/types to F16, then marks `prealloc_x_need_sync` for later reuse. No persistent KV representation or host copy changes. Capture two conversion launches, scratch allocation/reallocation, barrier, FA path and actual physical bytes; a one-time activation marker is insufficient.
- Proposed `dequant_q4_0.comp`: workgroup 256; `i=wg.x*4+lane/64`, `ib=32*i+(lane%32)`, `il=(lane%64)/32`; read `block_q4_0.d` and 8 nibble-packed bytes per half, write 8 low and 8 high elements. For each block start `e0=ib*32`, output base is `e0%HS + ((e0/(HS*NH))%KV)*HS + ((e0/HS)%NH)*(HS*KV) + (e0/(HS*NH*KV))*(HS*KV*NH)`. Offsets `8*il+[0..7]` and `8*il+16+[0..7]` reconstruct 32 values. Require `HS%32==0`, bounded `ib<nel/32`, no 32-bit product overflow, exact layout [HS,NH,KV,NS] -> [HS,KV,NH,NS], and F16 rounding parity. Keep stock `dequant_q8_0.comp` untouched.
- Physical traffic ceiling, **theoretical only**, for D=256, 4 KV heads and K+V at KV=80,000: F16 scratch **312.5 MiB** versus Q4_0 cache **87.9 MiB**, i.e. **224.6 MiB additional live scratch** if no existing scratch capacity is reusable. Each eligible invocation also reads Q4 cache, writes full F16 K+V and rereads F16 during FA. At 32K: 128 MiB F16 scratch versus 36 MiB Q4. Do not confuse this with persistent model KV bytes, or sum overlapped GPU kernel times into critical-path wall.

## Cheapest discriminator and observed evidence

**Executed host-only:** eight pinned/PR-source assertions passed (Q8-only baseline, both-type/64-row/coopmat2 gates, scratch/sync, Q4 generator and nibble code); five deterministic Q4_0 nibble/layout fixtures at HS=32/64/128/256, NH=1/3/4/8, KV=1/7/33/64/257 and NS=1/2 had zero duplicate/missing output indices and zero F16-bit mismatches against a separately computed decode+transpose reference. A 90-case predicate model showed 0/18 n=1 cases eligible and 4/18 n=64 cases eligible. These are **host emulations/static checks**, not shader compilation, SPIR-V validation, GPU parity or benchmark. In particular, the emulation does not reproduce Vulkan driver arithmetic/rounding or descriptor bounds.

**External, not BigCherry:** #30738 reports RADV GCN1 (HD 7900, *not* Radeon RX 7900 XTX/gfx1100) Qwen3.5-4B at 32K: Q4_0/Q4_0 pp512 82.46 -> 116.44 t/s; Q8_0/Q4_0 pp512 73.78 -> 116.24 t/s; tg128 13.19 -> 16.60 and 12.36 -> 16.83 t/s respectively, but the latter is not causally attributable to the n>=64 gated shader. PPL was measured only on four 2048-token chunks, not full-model logits or long-context multi-request correctness. No BigCherry Vulkan measurements exist. vLLM ROCm `vllm/v1/attention/backends/rocm_attn.py` uses FP8/paged KV and explicit backend/head-size admission; its quantization format and layout are not a drop-in Q4_0 GGUF shader. SGLang AITER/TileLang are likewise different kernels; reuse capability gating, not performance ratios.

## Conditional implementation/benchmark gate (after Vulkan resume only)

1. **Upstream-first:** check whether #30738 merged into the *new* pinned llama.cpp before any local port. If absorbed, use the upstream implementation; no BigCherry duplicate patch. If open, prototype only its three source edits on a disposable exact-pin overlay, default off in BigCherry recipe; no second route selector, no new allocator or environment knob. Record shader SPIR-V compilation and descriptor-range validation for both gfx1100 and gfx1201, RADV vs AMDVLK/proprietary separately. gfx1030 is an optional negative/portability control, not a qualifying substitute.
2. **Host/validation:** add a reproducible shader-output fixture for Q4_0/Q4_0, Q4_0/Q8_0, Q8_0/Q4_0 and Q8_0/Q8_0; HS 32/64/128/256, NH 1/4/8, KV 1/31/32/33/512/32768, NS 1/2, padded tails, both K and V roles, positive/negative scales, signed zero, extremes, NaN handling and descriptor/overflow bounds. Compare full F16 bit patterns to baseline Q4 dequant+transpose, then backend `FLASH_ATTN_EXT` reference, full-vocabulary logits/KLD/greedy, same-process repeated requests, multi-ubatch, long context, graph/scratch replay and MTP acceptance. Negative cases: Q=1/5/63, non-dense views, coopmat2, F16, mixed F16/Q4, Q8-only stock, non-AMD.
3. **Route/physical accounting:** capture `use_dequant_kv` by invocation, Q/K/V shapes/strides/types, device/ICD/driver, bound pipeline, conversion launches, FA launches, source/cache and F16 scratch bytes, maxStorageBufferRange, prealloc growth, barriers, kernel time, VGPR/LDS/occupancy where available, and *serialized critical-path* wall. If no Q4 path is reached or removable critical-path time <2.9126% of wall (max possible +3% throughput), close `NO_ROUTE`/`BELOW_CEILING` without hardware A/B. Reject if memory headroom cannot fit the scratch or output differs.
4. **Only after those gates:** identical-pin baseline vs single shader candidate, one GPU at a time, Qwen/Flash-Next relevant Q4_0 KV and Q8_0/Q4_0, pp64/512/2048, ctx 8K/32K/80K, ub16/128/512, Q=1/5/63 and Q8_0/Q8_0/F16 controls, optional MTP verify widths 2..5. Four independent sessions/architecture, >=10 paired ABBA rounds, CI95-low >=3% **E2E prefill** throughput, <=1% decode/control regression, no missing attention work, no allocation/replay/quality regression. Reject if effect is only GCN1 or cannot be reproduced on the actual RDNA driver/ICD. Preserve fallback to native quantized FA.
5. **Terminal:** `UPSTREAM_ABSORBED`, `NO_ROUTE`, `NO_HEADROOM`, `NUMERIC_OR_MEMORY_FAIL`, `NO_RDNA_GAIN`, or `QUALIFIED_VK_Q4_TRANSPOSE`. No experiments are queued by this audit.

## References and dependencies

- https://github.com/ggml-org/llama.cpp/pull/30738 (open; actual 3-file diff, 2026-10-18)
- https://github.com/ggml-org/llama.cpp/pull/25494 (earlier Q8_0 fused transpose mechanism)
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/rocm_attn.py (non-equivalent FP8/paged-KV comparison)
- RRVP02 pause; TRVP12 runtime adapter; TRVP13/14/15 replay/route/candidate qualification; PKC04 closed ownership decision; PRVP02 separate CM1 safety. No neighbouring protected plan or patch was edited.

## Change Log

- 2026-11-22T05:25:05.063173+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4573
- 2026-11-22T13:07:03.946002+00:00 (updated-by): Updated: section:notes
- 2026-11-22T13:12:30.797521+00:00 (state-transition): State: pending → deferred
