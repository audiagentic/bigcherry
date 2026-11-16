---
id: VTP43
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-15T21:38:40.989488+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# bf16_fp8 KV cache: bf16 keys, E4M3 values (more -long capacity than bf16, closer to bf16 than fp8)

## Description

Owner request (2026-11-16): a KV cache type with bf16 keys and E4M3 values, `--kv-cache-dtype bf16_fp8`. Keys feed the softmax exponent (q.k errors are amplified), values are averaged linearly by the attention weights, so quantising only V should keep quality close to bf16 while cutting the per-token cost from 4*HD to 3*HD bytes: the qwen27b -long-bf16 cache goes from 624k to about 830k tokens (+33%), between bf16 and fp8 (1.25M, KLD 0.0054 mean on long documents with a wide tail). Generic engine feature (a layout like fp8_v4/k6_v4), offered by the qwen35 architecture at head_dim 256 / gqa 6 where libr4d's h256 rows exist; libr3 (gfx1100 native) must serve it too.

## Steps

1. **Design review 1 (GPT, done):** findings recorded; plan revised to option (b).
2. **Design review 2 (GPT):** the revised plan reviewed with a focused prompt (option b only, lifecycle audit, kernel option bits, libr3 rewrite anchors) before any code; findings resolved and recorded.
3. **ABI, engine, arch (card-free):** `RAD_KVQ_B16F8 = 3` and its layout in `abi/rad_builder.h`; `--kv-cache-dtype bf16_fp8` in config (validation list, help, explain); bringup kv_quant mapping, `kv_quant_ok` refusal and streaming refusal; qwen35_fp8 probe bit at head_dim 256 / gqa 6, declaration (row 3HD, op dtype name), notes; serving v2 contract value + generated docs. Tests: config_test, arch_test (row size, op banding, drafter dtype, refusal), kvbind/mem page sizing.
4. **Writers and attention kernels (libr4d):** rows for the store, fused store, prefill, split prefill, decode and gated decode; resolution tests (no bf16/fp8/ref row matches `bf16_fp8`); selftest host references and rows for each.
5. **libr3 native rewrites** and selftest/perf formats.
6. **Implementation validation:** lane build; card-free suites; selftest correctness on every card; spill/VGPR report per instantiation; core lifecycle tests on the layout (copy, prefix reuse, rollback, checkpoint).
7. **Code review (GPT)** of the engine patch and the libr3 change; findings resolved.
8. **Hardware validation (gpu lease):** KLD against the bf16 cache on the same short and long corpora as fp8's 0.0022 / 0.0054; capacity; decode tok/s at 1/4/8 streams, TTFT 2048 / 60k against bf16 and fp8; VTP42 mixed workload once.
9. **Config and docs:** a `-long-bf16fp8` variant (or the owner's choice), evidence, patch docs, ledger.

## Detailed Solution & Technical Design

**Revision 2 (after GPT design review 1, 2026-11-16, `docs/evidence/2026-11-16-vtp43-bf16-fp8-kv/design-review-1.md`): declare the group as E4M3, like fp8_v4/k6_v4 (option b).**

**Layout (RAD_KVQ_B16F8 = 3), a token's row in a head at head_dim HD:** `[0, 2HD)` K as bf16 (what the bf16 cache stores), `[2HD, 3HD)` V as E4M3 codes (the fp8 cache's convention: saturating at +-448, scale 1.0). 3HD bytes a token and head against 4HD (bf16) and 2HD (fp8). Declared to the core as RAD_F8E4M3 with head_dim 3HD/2, so the core's `2 * head_dim` extent (kvbind.cpp:43) is 3HD bytes and every generic consumer (page sizing, prefix cache, host tier, checkpoints, copy/defrag, rollback) works on raw bytes of the declared group exactly as for fp8_v4. This reuses the packed-layout machinery end to end: bringup maps any non-bf16 name to E4M3 and kv_quant from the name; `kv_quant_ok` refusal; the streaming guard; the arch's `ctx->kv_quant` read under an E4M3 group; the drafter group stays E4M3 (as it does for fp8_v4, declare_dflash2) -- a drafter cache dtype only moves draft acceptance, the trunk verifies. Op dtype name `bf16_fp8` separates every row: no existing bf16/fp8 kv_store, attention or reference row bands on it, and the host reference path (libref ref_attn_paged / ref_kv_store) has no row for it, so a missing kernel is a refusal at resolve, not a misread (GPT blocking 2-3). Why not declare as bf16 (option a): the core and the host reference treat a bf16 group as equal K and V halves and the bf16 writer rows would match; every one of those would need a guard (GPT blocking 1-4).

**Kernels (libr4d, h256 gqa6, the bf16 K and fp8 V paths already exist in the bodies):** a new option bit (name: `O_KB16V8`, K bf16 / V E4M3; bit must be free in both templates and in libr3's native copies) on the h256/gqa6 prefill, split-KV prefill, decode and gated-decode bodies: K is fetched exactly as the KVP=1 (bf16) form does; V is fetched as bytes at byte offset 2HD in a 3HD-byte row and converted E4M3 -> bf16 by `fp8x8_to_16x4w<D>` (the library's conversion; D = bf16 maths, F16 off) into the existing sV16 tile. The group's strides are bytes (E4M3 group), so the K path addresses bf16 by byte offset/2. P.V and Q.K then run in bf16 as in the bf16 cache: exact K, V error only from E4M3. Staging V widened in LDS keeps the bf16 tile (P_TILE 48 fits; bf16 K+V at tile 64 exceeds 64 KiB) -- the alternative O_V8-style code tile (half the LDS) is a measured option if a body spills (decode GPREV is already gated off when VREGS > 3). Each new instantiation's VGPR/scratch is checked (`-Rpass-analysis=kernel-resource-usage`) before use.

**Writer:** `kv_store_b16f8` and the fused `qk_norm_rope_gate_kv_store_b16f8` rows: K stored as bf16 as the bf16 writer does, V converted to E4M3 as the fp8 writer does (kv_store_q4's structure in r4d_model_bf16.hip: a variant with `kq==3`). Rows band on `kv_dtype = bf16_fp8`.

**E4M3 widening:** use the library's `fp8x8_to_16x4w<D>` (handles subnormals and the E4M3FN NaN pattern as the fp8 cache does); selftest rows cover +-0, subnormals, normal endpoints, +-448 saturation and both NaN encodings against the existing fp8 reference.

**libr3 (native gfx1100):** the native rewrites that replace K/V staging and the split-store (`native/r4d_attn_prefill_h256_gqa6.hip.rw`, `r4d_attn_paged_h256_gqa6.hip.rw`) are checked against the patched templates; the decode path libr3 builds from libr4d via r3_compat.h is covered by r3_selftest rows; selftest `--case`, `--perfpaged` gain the b16f8 format.

**Out of scope / refused:** other geometries and architectures (kv_quant_ok bit clear), KV streaming of the layout, QSA/indexer layers (gqa8 family), calibrated V descales.

## Code Samples & Guidance



## Files

engine patch (radNNNN): abi/rad_builder.h, core/config.cpp, core/rad_core.h (help text only), core/engine_bringup.cpp, core/kvstream/bringup.cpp, arch/qwen35_fp8/qwen35_fp8.cpp, libr4d/r4d_attn_{prefill,decode,paged}_h256_gqa6.hip, libr4d/r4d_model_bf16.hip (kv_store), libr4d/r4d_qk_norm_rope_gate.hip (fused store), libr4d/r4d_rows.cpp, libr4d/r4d_selftest.cpp, tests/{config,arch,mem}_test.cpp. libr3: native/*.rw for the h256 attention units, selftest formats. Serving contract serving-v2.toml (kv_cache_dtype) + generated docs. engines/radiance/configs/qwen27b.toml.

## Validation

GPT design reviews 1 and 2 before implementation and GPT code review after; card-free engine tests (config, arch, bringup refusals, streaming refusal, row resolution, page sizing, layout lifecycle); libr4d/libr3 selftest correctness for the new rows on every card; no spill in the new instantiations; KLD against bf16 (short and long corpora); capacity and speed against bf16 and fp8; VTP42 mixed workload once.

## Effort & Risk

L. Risks: the h256 bf16 kernels sit near the 256-VGPR bound (O_VF8 conversion may spill: measure per instantiation, fall back to staging V as codes in LDS like O_V8); libr3 native rewrites must keep matching the patched templates; V outliers above 448 saturate (the fp8 cache has the same limit; KLD will show it).

## Standards



## Acceptance Criteria

- [x] GPT design review 1 done; plan revised to option (b).
- [x] GPT design review 2 done and findings resolved before implementation.
- [ ] `--kv-cache-dtype bf16_fp8` serves qwen27b at -long with 3HD bytes a token and head (capacity >= 1.3x bf16), refused clearly on other geometries, architectures and with KV streaming.
- [ ] No bf16, fp8 or reference row resolves for `bf16_fp8` (test); a missing kernel is a refusal.
- [x] Selftest rows for the new attention and store kernels pass on all cards (gfx1201, gfx1100), including E4M3 edge values (bf16_fp8) and zero/floor/cap scale groups (bf16_i8); no spill in the new gfx1201 instantiations.
- [x] KLD against bf16 well below fp8's on the long corpus (target <= half of 0.0054 mean, no p99.9 outliers like fp8's), or the result is recorded and the owner decides. RECORDED, owner decides: bf16_i8 0.00269 mean / p99.9 0.186 against fp8 0.00495 / 0.591 (new reference); docs 00/02 p99.9 stay 1.9/3.3 and bf16 itself across builds shows 1.9/5.2, so the per-document < 1 bar is not reachable on this corpus (owner accepted 2026-11-16: bf16_i8 ships opt-in on its mean KLD, top-1 98.67% and PPL ratio 0.9996; an f32 cache was rejected as pointless since K/V are bf16 already; a bf16-reordering floor test is optional follow-up).
- [x] Decode and prefill speed no worse than fp8's (long argv: 1/4/8 streams and TTFT within 3% for both caches).
- [ ] Serving enum and generated docs updated; GPT code review done and findings resolved; PR merged with evidence.

## Notes

**2026-11-16 GPT design reviews.** Review 1 (revision 1, bf16-declared group): 5 blocking findings, all about a bf16 group being read as equal K/V halves and row collisions -> plan switched to E4M3-declared (option b). Review 2 (revision 2): row collisions and core sizing resolved; found name lists to edit (config, bringup, streaming, arch), stride units (bytes vs uint16 K path), V geometry must not follow KVP, descales, libr3 anchors, acceptance numbers. All resolved in revision 3 (this plan): `docs/evidence/2026-11-16-vtp43-bf16-fp8-kv/design-review-{1,2}.md`. Drafter verified: its group keeps dtype name fp8_e4m3 (rad_arch.h:1741), as for fp8_v4. Acceptance numbers set: long-corpus KLD mean <= 0.0027 and no document p99.9 above 1; decode/TTFT within 3% of fp8; capacity >= 1.3x bf16's 624,051 tokens. Implementation starts now (steps 3-6).

**2026-11-16 results.** bf16_fp8 (rad0037) missed the long-corpus bar (0.00336 mean; docs 00/02 p99.9 3.3/4.0; 79% of the mass in 277 exact-copy tokens). Thinking E4M3 V precision was the cause, bf16_i8 (rad0038: int8 V, f16 scale per 32 dims, 784 B, 1.31x bf16) was built and gave 0.00309 against the same reference. The missing control (plain bf16 against the same older-build reference: 0.00315, same tail) showed the reference itself carried a cross-build floor; re-recorded with the current build, bf16 vs bf16 = 0 and bf16_i8 0.00269, bf16_fp8 0.00323, fp8 0.00495. bf16_i8 dominates bf16_fp8 (similar capacity, better quality, same speed); both patches are `untested`, so they sit in the `kv-bf16-packed` experiment (recipes.toml), not the production source; no `-long-bf16i8` config ships until they are promoted (the evidence README has the measured numbers for one). Details: `docs/evidence/2026-11-16-vtp43-bf16-fp8-kv/README.md`.

## Change Log

- 2026-11-15T21:38:40.989488+00:00 (created-by): Created by agent
- 2026-11-15T21:39:40.921837+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-15T22:08:47.016067+00:00 (updated-by): Updated: section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- 2026-11-15T22:10:57.429764+00:00 (updated-by): Updated: section:notes
