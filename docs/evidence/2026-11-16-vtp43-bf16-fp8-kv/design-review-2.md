# VTP43 design review 2 (GPT via Codex, read-only, revised plan and source excerpts supplied in the prompt)

Reviewed plan revision 2 (group declared E4M3 with head_dim 3HD/2). The reviewer could not run shell tools; claims it could not verify are marked in its text. Resolutions are in the table at the end.

## 1. Did revision 2 resolve the first review’s blocking findings?

**Partly. Overall: blocking.** Declaring the group E4M3 fixes the core’s **raw row size**: `kvbind.cpp` uses `2 * head_dim` elements for a full KV group, so declaring `head_dim = 3HD/2` gives a `3HD`-byte row (kvbind.cpp:35–44). It does not make all E4M3 consumers understand the mixed K/V layout.

| First-review finding | Resolved? | Evidence and reason |
|---|---|---|
| Arch dispatch reads `kv_quant` only under an E4M3 group | **Yes, in principle.** | `qwen35_fp8.cpp` reads `ctx->kv_quant` only when `fd.dtype == RAD_F8E4M3` (lines 685–686). But the plan must explicitly add the new name to the `kvq_name` mapping and row-size calculation (currently only `fp8_v4`/`k6_v4`, lines 686–698), and offer its probe bit (currently only V4/K6V4, lines 1881–1889). |
| Core and host reference treat a bf16 group as equal K/V halves | **Yes for raw sizing; no for generic host-reference support.** | E4M3 declaration avoids bf16 element sizing (kvbind.cpp:35–44; mem/kv.cpp:146–165). But the supplied `ref_kv_store_streamed_shape` still describes two halves and a `2*hd` row (ref_shapes.cpp:1615–1625). The plan must ensure all ref paths for this op name refuse or have a dedicated row. |
| A bf16 group would match bf16 writers and `ref_kv_store` | **Yes for row resolution.** | The existing bf16 attention row is constrained to `kv_dtype="bf16"` and fp8 rows to `"fp8_e4m3"` (r4d_rows.cpp:4140–4146, 4166–4168); the new spelling has no matching row. The bf16 store has a bf16 dtype requirement (r4d_model_bf16.hip:2385–2394). |
| Streaming guard and bringup name mapping only know fp8_v4/k6_v4 | **No. Blocking.** | Engine quant mapping and probe refusal explicitly list only those names (engine_bringup.cpp:1052–1054, 1251–1262); streaming’s packed-layout special case does too (kvstream/bringup.cpp:59–67). E4M3 dtype alone does not reject streaming: its dtype gate accepts E4M3 (lines 77–85). Plan says “streaming refusal” but does not spell out adding `bf16_fp8` to these name-based checks. |
| Drafter dtype decision | **Mostly resolved, but plan needs to make it explicit and safe.** | Revision 2 keeps the drafter E4M3, consistent with `declare_dflash2` following the trunk dtype (arch excerpt lines 659–685). But changing the trunk to the new spelling can affect drafter row support: `rad_block_dflash2.h` accepts only `"fp8_e4m3"` or `"bf16"` at its gate (grep excerpt lines 803–804). Add explicit drafter behavior/row coverage or refuse unsupported draft mode. |

**No existing row can resolve `kv_dtype="bf16_fp8"`**, based on the shown constraints: bf16 rows require `"bf16"`; plain fp8 rows require `"fp8_e4m3"`; packed rows require `"fp8_v4"` or `"k6_v4"` (r4d_rows.cpp:4140–4168). The libref excerpts likewise show a generic half-split shape rather than a `bf16_fp8` row (ref_shapes.cpp:1600–1601, 1615–1625). A missing-kernel failure at resolve means the resolver finds no candidate whose operation and constraints match the op’s `kv_dtype` and geometry. That should surface as a no-kernel/unsupported resolution error before execution; it is not itself a safe fallback or an automatic refusal. Ensure bringup treats that error as fatal rather than running with an incompatible row.

## 2. E4M3 consumers that still assume ordinary fp8 layout

The core’s generic raw-byte lifecycle can size, copy, and retain the declared `3HD` row. It cannot interpret the row. The concrete E4M3 assumptions visible in the excerpts are:

- **Architecture declaration:** `qwen35_fp8.cpp` currently derives packed row size only for V4/K6V4 (lines 680–698). Add the B16F8 mapping and `row16` equivalent; otherwise the group is not declared with the planned shape.
- **Standard fp8 store and fused store:** `r4d_rad_kv_store_fp8` converts both K and V using the standard E4M3 two-half store and descales (r4d_model_bf16.hip:2397–2415). New stores must be separate rows for the mixed layout. The excerpts do not include the fused store implementation, so I cannot enumerate its exact offsets.
- **Standard fp8 attention:** ordinary 8-bit V fetch uses `HEAD_DIM` as V’s byte offset, and the standard K path reads the first half as E4M3 (decode excerpt: 333–349, 426–432; prefill has the same layout, with its V fetch beyond the supplied lines). For B16F8, K must be read as bf16 and V starts at byte `2HD`.
- **K descales:** prefill reads `k_descale` and `v_descale` and folds K descale into attention scale (prefill:238–240). Decode reads `k_descale` (decode:202–205). The B16F8 layout specifies exact bf16 K and unit-scale E4M3 V; new rows must not apply ordinary fp8 descales accidentally. `O_QK8`/`O_PV8` are also explicitly 8-bit-cache product paths (prefill:146–165) and must be unavailable for this mixed format unless deliberately implemented.
- **fp8 stream paths:** the stream store supports standard E4M3 or bf16 rows (kvstream/bringup.cpp:77–85; model store comment at 2449 onward), not this mixed layout. The current packed-name warning does not include `bf16_fp8` (kvstream/bringup.cpp:59–67).
- **Checkpoint/host-tier behavior:** the excerpts do not show their KV-specific read/write implementations, so I cannot verify whether they interpret the shape or merely copy bytes. The plan’s claim that these paths work “on raw bytes” needs a code audit, especially checkpoint metadata/restore and host-tier serialization. Same for prefix reuse, rollback, copy, and defrag: raw-copy operations should be layout agnostic, but this is not established by the supplied code.

The new `O_KB16V8` must be independent of `V4`/`K6` and ordinary fp8 selection. Those branches are currently keyed by `KVP`, `O_V4`, and `O_K6` (decode:103–116; prefill:148–165), not by a format enum.

## 3. Kernel-body coherence and required edits

The proposed algorithm is coherent at a high level—bf16 K, E4M3 V widened to the bf16 math tile—but the shown addressing is **not** coherent if the E4M3 group strides are bytes.

- **Stride units:** `a.kv_block_stride` and `a.kv_head_stride` are added to byte pointers in the existing E4M3 fetch (decode:344–349; prefill:322–324). The KVP=1 branch then casts to `uint16_t*` and adds the same `o` as a count of uint16 elements. That doubles the stride/offset. For B16F8, compute a byte offset first, then cast the addressed K pointer (or divide the byte offset by two only after validating even alignment). Do this consistently for the K fetch, direct-K/LDS paths, and any KRAW/grow path.
- **`ROWB`:** existing code selects `2*HEAD_DIM` for ordinary formats, V4 `25*HEAD_DIM/16`, or K6 `22*HEAD_DIM/16` (decode:105–109; prefill:150–154). Add an explicit B16F8 row size of `2*HD` bytes where the kernel’s `HEAD_DIM` remains the logical attention head dimension; do not conflate it with the group’s declared `3HD/2` dimension. Likely this requires a separate byte-pitch constant rather than the current formula.
- **V offset:** ordinary fp8 reads V at `HEAD_DIM` (decode:333–349); B16F8 reads at byte `2*HD`. Add a B16F8-specific offset.
- **`V4`/`K6` constraints:** they are derived from `O_V4` and `!KVP`; `K6` depends on `V4` (decode:103–116; prefill:148–165). B16F8 must not set `O_V4` or flow through V4 scaling/code staging. Add static assertions for mutually exclusive layout bits and for `O_KB16V8` requiring the bf16 math/form (`KVP=1`, F16 off, unless the conversion helper’s actual `D` type requires otherwise). Existing `static_assert(!V4 || F16)` is not a B16F8 assertion.
- **`KVP` reaches more than K decoding:** it controls tile geometry and V staging too: `NTR`, `DG`, `d0`, and whether `storeV` directly widens bf16 or calls `fp8x8_to_16x4w` (decode:129–130, 338–342, 354–384; prefill:180–185). Simply using KVP=1 would make V fetch use the bf16 form’s 32-dimension groups and `storeV` treat V bytes as bf16. Add a separate mixed-format V fetch/stage path (or separate V geometry selector), including its own `NTR`/`DG`/`d0`.
- **16-byte V load geometry:** existing `load_tr_b128` loads 16 bytes. For bf16 V, that covers 8 dims per load and the KVP=1 branch’s geometry is built around 32-dimension groups (decode:338–349). For E4M3 V, 16 bytes cover 16 dims. Define and check the mixed-format lane mapping so each load fetches the intended keys and dims, then widen all 16 values correctly into sV16. The current ordinary fp8 mapping uses `DG=HD/64`, `d0=dg*64+vg*16`, but its 16-byte load is only one lane’s 16-dim segment for one key (`decode:338–349`); verify the new mapping does not inherit the bf16 grouping.
- **Decode K paths:** the excerpt’s `KLDS` K path already distinguishes KVP bf16 from fp8 (decode:415–435), but uses a `uint16_t*` plus element offset. Non-KLDS direct K uses `grow` with a KVP multiplier (decode:444–459), another address path requiring byte-correct mixed handling.
- **Prefill split:** `O_PSPLIT` is a separate instantiation (prefill:166–167; paged dispatch:86–98); include it in B16F8 rows and tests.

Concrete plan edits: define `O_KB16V8` in both templates and reserve a free bit in native libr3 copies; introduce explicit logical-HD, row-byte stride, K-byte offset and V-byte offset helpers; split K and V vector geometry; audit KLDS, direct K, KRAW, V prefetch, split prefill and gated decode; assert legal option combinations; add exact op rows for ordinary, split, decode, and gated cases. The excerpts do not include the decode launcher/row table tail, so I cannot verify which specific gated instantiations currently exist.

## 4. libr3 rewrite anchors at risk

Both supplied `.rw` files anchor text that the new option is likely to change:

- **`native/r4d_attn_prefill_h256_gqa6.hip.rw`**:
  - Lines 56–75 replace the complete `fetchV` lambda, including its `HEAD_DIM + d0` offset and KVP-dependent geometry.
  - Lines 121–159 replace the `storeV` lambda, including the KVP bf16 versus fp8 conversion branch.
  - Lines 19–29 anchor around option declarations/constraints; adding option bits or static assertions near those anchors risks a failed exact match.
- **`native/r4d_attn_paged_h256_gqa6.hip.rw`**:
  - Lines 11–24 and 40–53 anchor the prefill launch expressions; adding a B16F8 option to launch templates changes those match sites.
  - Lines 26–38 anchor the `CAN_SPLIT`/option logic; the new format must be included without accidentally enabling gfx11 native code for unsupported product legs.
- The plan names only those two rewrite files, but also says there are decode and gated-decode B16F8 bodies. The shown decode path is part of the paged unit, while the native prefill rewrite modifies common template code. I cannot verify that the libr3 build has an equivalent decode rewrite anchor or that the plan includes all generated/native source copies. Add rewrite application/match validation for every changed template section and inspect generated libr3 source before implementation.

## 5. Missing steps, acceptance criteria, and risks

**Blocking omissions:**

- Add `bf16_fp8` to config validation, bringup `kv_quant` mapping/refusal, streaming name guard, arch `kvq_name`/row-size mapping, and probe bit. The plan mentions these broadly, but its “any non-bf16 name” wording is insufficient where current code has explicit name lists (config.cpp:679–680; engine_bringup.cpp:1053–1054, 1254–1262; qwen35_fp8.cpp:685–698, 1881–1889; kvstream/bringup.cpp:59–67).
- Specify drafter behavior: the trunk group is E4M3, but `g.kv_dtype` becomes `"bf16_fp8"` and DFlash2’s shown capability gate accepts only `"fp8_e4m3"` or `"bf16"` (grep excerpt lines 803–804). Add matching drafter rows or refuse draft use explicitly.
- Add host reference rows and reference shape/semantic handling for every relevant op, plus a test that no generic fp8 or bf16 reference operation silently runs.
- Add a named, layout-aware checkpoint/host-tier/prefix lifecycle audit. Generic byte copies may be safe; dtype/shape-based transforms may not be. Current excerpts do not prove either.

**Acceptance criteria to tighten:**

- “A missing kernel is a refusal” must name the expected resolver error and assert bringup returns it before allocation or execution; don’t count a resolver miss alone as a clean refusal.
- Test packed page size and actual `3HD` row stride, all address paths (direct K, KLDS, KRAW, V staging), ordinary and split prefill, decode, gated decode, and libr3 rewrite application.
- Make streaming refusal explicit for `bf16_fp8`; test it even though group dtype is E4M3.
- Include draft/MTP behavior and checkpoint save/restore, host-tier transfer, prefix reuse, rollback, copy/defrag in lifecycle criteria.
- Define speed criteria per workload and ensure the target “no worse than fp8” has an agreed measurement tolerance.

**Risks:**

- Main risk is correctness from mixed byte/element addressing and reusing KVP-dependent V geometry; spill is secondary until the indexing is correct.
- The stated capacity is inconsistent: description says about 830k vs 624k (+33%, 1.33×), while acceptance says `>=1.3x`, which is compatible but should specify exact configuration assumptions.
- “No p99.9 outliers like fp8’s” lacks a numeric threshold. Define the metric and threshold.
- The excerpts do not include full decode/gated dispatch, host reference registrations, or checkpoint/host-tier implementation, so those details remain unverified.

## Resolutions (revision 3 of the plan)

| Finding | Resolution |
|---|---|
| Config validation, bringup `kv_quant` mapping and `kv_quant_ok` refusal, streaming name guard, arch `kvq_name`/row size and probe bit list only the two names | Each site is edited explicitly: config.cpp:679-680, engine_bringup.cpp:1053-1054 and 1254-1262, kvstream/bringup.cpp:62, qwen35_fp8.cpp:685-698 and 1888. Tests: config parse, bringup refusal without the probe bit, streaming refusal before allocation. |
| Drafter: DFlash2 gate accepts only "fp8_e4m3" or "bf16" | Verified in source: the drafter group is declared by `drafter_kv_group` (rad_arch.h:1741-1750) with its own `dg.kv_dtype = rad_dtype_name(d.dtype)`, i.e. "fp8_e4m3" under an E4M3 `kv_dtype`; only the trunk's `g.kv_dtype` becomes "bf16_fp8" (qwen35_fp8.cpp:701). The drafter behaves as it does for fp8_v4. arch_test asserts the drafter group and its op banding stay "fp8_e4m3". |
| A resolver miss is not by itself a clean refusal | Acceptance now names the check: with the layout requested and no kernel row, bringup fails with the resolver's no-kernel error before any allocation (test with the host-only libraries, which have no rows for any packed layout: libref/ref_attn.cpp:643, libavx/avx_attn.cpp:1128). |
| Stride units: E4M3 group strides are bytes, the KVP=1 K path indexes uint16 | New rows address K by byte offset: the K pointer is the byte address cast to uint16; every K path (KLDS, direct K, KRAW, split prefill) is audited for it. A separate byte-pitch constant (row = 3*HD bytes, logical HEAD_DIM stays 256) replaces the ROWB formula for the new option. |
| KVP reaches tile geometry and V staging (NTR, DG, d0, storeV) | The new option keeps the K side of the bf16 form (bf16 Q.K, KVP-style K fetch) and takes V geometry from the E4M3 form (DG = HD/64, d0 = dg*64 + vg*16, byte_gather + fp8x8_to_16x4w into sV16); a separate V geometry selector, not KVP, drives V. |
| O_V4/K6, descales, O_QK8/O_PV8 | `O_KB16V8` is independent of O_V4/O_K6 (static_assert on exclusive bits, on F16 off, and on no 8-bit product paths); rows leave k/v descale null (absent means 1.0, decode:202-205, gqa8:241), so no fp8 descale is applied. |
| Host reference shapes (ref_shapes.cpp:1615) | libref/libavx have no row for any packed layout; no row is added for this one. Resolution test: no ref_* row matches kv_dtype "bf16_fp8". |
| Checkpoint, host tier, prefix reuse, rollback, copy/defrag unverified | Audit step before code: read each KV consumer (core/mem/kv.cpp, prefix cache, host tier, checkpoint, rollback) for shape or dtype interpretation; the same consumers already carry fp8_v4/k6_v4 packed rows, so an interpretation would already be wrong there. Lifecycle tests on the layout. |
| libr3 .rw anchors (fetchV, storeV, launch expressions, CAN_SPLIT) | The edit adds new code beside the anchored text rather than inside it where possible; every changed anchor is re-checked by applying the rewrites to the patched source (`r3_rewrite.py` fails on a miss) before the native build; decode comes from libr4d through r3_compat.h with no rewrite. |
| libhalo (second kernel library) has the same family | Out of scope; its rows do not exist for the layout, so selecting it refuses at resolve. |
| Acceptance: numeric KLD threshold, speed tolerance, capacity assumptions | KLD: long-corpus mean <= 0.0027 (half of fp8's 0.0054) and no document with p99.9 above 1 (fp8 reached 4 and 8). Speed: decode tok/s and TTFT within 3% of fp8's or better. Capacity: engine's `kv_cache_max_tokens` for the -long-bf16 layout at the same context and headroom, >= 1.3x bf16's 624,051. |
