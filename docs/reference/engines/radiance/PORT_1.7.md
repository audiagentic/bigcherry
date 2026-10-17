# MEN12 analysis: porting BigCherry's radiance patches and components from v1.3.0 to v1.7.0

## Status (MEN12 part 1, applied)

Applied by the part-1 change: pin 48f7b8e (the v1.7.0 release commit `1e968f6` plus two NOTICE-only commits; the source
tree is identical to the tag), rad0001 superseded, rad0003 re-anchored (+ explicit `<cstdlib>`), libr3 / libr3moe / helper
prepared for ABI 17 with `R3_NATIVE=OFF` (forwarding `rad_kernel_collective` clamped to 1, dead `RAD_W4_GROUP` replace
removed, gdn-chunk-scan native shelved to `native-shelved/`), test fixtures regenerated at `tests/fixtures/48f7b8e`.
"Verified" below means a text-level check; no build was run.

## Status (MEN12 part 2, applied)

rad0002 is re-authored on 1.7.0 as thin hooks (section 3, with the deviations below) and the shadow compare / mask-only
diagnostics are split into `rad0004_moe_remote_diagnostics` (diagnostic build only: macro `BC_MOE_REMOTE_DIAG`, source
`radiance-diag`, build `moe-diag`; patch-lint fails if it enters any recipe not named `*-diag`). rad0002 targets 1.7.0
only. The core surface is **14** one-line call sites in 4 upstream files (was 20 edits, about 170 inserted lines; the
section 3 estimate was 15): moe_fp8.h 6 (include, member, declare, after_route, merge, `issues_ar`), hc.h 2 (include,
`takes_ar`), planner.cpp 4 (include, `pin_mapped`, `weight_footprint`, `rank_units`), engine_bringup.cpp 2 (include,
placeholder). rad0004 adds 4 `#ifdef BC_MOE_REMOTE_DIAG` sites (include, member, declare, after_down). Deviations from
section 3: the planner/bring-up helpers are header-only templates in `core/place/bc_moe_remote_place.h` (the name
`bc_moe_remote.h` is the C ABI in `abi/`); a production build answers "no remote rows" when shadow or mask-only is
requested; the libr3moe plugin registers once through `bc_moe_remote_register(BcMoeRemotePlugin{struct_size, layer_map,
owns_expert})` in place of the two `dlsym` hooks and mirrors the struct locally (a test pins the two copies equal); the
step gate is decided on `T` everywhere, including `issues_ar()` and HyperConn.

Still open: native re-ports (section 2c, attention first) and everything that needs a Brutus build or GPU run (the
verification list is in the MEN12 hand-off).

Inputs: BigCherry working tree, upstream partial clone with tags v1.3.0 (89cee7c) .. v1.7.0 (tag resolves to the release
commit `1e968f6`; pin 48f7b8e, same tree).

Method note: I extracted `abi/ libr4d/ core/ arch/` at both tags and (a) diffed the ABI headers with comments stripped,
(b) ran every patch anchor (the real `Edit.anchor` regexes) against both trees, (c) replayed `rewrites.txt` + `native/*.rw`
exactly the way `r3_rewrite.py` applies them against both trees. Nothing was compiled (no ROCm here), so every "compiles"
statement below is a text-level finding, not a build result.

Upstream history caveat: since 1.5.0 upstream publishes each release as ONE squashed commit (`radiance 1.4.0` dc339b4,
`1.5.0` b93b531, `1.6.0` b068190, `1.7.0` 1e968f6). Individual feature commits only exist for the Gemma 4 / ParoQuant
merge range. So "the upstream commit" for most items is the release commit.

---------------------------------------------------------------------------------------------------------------------
## 1. Per-patch, per-edit analysis

### 1.1 rad0001_drain_mover_before_arena_free (1 edit) -> FIXED UPSTREAM, retire the patch

- Edit `drain-mover-before-shutdown`: file `core/engine.cpp`, anchor `^    for \(auto& r : ranks_\) r->ctx\.shutdown\(\);$`
  in `Engine::~Engine()` (v1.3.0), replaced by a loop that does `rad_dev_set(r->device)`, `rad_stream_sync(mover.transfer_stream())`,
  then `ctx.shutdown()`.
- v1.7.0 `core/engine.cpp` lines 77-84 (`Engine::~Engine`):
  ```cpp
  /* Drain the transfer streams before the contexts. Ctx::shutdown drains the compute streams
   * and frees the arena, but the mover's stream also copies into the arena (the prefill
   * stager's ring, lent slab slots) and would otherwise drain only later, with the mover. A fill
   * landing in a freed arena is a GPU memory fault. */
  for (auto& r : ranks_)
      if (RadStream s = r->mover.transfer_stream()) (void)rad_stream_sync(s);
  for (auto& r : ranks_) r->ctx.shutdown();
  ```
- Same mechanism, same diagnosis (stager ring + lent slab slots). Differences: upstream does not call `rad_dev_set` first and
  drains in a separate loop before all `shutdown()`s. The patch's own anchor still matches exactly once in v1.7.0 (the
  shutdown line is unchanged), so a naive rebase would apply and DOUBLE the drain; it must be removed, not re-anchored.
- Introduced in `1e968f6` (radiance 1.7.0). Verified per tag: absent in v1.3.0/1.4.0/1.5.0/1.6.0, present in v1.7.0. The
  1.7.0 release notes list it only under "many fixes".
- Action (done): rad0001 is `superseded` and no longer named by any patch set in `engines/radiance/recipes.toml`. Open question (not answerable statically): upstream skips `rad_dev_set(r->device)`;
  `hipStreamSynchronize` on a stream is device-independent in HIP so it should be fine, and the Flash-Next two-XTX
  3/3-clean-exit check should be re-run once on the new pin to confirm.

### 1.2 rad0003_mtp_full_head_switch (1 edit) -> still applies; keep, tiny risk

- Edit `mtp-full-head-switch`: file `arch/common/rad_block_mtp_fp8.h`, anchor `^[ \t]*if \(!head \|\| !lm\) return false;$` inside
  `mtp_draft_head_q2`; inserts `if (getenv("RAD_MTP_FULL_HEAD") != 0) return false;`.
- v1.7.0 same file, line 106-111:
  ```cpp
  inline bool mtp_draft_head_q2(RadBuilder* b, const char* head, const char* lm) {
      if (!head || !lm) return false;
      if (map_copy(b, head, lm) < 0) return false;
  ```
  Anchor matches exactly once; semantics unchanged.
- Upstream changes to this header (dc339b4 1.4.0, 1e968f6 1.7.0): the layer's `mlp.declare` / `op_norm` declare order was
  swapped, the draft-out early return now calls `attn.store_kv`, and `mlp.step_rows(c, L)` replaced `mlp.step`. None touch
  the patched function.
- One real compile risk: upstream REMOVED `#include <cstdlib>` from this header (diff v1.3.0..v1.7.0). The patch uses
  `std::getenv`/`std::atoi`. `<cstdlib>` still arrives transitively (`arch/common/rad_arch.h:24`, `rad_block_moe_fp8.h:54`),
  and the header includes `rad_block_attn_gated_fp8.h`/`rad_block_mlp_fp8.h`, so it should compile, but the patch should add
  its own `#include <cstdlib>` (one extra line in the same edit or a second anchored edit) rather than rely on that.
- No upstream equivalent of a "use trunk lm_head for MTP drafts" switch exists: `grep` for `RAD_MTP`/`full_head`/`draft_head`
  in v1.7.0 core/arch/docs finds only the `dflash.draft_head` weight name (docs/ARCHITECTURES.md:485) and a rope comment.
  Evidence in SUMMARY says it did not explain the acceptance gap; consider leaving it `untested` (no new work).

### 1.3 rad0002_moe_remote_handoff (2 created files + 20 core edits in 4 files)

Created files (no anchor problem; `create=True`):
- `abi/rad_moe_remote.h` (25 lines): C ABI of the policy gate. Not affected by ABI 17 (it only uses stdint types).
  Note: it lives in `abi/` but is not part of upstream's `rad_abi.h` family, so nothing upstream touches it.
- `core/place/moe_remote.cpp` (108 lines): env gates `R3_MOE_REMOTE_MIN/MAX_ROWS`, `R3_MOE_HELPER_SHADOW/MASK_ONLY/SKIP_OWNED`,
  layer-map store/replay, owner hook, `rad_moe_remote_skip_unit`. Compiled because `core/CMakeLists.txt` GLOB_RECURSEs
  `place/*.cpp` (v1.7.0 `core/CMakeLists.txt:8-16`, `RAD_CORE_DIRS` contains `place`; a NEW directory such as `core/bc/`
  would NOT be picked up without a CMake edit). Arch plugins bind to these symbols at dlopen because the `radiance` binary
  is `ENABLE_EXPORTS ON` (`core/CMakeLists.txt:83-87`); unchanged from 1.3.0.

Core edits. "Anchor" = does the patch's regex still match exactly once on v1.7.0; "Semantics" = whether the surrounding code
still supports the edit.

File `arch/common/rad_block_moe_fp8.h` (9 edits; v1.7.0 file is 1661 lines, heavily rewritten, MoeFP8 gained decode expert prefetch,
a fused router and an overlapped all-reduce):

| # | Edit id | What it does | v1.3.0 anchor | v1.7.0 anchor/semantics |
|---|---|---|---|---|
| 1 | include-remote-moe-abi | `#include "rad_moe_remote.h"` | `^#include "rad_device\.h"$` | matches (1). OK |
| 2 | remote-ops | adds 5 `rad_op` members for mailbox ops | `rad_op op_ar = 0, op_add = 0;` | matches at line 379. OK |
| 3 | publish-runtime-layer-map | after `reg[]` is final, builds `remote_slot[]` and calls `rad_moe_remote_layer_map` (+ shadow print) | `if (std::find(prot.begin(), prot.end(), e) == prot.end()) reg.push_back(e);` | matches at line 747, `n_reg`, `prot`, `reg`, `ne`, `c.ff_hi4` still in scope. OK |
| 4 | declare-remote-mailbox-ops | declares `moe_mailbox_publish/add/shadow/mask_ref` ops, errors if absent (~50 lines) | `\{w\.x, w\.h\.x\}, \{w\.x\}\);` (end of `op_add` declaration) | matches (line 1347). Placement now sits just BEFORE new `xp_parts` block (lines 1351-1370). Upstream added `rad_op_declared(b, op)` (1.5.0, b93b531, `abi/rad_builder.h`) which is the right way to decide "is the mailbox plugin loaded" instead of a failing `RAD_OP`. OK mechanically |
| 5 | remote-pass-state | `remote_policy/remote_shadow/remote_active/remote_mask_only` consts in `pass()` | `const bool shared = op_sgate != 0;` | matches (line 1395). `pass()` signature gained `bool own_ar` |
| 6 | publish-remote-and-rescatter | publishes routes (ids masked in place by the publish op), re-scatters | `const int64_t n4[4] = {` (insert_before) | matches (line 1453). Context changed: the router may now be the fused `router_gemm_topk_scatter` (`rfused`) and `rad_expert_prefetch_step` follows n4. `op_scatter` is still unconditionally declared (line 696) so the re-scatter works. `ecnt` is now a local `RadOperand` (`praw(counts)` straight into the core's per-layer row, line 1411) and is used in the patch's re-scatter call: still in scope, but the patch text must keep using it so the heat engine sees masked counts |
| 7 | shadow-compare-remote-down | mask-only capture/publish/restore + shadow compare, before shared down | `if (shared) sh_down\.step\(c_, w\.sff, w\.sout, T, r0, rows\);` | matches. OK |
| 8 | force-plain-gather-for-remote | `remote_active ? moe_mailbox_add : (!gather_taken ? moe_gather)` | `if (!gather_taken(g, T, gather_out_rows, gather_out_rows6))` | matches (line 1563). `gfold` is still a member (line 349/1211). OK |
| 9 | add-remote-before-all-reduce | forces the standalone all-reduce when remote is active, even though the consumer HC would fuse it | `if (op_ar && !ar_taken(g, ar_out_take != kArOutNorm ? T : rows, ar_out, ar_out_take))` | **ANCHOR FAILS (0 matches).** v1.7.0 line 1565: `if (own_ar && issues_ar(T, rows)) issue_ar(c_, r0, rows, false);` with `issues_ar()` (line 411) = `op_ar && !ar_taken(...)` and `issue_ar()` (line 416) choosing the overlap-declared collective `op_ar_ov`. The pipelined-prefill callers (`arch/qwen4exp_fp8/qwen4exp_fp8.cpp:1380-1381,1506`, `qwen35_fp8`) call `mlp.issues_ar(T,R)` / `mlp.issue_ar(...)` themselves. The edit must be re-derived at `issues_ar()` rather than at the call |

File `arch/common/rad_block_hc.h` (4 edits; all four anchors still match once):
- include `rad_fp8.h` anchor (OK), `if (!op_write) return;` (line 372, `HyperConn::write`), `if (takes_gather(T)) {` (373),
  `RAD_ISSUE_N(c_, takes_ar(T) ? ar_write(T) : op_write, rows,` (387). Semantics unchanged: `write()` still consults
  `takes_gather(T)`/`takes_ar(T)`. These two are in-class members (`rad_block_hc.h:146,150`) used only by `write()`; a single
  change to `takes_ar()` would cover both runtime branches (see section 3).

File `core/place/planner.cpp` (4 edits):
| Edit id | v1.3.0 anchor | v1.7.0 |
|---|---|---|
| include-remote-moe-abi-planner | `^#include "planner\.h"$` | matches |
| remote-owned-unit-unplaced | `^[ \t]*bool host_only = false, direct = in_\.container;$` (in `Planner::pin_mapped`'s per-unit loop) | **ANCHOR FAILS (0 matches).** Renamed: v1.7.0 `pin_mapped` (planner.cpp:266-) has `bool host_only = false, stored = true;` (line 270); the loop now computes `stored` per weight via `weight_file_direct(...)`. The inserted block (set Tier::Mapped, movable=false, Site::DeviceZeroCopy, `continue`) is still valid as a first statement of the loop body |
| remote-owned-unit-footprint | `for \(U& mu : units\) mu\.mapped = mapped_unit\(mu\.mapped, container && mu\.direct, host_backend\);` | matches (line 201, `weight_footprint`). `U` still has `layer/expert/mapped` |
| rank-units-skip-preplaced | `for (int32_t i = 0; i < (int32_t)p->units.size(); ++i) order->push_back(i);` in `Planner::rank_units` | matches (line 441). Still required: `assign_expert_tiered`'s `fill` lambda (planner.cpp ~line 660-690) sets `mu.tier = VRAM` for EVERY `RAD_ACCESS_CONDITIONAL` unit in `order`, with no `preplaced()` check, so an un-skipped Mapped placeholder would be overwritten. `MoveUnit::preplaced()` (`planner.h:87`) is `tier == Mapped || host_forced`; upstream only uses it in the passes at planner.cpp:527,551,556,609,628,649,1106 |

File `core/engine_bringup.cpp` (3 edits; all anchors match once):
- include `engine_priv.h` (OK); `if ((w.tier == Tier::Mapped || w.tier == Tier::SSD) && (!direct || !e)) {` (now line 2386); `if (w.tier == Tier::Mapped) {`
  (line 2399). Surrounding code is the same shape (byte-count check `n != w.stored_bytes`, `file_->data(p0)`, `r->mover.publish_mapped`).
  The patch's `remote_unit` block still has the identifiers it needs (`file_`, `w.sel`, `r->mover`, `i`, `cfg_.max_tok`).

Summary: 18 of 20 core anchors still match; 2 fail (`add-remote-before-all-reduce`, `remote-owned-unit-unplaced`), and
a third (publish-remote-and-rescatter) is context-fragile. Anchors matching is NOT the same as correct: see risks in section 5.

Upstream equivalent of rad0002? **None found.** `grep -ril "remote\|helper"` over v1.7.0 `core/` hits only unrelated words;
there is no cross-process/cross-device expert compute, no mailbox, no "owned expert" concept. The closest thing is the
tiering / prefetch machinery in section 4.

---------------------------------------------------------------------------------------------------------------------
## 2. ABI 15 -> 17: what changed and what it hits in our components

Releases: ABI 15 in 1.3.0/1.4.0/1.5.0; ABI 16 in 1.6.0 (b068190: "all_reduce takes an optional `overlap`", image ops);
ABI 17 in 1.7.0 (1e968f6: "plugin ABI 17 and changed op schemas: out-of-tree plugins must be rebuilt").
`RAD_ABI_VERSION` is compared for equality by the loader and by our helper (`moe_remote_gpu.h:1256`), so everything we ship
(libr3, libr3moe, r3_moe_helper, r3_ngram_helper if it checks) must be rebuilt against the 1.7.0 headers.
`RAD_FORMAT_VER` (container format) is still 2 (`abi/rad_format.h:29`), so existing `.rad` containers should keep loading;
I found no release-note statement requiring re-conversion except ParoQuant containers made with PR builds.

Header-by-header (comment-stripped diff; headers with no code change: `rad_device.h`, `rad_quant.h`, `rad_sample.h`):

| Header | Code change v1.3.0 -> v1.7.0 | Hits us? |
|---|---|---|
| `rad_abi.h` | `RAD_ABI_VERSION 15u -> 17u`. **Removed** `typedef RadDescribeFn` and the **`RadKernelInfo::describe`** member (last field). Added `RAD_FILL_SAMPLE_PARAMS`, `RAD_OPD_F_IDX_PREFIX`, `RAD_OPD_F_IDX_HISTORY`. Added `RadKernelCollectiveFn` + optional plugin symbol `rad_kernel_collective` and `#define RAD_LINK_ENDED 1` (first present in v1.4.0). | **Yes, see 2a-2c** |
| `rad_types.h` | **Removed** `RadLaunchDesc`, `RAD_MAX_KPARAMS`, `RAD_KPARAM_BLOB` (all existed only for `describe`) | No use found in our sources (`grep` for `RadLaunchDesc`/`describe` over the component: none) |
| `rad_builder.h` | `RadBuildCtx.max_mixed_tok` added; `rad_op_declared()`, `rad_drafter_shared_kv()`, `rad_declare_generator()` + `RadGeneratorDecl`, `RadAudioFormat`/`rad_arch_audio_format`, `RadChatFormat` gained args-form fields, `RadArchProbe` gained `gen_rows/gen_patches` | Only arch-plugin / core-block consumers. Our rad0002 block code uses `RAD_OP`, `rw`, `rad_decl_op`, which are unchanged. `rad_op_declared` is a useful new tool for the mailbox declare (section 3) |
| `rad_runtime.h` | `RadBatch` gained `gen`, `gen_b` at the END plus new `RadGenBatch`; added `RadPrefetchNext` and `rad_expert_prefetch`, `rad_expert_prefetch_mark`, `rad_expert_prefetch_step` (decode next-layer expert prefetch, 1.4.0 dc339b4) | libr3/libr3moe/helper do not include `rad_runtime.h`: no source change. RadBatch layout only matters to code that indexes past `enc_max_seg`: none found |
| `rad_encoding.h` | `rad_enc_givens_parts()`, stricter `givens` validity (`t.part`, `t.scale`, `t.dense`) | Helper includes `rad_encoding.h` (`r3_moe_helper.cpp:3`) and uses `rad_enc_view`; givens only matters for ParoQuant. Recompile only |
| `rad_format.h` | `RadVocabHeader.reserved[4] -> flags + reserved[3]` (same size), `RAD_VOCAB_IGNORE_MERGES` | Helper includes `rad_format.h` (container reader). Size-neutral. Recompile only |
| `rad_plugin.h` | `rad_round_even_sat`; integer `rad_store_f32` now RTNE-saturating (+`RAD_U16` case); `rad_f32_to_fp8e5m2` tie fix. Comment rewrite | Behavioural only; r3_ngram_host/test include it |
| `rad_planes.h` | NEW header (`RadPlanesEntry/Header`, `rad_planes_*`) | Not used by us |
| `rad_philox.h` | NEW (was an empty file) | Not used |
| `rad_yarn.h` | now includable from C; `<cmath>` only under `__cplusplus` | None |

Also changed (not in `abi/`, but part of the plugin contract): op SCHEMAS. Comparing all 106/144 `SCHEMA(...)` tables in
`libr4d/r4d_rows.cpp`: none removed, 38 added (image/Gemma ops), and these changed: `all_reduce` (optional `overlap`),
`gemm_nt_q` (+`pb1`,`pb2` operands, 1.5.0), `attn_dense`, `conv2d`, `rope`, `rope_table`. The loader requires identical
schemas across plugins (1.5.0 note), and libr3 re-exports libr4d's own schema tables (`r3_registry.cpp:41-46`), so libr3 follows
automatically. The four ops our HELPER calls directly on libr4d rows (`moe_gemm_q`, `had_quant_act_fp8`, `gated_had_quant_fp8`,
`moe_gather`; `moe_remote_gpu.h:1281-1284`, params at 720-775/835/997-1060/1355-1358/1570-1650) have **byte-identical param and
operand tables** in 1.3.0 and 1.7.0 (compared programmatically), so the helper's hand-built `RadArgs` stay valid. The kernels
behind them changed internally (`r4d_moe.hip` 237 changed code lines: `R4D_WAVES`, `stage8_keep`, unaligned loads) but the layout
hooks `r4d_rad_layout_w4*` the helper depends on only gained a 5-bit variant.

### 2a. `RadKernelInfo` loses `describe` (struct layout change)
- Hits: every place that builds a `RadKernelInfo`: `helper/moe_mailbox_ops.hip:674-735` (4 rows: kPublish, kAdd, kMaskRef, kShadow),
  `r3_ngram_host.cpp:520-540` (`r3_ngram_row`). Both use value-init `RadKernelInfo x{}` and assign named fields, never `describe`,
  so they compile unchanged; they just need the 1.7.0 header. `r3_registry.cpp` passes through libr4d's rows by pointer.
  Consumers of `RadKernelInfo*` across a dlopen boundary (`moe_remote_gpu.h:1249-1284` iterates the libr4d plugin's rows via
  `rad_kernel_at`) are fine as long as both sides are built against the same header (guaranteed by the ABI equality check).

### 2b. NEW optional symbol `rad_kernel_collective` (this is a real functional hazard, not just recompile)
- v1.7.0 `libr4d/r4d_rows.cpp:6573` exports `rad_kernel_collective(int i)` (declared collective rows return 1; the rows that
  release the P2P link themselves via `link_gate`/`r4d_ar_link_landed` return 2 = `RAD_LINK_ENDED` promise). Absent that symbol the
  engine treats only the conventional `all_reduce` op as a collective (`rad_abi.h` doc on `RadKernelCollectiveFn`).
- libr3 compiles `r4d_rows.cpp` with a `COMPILE_DEFINITIONS` rename list (`CMakeLists.txt:81-82`) that renames
  `rad_plugin_info, rad_kernel_schema_*, rad_kernel_count/at/concurrent` to `r4d_up_*`. `rad_kernel_collective` is NOT in the list,
  so libr3 would export upstream's version, which indexes the UPSTREAM table (inside r4d_rows.cpp `rad_kernel_at` is the renamed
  `r4d_up_kernel_at`) while the engine indexes libr3's FILTERED live table (`r3_registry.cpp:25-50` drops stub rows and appends
  the n-gram row). Index mismatch => wrong rows flagged as collectives.
- Required component change: add `rad_kernel_collective=r4d_up_kernel_collective` to the define list (CMakeLists.txt:82), declare
  `r4d_up_kernel_collective` in `r3_registry.cpp`, and export a forwarding `rad_kernel_collective(i)` that maps live index -> upstream
  index (custom n-gram row -> 0) exactly like `rad_kernel_concurrent` does (`r3_registry.cpp:56-60`).
- Decision to make: clamp the forwarded answer to 1 (never 2). A row that answers 2 must decrement the link gate in-kernel on every
  path; libr3's gfx11 `native/r4d_ar_entry.hip.rw` rewrites large parts of that file and has not been written against
  `r4d_ar_link_landed`, and the header says a wrong 2 "leaves the gate odd, and every yielding DMA list on that card waits
  forever". Answering 1 is always safe ("a path that ignores the word is always safe", rad_abi.h). `libr3moe`'s
  `rad_kernel_concurrent` stub (`moe_mailbox_plugin.cpp:42`) needs no collective export (its rows are not collectives).

### 2c. Rewrites/native blocks vs new libr4d sources (component build breaks at configure time)
`r3_rewrite.py` fails configure on any `.rw` block that does not match exactly. Replaying `rewrites.txt` and all `native/*.rw`
sequentially: **v1.3.0: 0 of 124 blocks fail; v1.7.0: 38 of 124 fail** (rule `s_wait_storecnt` count 8 -> 9, still hits; selftest rule hits 1).
Per file (failing block numbers in the `.rw`, counted from 1):
- `r4d_ar_entry.hip.rw` 11 blocks: #2 (`int  host       = 0;`), #7 (`... cfg->drain, cfg->acq, wire,` -> the new `link_gate`/overlap args)
- `r4d_attn_paged_h256_gqa6.hip.rw` 2 blocks: none fail
- `r4d_attn_prefill_h256_gqa6.hip.rw` 9 blocks: #3, #5, #6, #7, #8 (this one backs **R3_ATTN_NATIVE, the only `production` native**)
- `r4d_fused_quant_fp8.hip.rw` 9: #6
- `r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip.rw` 19: 16 fail (#3-#18). Backs `R3_GDN_NATIVE`, status `evaluated`/shelved (wrong in serving), so drop the whole file rather than port
- `r4d_gemm_fp8a8.hip.rw` 9: #6
- `r4d_gemm_mxfp4a8_decode.hip.rw` 9: #5
- `r4d_gemm_mxfp4a8_tiled.hip.rw` 16: #1, #9, #11, #12, #13, #14, #16
- `r4d_hc_bf16.hip.rw` 16: #8, #13, #15 (backs `R3_HC_NATIVE`, experimental)
- `r4d_moe.hip.rw` 7: #1, #3 (backs `R3_MOE_NATIVE`, experimental)
- `r4d_selftest.cpp.rw` 17: none fail
(File-to-setting mapping is inferred from names and `component.toml`, not from reading each rewrite.) Because `R3_NATIVE=OFF`
builds every unit through `r3_compat.h` only (CMakeLists.txt:20, option `R3_NATIVE`), the port can bring libr3 up with natives OFF first
and re-port the natives one by one.

Other libr3 build-surface changes:
- `libr4d/CMakeLists.txt` unit list went 66 -> 81 `"r4d_xxx|flags"` lines; the CMake regex (`^  "r4d_[a-z0-9_]+\|`) still matches the format
  (comments between entries are fine). New units: `attn_gen, attn_paged_h128_gqa2, attn_paged_h256_gqa2, attn_paged_window,
  expert_scale_bf16, gelu_mul_quant_i8, gemm_w5a16_nt_m64, gemm_wpa8i8_tiled, gen, givens_dense, givens_rotate, norm_join_bf16,
  pair_argmax_f32, qkv_norm_rope, rmsnorm_quant_i8`. 15 of them are new wmma-using units touching `__builtin_amdgcn_*_gfx12`
  (the `grep` list: attn_gen, attn_paged_window, gemm_w5a16, gemm_wpa8i8_tiled, givens_dense) and may need new `r3_compat.h`
  shims or `exclude.txt` entries (currently "(none at present)"). New `__builtin_amdgcn_*` names vs 1.3.0: only
  `ds_bpermute`, which r3_compat.h already lists; no new `asm volatile` mnemonics. Whether they compile for gfx1100 is unverified.
- `${RAD_W4_GROUP}` no longer appears in libr4d's flags; the `string(REPLACE "\${RAD_W4_GROUP}" ...)` at CMakeLists.txt:55 is now dead.
  By the repo's no-legacy rule it should be removed in the same change.
- `r4d_rows.cpp` also gets compiled with `rad_hip_repeatable` (`cmake/RadAddPlugin.cmake`, 1.4.0): when the compiler is LLVM >= 24 (ROCm 10.1)
  `rad_add_plugin` adds `-mllvm=-amdgpu-enable-vopd=false` to our plugins. v1.7.0 README: source builds take ROCm 10.1 or 7.2
  (v1.3.0: 7.2); a 7.2 toolchain remains valid. Which ROCm Brutus has: not found in the repo files I read.
- `component.toml:12` `built-against` and the "radiance 1.3.0" comment need the new SHA; `engine.toml`/serve/metrics names
  (`radiance:draft_tokens_total`, `radiance:draft_accepted_total`) still exist at `core/server/metrics.cpp:372-374`.
  The `--tp-wire` and other flags used in the `flashnext-2xtx` runtime profile still exist in `core/config.cpp`. Removed upstream
  flags (`--prefix-cache-dir`, `--prefix-cache-disk-mib`, `--prefix-cache-host-mib`) are not used by any profile (grep over `engines/` and `src/`; only fixture text mentions them).

Component files and what they must change:

| File | Lines | Change |
|---|---|---|
| `engines/radiance/components/libr3/CMakeLists.txt` | 81-82 | add `rad_kernel_collective=r4d_up_kernel_collective` rename; line 55 drop dead `RAD_W4_GROUP` replace |
| `r3_registry.cpp` | 8-18, 56-62 | declare `r4d_up_kernel_collective`; export forwarding `rad_kernel_collective` (clamp 2->1) |
| `native/*.rw` (9 files) | see 2c | rewrite 38 failing blocks, or remove file |
| `exclude.txt` | whole | add any new unit that does not compile under `r3_compat.h` |
| `component.toml` | 12, 15 | `built-against` SHA; summary strings mentioning "patch rad0002" if hook names change |
| `helper/moe_mailbox_ops.hip` | 674-735 | recompile only (no `describe`) |
| `helper/moe_remote_gpu.h` | 1249-1284 | recompile only; ABI check uses `RAD_ABI_VERSION`; re-verify row names `moe_gemm_w4a8`/`moe_gather_bf16` still exist (they do in `r4d_rows.cpp` v1.7.0 lines 5592, 5654) |
| `helper/moe_remote_hook.cpp` | 164, 179 | `dlsym(RTLD_DEFAULT, "rad_moe_remote_set_layer_map_hook"/"..._owner_hook")`: change only if the extension API is renamed (section 3) |
| `r3_ngram_host.cpp` | 161-170 | looks up libavx's `embed_lookup_q` by op name; row still exists (`libavx/avx_registry.cpp:776,1162`); recompile |

---------------------------------------------------------------------------------------------------------------------
## 3. Proposed thin-hook design for rad0002 on v1.7.0

Goal (owner rule): our policy in our own files; core edits only as thin call-through hooks.

### 3.1 Count: in-core edit sites, before vs after

Before (v1.3.0-style patch, ported 1:1): rad0002 has **20** Edit sites in 4 core files (+167 inserted core lines; 2 created files,
133 lines). With rad0001 (1 site, 7 lines) and rad0003 (1 site, 4 lines): **22 sites, 178 inserted core lines**.

After (proposed): rad0002 **15** sites (11 if the diagnostic modes and include edits are dropped, see below), roughly **25 inserted core
lines**, every one a one-line call or include; rad0001 is gone (upstream-fixed); rad0003 stays at 1. Total **16 sites**.

| File | Before | After | Sites after |
|---|---|---|---|
| `arch/common/rad_block_moe_fp8.h` | 9 (120 lines) | 7 | include; member; declare; after_route; after_down (optional); merge-gate; issues_ar |
| `arch/common/rad_block_hc.h` | 4 (7 lines) | 2 | include; `takes_ar` |
| `core/place/planner.cpp` | 4 (26 lines) | 4 | include; pin_mapped; footprint; rank_units |
| `core/engine_bringup.cpp` | 3 (14 lines) | 2 | include; placeholder publish |
| `core/engine.cpp` (rad0001) | 1 | 0 | dropped |

Further reduction options: (a) force-include a single header via build flags (`-include bc_moe_remote.h` on `rad_core` and the arch
plugins from our build recipe) removes the 4 include edits (-> 11); needs recipe/CMake flags, not a source edit, so it is obscure to
readers; (b) drop shadow/mask-only (both `status = diagnostic` in component.toml, correctness proven in 1.3.0-era work) -> removes
`after_down` and the mask-only branch (-> 10).

### 3.2 Extension-point files (all ours, no core file modified)

1. `abi/bc_moe_remote.h` (C, replaces `rad_moe_remote.h`): the contract between the engine and the libr3moe plugin. By the no-legacy
   rule rename in one change and update libr3moe in the same commit. Sketch:
   ```c
   typedef struct BcMoeRemotePlugin {          /* filled by libr3moe, registered once */
       uint32_t struct_size;
       void (*layer_map)(int layer, int n_expert, const int32_t* slot_of_expert);   /* today: set_layer_map_hook target */
       int  (*owns_expert)(int layer, int expert);                                    /* today: owner_hook */
   } BcMoeRemotePlugin;
   int  bc_moe_remote_register(const BcMoeRemotePlugin* p);   /* engine export; replaces 2 dlsym lookups in moe_remote_hook.cpp */
   int  bc_moe_remote_wanted(int64_t step_rows);              /* env R3_MOE_REMOTE_MIN/MAX_ROWS */
   int  bc_moe_remote_wanted_upto(int64_t max_step_rows);
   int  bc_moe_remote_shadow(void);  int bc_moe_remote_mask_only(void);
   void bc_moe_remote_layer_map(int layer, int n_expert, const int32_t* slot_of_expert);  /* called from block declare */
   int  bc_moe_remote_skip_unit(int layer, int expert, int64_t max_step_rows);          /* SKIP_OWNED policy */
   ```
2. `core/place/bc_moe_remote.cpp` + `core/place/bc_moe_remote.h` (must be under an existing `RAD_CORE_DIRS` dir, e.g. `place/`, to avoid a
   CMake edit): everything that is currently `moe_remote.cpp`, plus the planner/bring-up helpers that need engine types:
   - `void bc_moe_remote_pin_units(std::vector<MoveUnit>&, int64_t max_tok)` (sets Mapped/!movable/DeviceZeroCopy/reason for each
     skip-unit; exactly today's inserted planner block, hoisted out of the per-unit loop);
   - `template<class U> void bc_moe_remote_mark_footprint(std::vector<U>&)` (`mu.mapped = true` where `skip_unit(.., 0)`);
   - `bool bc_moe_remote_publish_placeholder(int64_t max_tok, WeightInfo& w, const RadFileEntry* e, const RadFile& f, Mover& m, rad_weight i)`
     (today's `remote_unit` publish; needs only `file_->data/plane` and `Mover::publish_mapped`, which the patch already calls).
3. `arch/common/bc_moe_remote_block.h` (header-only, templated on the block type so it can be included BEFORE `struct MoeFP8` is complete):
   ```cpp
   struct BcMoeRemote {                       // member of MoeFP8: `BcMoeRemote bc_remote;`
       rad_op publish=0, add=0, shadow=0, mask_capture=0, mask_restore=0;
       template<class Moe> int  declare(RadBuilder*, const Moe&, const std::vector<int64_t>& reg, int layer); // layer map + 4 ops via rad_op_declared()
       bool active(int64_t T) const;          // ops declared && bc_moe_remote_wanted(T) && !shadow
       template<class Moe> void after_route(RadCtx*, const Moe&, int64_t r0, int64_t rows, const RadOperand& ecnt) const; // publish (+rescatter, mask-only)
       template<class Moe> void after_down (RadCtx*, const Moe&, int64_t r0, int64_t rows) const;                          // mask-only/shadow compare
       template<class Moe> bool merge      (RadCtx*, const Moe&, int64_t r0, int64_t rows) const;                          // moe_mailbox_add instead of moe_gather
   };
   bool bc_moe_remote_hc_open(int64_t moe_top_k, int64_t T);   // for HyperConn: top_k > 0 && wanted(T) && !shadow
   ```
   Using upstream's new `rad_op_declared(b, "moe_mailbox_publish")` makes the "plugin not loaded" case a clean no-op instead of today's
   error path in `declare-remote-mailbox-ops`.

### 3.3 Core call sites (the whole in-core surface)

`arch/common/rad_block_moe_fp8.h`
- S1 top: `#include "bc_moe_remote_block.h"` (replaces edit `include-remote-moe-abi`).
- S2 members (next to `rad_op op_ar = 0, op_add = 0;`, line 379): `BcMoeRemote bc_remote;`
- S3 `declare()` after the `op_add` declaration (line ~1350): `RAD_ARCH_TRY(bc_remote.declare(b, *this, reg, l));`
  (covers edits `remote-ops` ctor state, `publish-runtime-layer-map` and `declare-remote-mailbox-ops`; `reg` and `prot` are in scope there).
- S4 `pass()` right before `const int64_t n4[4]` (line 1453): `bc_remote.after_route(c_, *this, r0, rows, ecnt);` (covers `remote-pass-state`
  and `publish-remote-and-rescatter`; `active()` is recomputed inside, it is a pure function of `rows`).
- S5 `pass()` right before `if (shared) sh_down.step(...)`: `bc_remote.after_down(c_, *this, r0, rows);` (optional; diagnostics).
- S6 gather: `if (!bc_remote.merge(c_, *this, r0, rows) && !gather_taken(g, T, gather_out_rows, gather_out_rows6))`
  (replaces `force-plain-gather-for-remote`).
- S7 `issues_ar()` (line 411): `return op_ar && (bc_remote.active(T) || !ar_taken(...));` Because both `pass()` (`own_ar && issues_ar(T, rows)`)
  and the pipelined callers in `qwen4exp_fp8`/`qwen35_fp8` call it, this one line replaces the failing `add-remote-before-all-reduce`
  anchor and keeps the pipelined path consistent. Note it changes the gating variable from per-pass `rows` (today) to step `T`; they are equal for decode-sized single-pass steps.

`arch/common/rad_block_hc.h`
- H1 include; H2 `HyperConn::takes_ar()` (line 146): `return ar_write(T) != 0 && !bc_moe_remote_hc_open(gsrc.top_k, T);`
  `takes_gather()` is `takes_ar(T) && ...`, and `write()` uses only these two, so this single change replaces the three behaviour edits
  (`remote-hc-write-state`, `remote-disable-hc-gather-ar`, `remote-disable-hc-ar-write`). Only the MoE-source connection has
  `gsrc.top_k > 0`, preserving "attention and delta-net writes keep their normal fused reductions".

`core/place/planner.cpp`
- P1 include `bc_moe_remote.h`; P2 `pin_mapped`: first statement `bc_moe_remote_pin_units(p->units, in_.config->max_tok);`
  (the per-unit loop's later `continue` for non-host-only/non-ngram units leaves the pre-set tier untouched, so no change to the loop);
  P3 `weight_footprint` after line 201: `bc_moe_remote_mark_footprint(units);`; P4 `rank_units` lines 438-441: skip preplaced units
  (3 lines; this cannot move out of the core).
  Optional P4 alternative to evaluate: have `bc_moe_remote_pin_units` also set the unit's access class so `assign_expert_tiered`'s fill loops
  (which only touch `RAD_ACCESS_CONDITIONAL`) ignore it. This would drop P4 but changes access-class bookkeeping used by budgeting; I did not verify the knock-on effects, so keep P4 unless that is checked.

`core/engine_bringup.cpp`
- B1 include; B2 inside the weight publish loop, immediately before the `(w.tier == Mapped || SSD) && (!direct || !e)` check (line 2386):
  `if (bc_moe_remote_publish_placeholder(cfg_.max_tok, w, e, *file_, r->mover, (rad_weight)i)) continue;` (one site instead of today's two).

### 3.4 Where each piece of today's in-core policy moves

| Policy piece (today) | New home |
|---|---|
| `R3_MOE_HELPER_SKIP_OWNED` placement exclusion (`rad_moe_remote_skip_unit` and its conditions: every step handed off, no shadow/mask-only, runtime map gives a regular slot, no protected experts in the layer) | `core/place/bc_moe_remote.cpp` (`bc_moe_remote_skip_unit`), unchanged logic; three callers: P2, P3, B2 |
| owner hook (`rad_moe_remote_set_owner_hook`) | `BcMoeRemotePlugin::owns_expert` registered via `bc_moe_remote_register`; stored in `bc_moe_remote.cpp` |
| layer map store + replay-on-late-hook | `bc_moe_remote.cpp`; published from `BcMoeRemote::declare` (S3) |
| preplaced units (`rank_units` skip) | stays a 3-line core edit P4 (cannot be moved), see optional alternative |
| `weight_footprint` exclusion | `bc_moe_remote_mark_footprint` (P3) |
| mask-only routing + shadow compare | `BcMoeRemote::after_route/after_down` in `bc_moe_remote_block.h` (S4/S5), or retire |
| publish placeholder | `bc_moe_remote_publish_placeholder` (B2) |
| HyperConn fused-gather/AR suppression | `bc_moe_remote_hc_open` + `takes_ar` (H2) |
| row policy gate (`R3_MOE_REMOTE_MIN/MAX_ROWS`) | `bc_moe_remote_wanted*` in `bc_moe_remote.cpp` |
| mailbox plugin registration | `moe_remote_hook.cpp` (libr3moe, ours) calls `bc_moe_remote_register` instead of two `dlsym`s |

Behaviour to verify when implementing (not answerable statically): `rad_expert_prefetch*` (new in the same function) must not pick our
Mapped placeholder experts. The plan kernel only considers table entries that fall inside the pinned pool ranges
(`core/runtime/xprefetch.hip:175-176`), and placeholders point into the container mapping, so they should be skipped; confirm at runtime
with `R3_MOE_HELPER_SKIP_OWNED=1` on Flash-Next. The same applies to `core/place/stager.cpp` (prefill stager stages "the pooled experts").

---------------------------------------------------------------------------------------------------------------------
## 4. Upstream changes that could replace or simplify our pieces

(Refs are release squash commits in the partial clone.)

1. **rad0001 shutdown drain: fixed upstream** in `1e968f6` (radiance 1.7.0), `core/engine.cpp` `Engine::~Engine`. Retire the patch.
2. **Decode next-layer expert prefetch + faster heat engine** in `dc339b4` (radiance 1.4.0): "the heat engine moves 64 experts a step (was
   8), its copies and the prefill's expert staging run on the card's DMA engines instead of as copy kernels, and at decode the next
   layer's likeliest experts are copied onto the card ahead of its GEMMs. Against 1.3.0 at --tp 1: prefill +17-20% from 8K tokens up,
   throughput under load +16-20%". Code: `abi/rad_runtime.h` (`rad_expert_prefetch*`), `core/runtime/xprefetch.{h,cpp,hip}`,
   `core/place/heat.{h,cpp}`, `core/place/stager.{h,cpp}`, wired in `MoeFP8::pass` (`xp_next`). This narrows the penalty of host-resident
   experts on a single card (the case our helper targets) but is measured at TP1; no number for TP2 with host pool was found. It does not
   provide compute on another device. It will move the "previous pin" baseline (expect Flash-Next prefill gains independent of our work) and
   interacts with placeholders (see 3.4). New deployment file `deploy/compose/flashnext-tp1.yaml` ("Qwen3.8-Flash-Next on one card and 64 GB of
   host RAM", 4096-token prefill steps, 301K cache) exists in 1.4.0.
3. **Prefill span / busy-prefill-chunk** (1.4.0 `--busy-prefill-chunk`; 1.7.0 opt-in `--prefill-span`, `--prefill-interleave`): changes arena
   sizing and the VRAM left to the expert slab (docs/GUIDE.md ~855-900: "a spanned step costs 320 MiB a card on Flash-Next at TP2"). Relevant to
   the budget our SKIP_OWNED path claims; no code of ours depends on it.
4. **`rad_op_declared(b, op)`** (1.5.0, `b93b531`, `abi/rad_builder.h`): lets block code ask whether an op is in the loaded vocabulary. Use it in
   `BcMoeRemote::declare` to make "mailbox plugin absent" a no-op.
5. **Fused router `router_gemm_topk_scatter`** (rows present in both versions, 7 vs 6 mentions; used in `MoeFP8::pass` as `rfused`): changes the op
   sequence our publish hook sits behind; correctness of masking still relies on `moe_scatter` re-run (still declared, line 696).
6. **Four-rank TP, `rad_kernel_collective`, link early release** (1.4.0/1.7.0): affects libr3's collective rows (2b); no benefit for gfx1100.
7. **libavx covers every op** (1.7.0 note: "`--kernels libavx` alone serves every shipped container"). Possible replacement for our
   `R3_MOE_HELPER_CPU` fp32 host reference and for the n-gram host gather (`R3_NGRAM_AVX`), both diagnostic/shelved; not investigated further.
8. **gfx1100 / RDNA3 kernels in libr4d: not found.** libr4d still builds only for `^gfx12` (`libr4d/CMakeLists.txt:21`). The only RDNA3 mentions
   in v1.7.0 are comments (`r4d_fp8_wmma.h:47-54`), the `cardclock.py` gpu_metrics note and test fixtures. Core HIP code (aql_blit, xprefetch)
   builds for `gfx11-generic` (`core/CMakeLists.txt:65`), so the core runs on gfx1100 as in 1.3.0. libr3 stays necessary.
9. **Remote / heterogeneous expert compute: not found.** No upstream mailbox, helper process, owner map or cross-device expert placement. The planner's
   tiers are VRAM / pinned host / pageable host / SSD / Mapped (`Tier` in `planner.h`) with `Site::{Device,DeviceZeroCopy,DeviceStaged,Host}`.
   `Site::Host` runs a weight's op on a host kernel ("no device kernel for its op"), but only for ops with no device kernel; there is no cold-expert CPU path.
10. **Container popularity profile** (`MoveUnit::profile_share`, `rad-convert --imatrix`, `rad-info --profile`) already existed in 1.3.0 and is unchanged;
    it is an alternative way to supply "coldest experts" to the planner, whereas our `R3_MOE_HELPER_HEAT` consumes a runtime histogram file.
11. **New op `expert_scale`, `pair_argmax`, `norm_add_norm_route` etc.** are Gemma 4 / image-model ops; irrelevant to Flash-Next and Qwen3.8-27B.
12. **New libr4d units for gfx12** (listed in 2c) bring new rows that libr3 must either compile through `r3_compat.h` or list in `exclude.txt` (rows whose unit is
    excluded are dropped by the stub filter). Not needed for our target models.

---------------------------------------------------------------------------------------------------------------------
## 5. Risks and ordered porting plan

### Risks

1. **Silent mismatch, no compile error**: `rad_kernel_collective` index mapping (2b) and the 2-vs-1 link-gate promise. Highest-severity item because it can hang
   collectives on the XTX ranks rather than fail to build.
2. **Native rewrites**: 38/124 `.rw` blocks no longer match; `R3_ATTN_NATIVE` (production, +5% prefill) is one of them. Each re-ported native needs the existing KLD
   and speed evidence repeated (component.toml quotes KLD 0.0035 for attention). Expect the biggest time sink here.
3. **rad0002 semantic drift behind matching anchors**: the new overlapped all-reduce (`op_ar_ov`, `own_ar`), fused router, decode prefetch, new `max_tok`/prefill-span
   arena sizing, and heat engine at 64 moves/step. The all-reduce/HyperConn gating (S7/H2) is the least certain; it needs a decode-token identity check
   (hand-off on vs off) and a TP2 run, and a check that the pipelined prefill path (`qwen4exp_fp8.cpp:1380-1506`) never takes remote steps.
4. **Placeholders vs new placement machinery**: prefetch and stager must ignore Mapped placeholders (statically confirmed for prefetch via pool-range test;
   not confirmed for the stager).
5. **Baseline drift**: 1.4.0 itself reports +16-20% throughput on Flash-Next at TP1. MEN12's acceptance ("within drift or explained") needs those notes cited, and the
   "previous pin" numbers must be measured with the same driver/ROCm.
6. **Toolchain**: 1.7.0 accepts ROCm 10.1 or 7.2; LLVM 24 turns VOPD pairing off for repeatable code. If Brutus changes toolchain with the bump, kernel timing changes
   are confounded with the engine bump.
7. **Test fixtures**: `engines/radiance/tests/fixtures/89cee7c/` hold 1.3.0 copies of 11 core/arch files for `tools/tests/patch/test_radiance_patch_packages.py` and
   `tools/tests/core/test_engine_recipes.py`; they must be regenerated at 48f7b8e (same file list minus `core/engine.cpp` if rad0001 retires; `core/place/planner.cpp`,
   `core/engine_bringup.cpp`, `arch/common/rad_block_{moe_fp8,hc,mtp_fp8}.h` are the ones the new patches touch). Docs that pin 89cee7c: `docs/reference/engines/RADIANCE_PLATFORM_DESIGN.md`,
   `RADIANCE_WIRE_QUALITY_DESIGN.md`, `engines/radiance/README.md`, `tools/lab/radiance/*/README.md`.
8. **Not verifiable here**: any build result, kernel correctness on gfx1100, the helper's behaviour against the 1.7.0 `moe_gemm_w4a8` kernel internals.

### Ordered plan with rough sizes

| Step | Work | Size |
|---|---|---|
| 0 | Decide scope: retire rad0001; keep rad0003; adopt the thin-hook rad0002 design; decide whether shadow/mask-only diagnostics are ported. Fetch 1.7.0 fixtures | 0.5 d |
| 1 | `pin-bump --engine radiance` to 48f7b8e; update `recipes.toml` (`pinned`, descriptions "radiance 1.3.0" -> 1.7.0; remove rad0001 from `engine-fixes-evaluated` and `radiance-lab`), `component.toml` `built-against`; regenerate `tests/fixtures/<sha>/`. Set rad0001 superseded | 0.5-1 d |
| 2 | libr3 minimum port with `R3_NATIVE=OFF`: forwarding `rad_kernel_collective` (clamp to 1), CMake rename, remove dead `RAD_W4_GROUP`, add any new-unit exclusions, rebuild libr3 / libr3moe / r3_moe_helper / ngram helper against ABI 17. Run `r3_selftest` / rad-kbench on gfx1100 | 1-2 d |
| 3 | rad0003: re-anchor check (already matches), add explicit `#include <cstdlib>`, update SUMMARY/patch.toml pin text | 0.25 d |
| 4 | rad0002 rewrite as thin hooks (section 3): new `abi/bc_moe_remote.h`, `core/place/bc_moe_remote.{h,cpp}`, `arch/common/bc_moe_remote_block.h`, 15 core call sites, libr3moe `moe_remote_hook.cpp` switched to `bc_moe_remote_register`; offline apply/idempotence and patch-lint | 2-3 d |
| 5 | Mechanics on GPU: smoke (no hand-off) on Flash-Next and 27B for identity vs the old pin, then `R3_MOE_HELPER` + SKIP_OWNED on Flash-Next TP2 with R9700 helper: greedy identity, repeat-run stress, shutdown cleanliness (this doubles as the check that the upstream drain replaces rad0001) | 1-2 d GPU |
| 6 | Native re-port, in priority order: attention prefill/paged (production, 5 blocks), then ar_entry (if `R3_AR_WIRE4` still wanted), moe/hc/fused_quant/fp8a8/mxfp4 (experimental), skip `gdn_chunk_scan` (shelved) and consider dropping blocks with no current use. Each native re-validated (KLD/speed) per the component.toml evidence | 3-5 d (attention alone ~1 d plus a KLD/prefill run) |
| 7 | MEN12 acceptance: `pin-status --engine radiance --complete`, smoke, Flash-Next layout, 27B two-XTX throughput and KLD vs previous pin; document the 1.4.0 heat/prefetch-driven drift | 1-2 d GPU |

Suggested order of risk reduction: do 1, 2, 3 first (gets a working stock+libr3 pin with rad0003), then 4-5 (rad0002), and leave 6 (natives) for last since libr3
runs correctly without them (just slower); 7 closes MEN12. Total roughly 10-16 working days dominated by step 6 and GPU time.

---------------------------------------------------------------------------------------------------------------------
## Items explicitly "not found" / unverified

- Upstream remote/helper/hetero-device expert compute: not found.
- gfx1100 kernels or an RDNA3 build path for libr4d: not found.
- Per-feature upstream commit hashes for 1.4.0-1.7.0 features: only squashed release commits exist (1.5.0 includes merged feature commits for Gemma 4/ParoQuant).
- Any compile or runtime result: none produced.
- Which ROCm version is installed on Brutus: not found.
- Whether `xprefetch`/`stager` skip Mapped placeholders at runtime: prefetch filter found statically; stager not read in detail.
- Whether new libr4d units compile under `r3_compat.h`: untested.
