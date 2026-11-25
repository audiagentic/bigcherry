# RCS23 stage 1: inventory and baseline of the IQR kernel units

Read-only. No kernel, patch or rewrite changed. Source: the pinned upstream (2.2.0, `b6b8751`), the managed patches and
libr3 native rewrites at this repository's main, the upstream mirror (tags v2.0.0, v2.1.0, v2.2.0 and main `1c104de4`),
and two unit-harness builds of the seven units below (gfx1100 through libr3, gfx1201 through libr4d), each compared with
release 2020.14.0.

Units of the IQR families: the four `r4d_gemm_iqr*` translation units (the three `_p1.._p3` files are five-line wrappers
that set the part number and include the main file, so the same 1,343-line source is compiled four times to split the
form matrix), `r4d_moe_iqr`, `r4d_iqr_embed`, `r4d_logits_vocab`. Their shared inputs: `r4d_iqr.h`, `r4d_iqr_dec.h`,
`r4d_gemm_8a8_prefill.h` (shared with the int8 rows) and the host declaration file `r4d_iqr_layout.cpp`.

## (a) What we already change in each file

| file | lines | libr3 rewrite blocks | lines matched / written by the rewrite | managed patches (edits, lines added) |
|---|---|---|---|---|
| r4d_gemm_iqr.hip | 1343 | 25 | 50 / 271 | rad0065 (2, 116) |
| r4d_gemm_iqr_p1.hip, _p2, _p3 | 5 each | 0 | - | - |
| r4d_moe_iqr.hip | 985 | 0 | - | - |
| r4d_iqr_embed.hip | 142 | 1 | 13 / 40 | - |
| r4d_logits_vocab.hip | 536 | 9 | 25 / 53 | - |
| r4d_iqr.h | 271 | 0 | - | - |
| r4d_iqr_dec.h | 446 | 0 | - | - |
| r4d_iqr_layout.cpp | 971 | 0 | - | - |
| r4d_gemm_8a8_prefill.h | 309 | 8 | 27 / 70 | rad0061 (3, 50), rad0065 (5, 23) |

Layers per unit (from `units` in patch metadata and the shared-unit record): `r4d_gemm_iqr` is edited by rad0065 and the
libr3 rewrite (recorded as shared); every other unit above has one layer or none. The rewrite size is small against the
unit (3.7%, 9.2% and 4.7% of the lines for gemm_iqr, embed and vocab) but the gemm_iqr rewrite writes 271 lines, mostly
the native gfx1100 fused single-band tile and the shared token-tile constant.

## (b) Upstream commits touching each file, per release

| file | v2.0.0..v2.1.0 | v2.1.0..v2.2.0 | v2.2.0..main (119 commits) |
|---|---|---|---|
| r4d_gemm_iqr.hip | 0 | 3 | 0 |
| r4d_gemm_iqr_p1/p2/p3.hip | 0 | 0 | 0 |
| r4d_moe_iqr.hip | 0 | 4 | 0 |
| r4d_iqr_embed.hip | 0 | 0 | 0 |
| r4d_logits_vocab.hip | 0 | 0 | 0 |
| r4d_iqr.h, r4d_iqr_dec.h, r4d_iqr_layout.cpp | 0 | 0 | 0 |
| r4d_gemm_8a8_prefill.h | 0 | 0 | 1 (vision tower reads int8 codes in the container's bytes) |

v2.0.0..v2.1.0 is one commit in the whole repository. In v2.1.0..v2.2.0 (148 commits) the changes are: `r4d_gemm_iqr.hip`
the release commit, the scalar-cache invalidate skip behind chained waits, and keeping short IQR verification on the
decode path; `r4d_moe_iqr.hip` the release commit, a lone spanned prefill's tail snapshot, a packed 128-row run map for
the MoE prefill GEMMs, and the same decode-path change. The IQR files existed unchanged since 2.0.0. Upstream main moves
only the shared 8a8 prefill header.

## (c) Row-table entries and their callbacks (`r4d_rows.cpp`, 468 rows)

| row | launch and scratch (unit) | layout, relayout, unrelayout | other callbacks |
|---|---|---|---|
| gemm_iqr_nt_m64, gemm_iqr_gated | r4d_gemm_iqr (m64, gated) | r4d_iqr_layout.cpp (iqr) | fusion `r4d_fuse_gemm_nt_q_gated_iqr` (gated row), tune validity |
| gemm_iqr_tiled | r4d_gemm_iqr (tiled) | r4d_iqr_layout.cpp (iqr) | `r4d_tune_valid_gemm` |
| moe_gemm_iqr, moe_gemm_iqr_mod4 | r4d_moe_iqr | r4d_iqr_layout.cpp (moe_iqr, mod4) | - |
| embed_lookup_iqr | r4d_iqr_embed | r4d_iqr_layout.cpp (iqr_embed) | `r4d_shape_embed_lookup` |
| logits_gemm_vocab_iqr | r4d_logits_vocab | r4d_iqr_layout.cpp (iqr_vocab, iqr_logits) | sampling (`r4d_sample_stages_f32`, penalties shape) |
| logits_gemm_iqr, logit_rerank_iqr | other units (not these seven) | r4d_iqr_layout.cpp (iqr_logits) | - |

Every row's weight layout and relayout live in `r4d_iqr_layout.cpp`, which no layer edits; the only cross-unit callbacks
are the shape and fusion hooks above and the SIMD relayout objects built from `r4d_hsimd`. On gfx1100 libr3 builds all
seven units (none is on its exclude list), so all nine rows are live there; on gfx1201 they are libr4d's own.

## (d) Build-unit snapshot per architecture (harness builds)

Both builds link only these seven units (rows of the other units are stubbed) and compile at the lane's flags. Against
release 2020.14.0, gfx1201 reports 0 added, 0 changed, 0 removed (3,253 kernels unchanged); gfx1100 reports the five
known IQR prefill forms at 240 VGPR with 2-3 spills that a cold build produces and the shipped objects do not (3,248
unchanged).

| unit | arch | kernels | max VGPR | kernels with VGPR spills | max private bytes | max LDS bytes |
|---|---|---|---|---|---|---|
| r4d_gemm_iqr | gfx1100 | 742 | 256 | 25 | 128 | 51204 |
| r4d_gemm_iqr_p1 | gfx1100 | 799 | 256 | 27 | 120 | 51204 |
| r4d_gemm_iqr_p2 | gfx1100 | 843 | 256 | 33 | 136 | 51204 |
| r4d_gemm_iqr_p3 | gfx1100 | 799 | 256 | 36 | 108 | 51204 |
| r4d_moe_iqr | gfx1100 | 50 | 240 | 13 | 412 | 8720 |
| r4d_iqr_embed | gfx1100 | 17 | 13 | 0 | 0 | 512 |
| r4d_logits_vocab | gfx1100 | 6 | 256 | 2 | 204 | 51200 |
| r4d_gemm_iqr | gfx1201 | 742 | 256 | 66 | 452 | 51204 |
| r4d_gemm_iqr_p1 | gfx1201 | 799 | 256 | 65 | 332 | 51204 |
| r4d_gemm_iqr_p2 | gfx1201 | 843 | 256 | 74 | 472 | 51204 |
| r4d_gemm_iqr_p3 | gfx1201 | 799 | 256 | 67 | 324 | 51204 |
| r4d_moe_iqr | gfx1201 | 50 | 239 | 7 | 32 | 8720 |
| r4d_iqr_embed | gfx1201 | 17 | 12 | 0 | 0 | 512 |
| r4d_logits_vocab | gfx1201 | 6 | 256 | 3 | 1104 | 51200 |

The four gemm units hold the same kernel templates split by form, so their kernel counts (742 to 843) are the form matrix,
not different code. The many VGPR-spilling kernels are the occupancy-capped decode and wide-tile variants the unit
instantiates on purpose; the baseline to protect is the per-kernel record, not a spill-free rule.

## Recommendation

**Whole-unit owned (three-way merge applies):** `r4d_gemm_iqr` with `_p1.._p3`, `r4d_iqr_embed`, `r4d_logits_vocab`. They
are the units we already rewrite (gfx1100) and patch (gfx1201), upstream changed them only in the 2.2.0 window and not
since, and each is self-contained over `r4d_iqr.h` / `r4d_iqr_dec.h`. Owning them makes the three layers (upstream text,
libr3 rewrite, patch) into one reviewed file per architecture-neutral unit with a three-way merge against upstream at
each bump.

**Keep upstream (no ownership yet):** `r4d_moe_iqr` (985 lines, no local edit; upstream still moves it, four commits in
2.2.0, so a fork would only add merge work), `r4d_iqr_layout.cpp`, `r4d_iqr.h`, `r4d_iqr_dec.h` (no local edits, no
upstream change since 2.0.0).

**Extracted family (structural review, not a three-way merge):** `r4d_gemm_8a8_prefill.h`. It is shared with the int8
rows (rad0061 and rad0065 patch it, eight rewrite blocks attach to it, upstream main touched it once), so a fork would
diverge for non-IQR units; extract the IQR-only parts (the fused single-band tile and the band dispatch) into the owned
unit and review the shared remainder structurally at each bump.

**Rewrites that could be retired if the overlay owns the unit:** all 25 blocks of `r4d_gemm_iqr.hip.rw`, the single block
of `r4d_iqr_embed.hip.rw` and the nine of `r4d_logits_vocab.hip.rw` (their content moves into the owned source behind an
architecture guard); the eight blocks of `r4d_gemm_8a8_prefill.h.rw` stay until the shared header is split. The stage-3
check before any retirement is the unit harness on both architectures: equal output hashes, equal resource records
(modulo renamed symbols), replay of the changed rows without regression.
