# VTP44 design review 1 (GPT, 2026-11-16)

Brief: the design text sent to the reviewer (read-only, no source access, so the reviewer worked from the brief).

## BLOCKING

- **The descriptor must define exact storage semantics, not just row size.** A formula such as `49/16 * HD` gives a byte count only when the result is integral; it does not define offsets, alignment, scale placement, packing order, or padding. For example, at an unsupported head dimension the core could allocate a page using the formula while a kernel computes a different scale offset, causing incorrect reads or writes. Make the descriptor specify the layout’s exact size and offset rules, with validation that each supported geometry produces integral, aligned storage. The table should be declarative data shared by the ABI, core, and plugins; avoid embedding a C++ function pointer in an ABI meant for other consumers.

- **A geometry set cannot stand in for geometry-specific tuning.** If a generated row uses one option-bit set or kernel instantiation for all listed tuples, a gqa12 case can inherit gqa6 tile sizes or LDS skew and spill registers or produce incorrect synchronization. Keep supported tuples and their kernel instantiations explicit. Let the generated row list refer to per-tuple tuning/instantiation metadata; share the row-construction logic, not tuning choices that differ by geometry.

- **“One rewrite per template” is safe only if it binds to every intended instantiation and proves that it did.** A broad textual rewrite can miss a newly added instantiation or match the wrong one; the gfx1100 build may still compile while silently omitting a kernel row. Require exact match counts and a post-rewrite audit that compares expected and produced instantiations. Prefer generating gfx1100 source from the same instantiation list if the build can support it; otherwise use tightly scoped template-level rewrites with mandatory count checks.

## SHOULD-FIX

- **Model limits should be manifest data, but kernel capability must remain engine data.** A manifest should select a layout and state model-specific limits. It must not make an unsupported `(head_dim, gqa, block_size, op)` appear supported. If the manifest requests a geometry without a matching kernel row, startup should refuse clearly; it must not fall back to another layout’s row.

- **The fused gate-and-quantizer path needs explicit composition semantics.** I’d model it as an operation/layer kind (`attn_paged_gate_quant`) that consumes a layout descriptor and has its own supported geometry/instantiation set. It is not merely a decorator if fusion changes the epilogue’s output representation, quantization parameters, or store path. Treating it as an independent third axis risks generating invalid cross-products; treating it as a decode-row decoration risks hiding those differences. A generated table should list only valid combinations.

- **I recommend two patches.** First extract the existing gqa6 rows into the descriptor and geometry-set representation, with no new geometry. Prove that refactor, then add gqa12 and qwen4exp plumbing in a second patch. This isolates failures and makes the textual guards easier to review. If the owner requires one experiment patch, keep the refactor and feature changes in distinct, guarded sections with separate validation artifacts; don’t promote it until both gates pass.

- **Prove the gqa6 refactor at multiple levels.** Compare a deterministic row-table dump including constraints, function/kernel identities, and option bits; compare the selected instantiations and relevant binary artifacts under the same compiler/toolchain; run the existing prefill/decode/store selftests; and compare KLD outputs on the same inputs and environment. “Byte-identical behavior” should be supported by these checks, not inferred from source-level similarity. Binary equality may be too strict across toolchains, so record the compiler and build inputs and distinguish code identity from output equivalence.

- **Keep future model forms out of this change unless needed.** Head dimension 128 should fit if the descriptor’s storage arithmetic and kernel support are genuinely parameterized. MLA, sliding-window semantics, or quantized K require new storage or attention contracts; don’t add speculative descriptor fields for them now. In particular, “V/K halves are direct-addressable” may not describe layouts with packed or interleaved data. Add such capabilities when a concrete kernel needs them, and make unsupported forms fail explicitly.

- **Likely reuse traps from rad0037/38:** the per-plugin layout-name lists and row-size lists; `gqa == 6` refusal logic; assumptions that page sizing is derivable from a single row-size multiplier; and separate `attn_decode_gq_*` rows with fused layout assumptions. Any retained copy can cause a new layout to be accepted by one layer but rejected or mis-sized by another. The brief also flags optional rewrite pairs anchored on patched text: that is a production fragility worth auditing, since a changed earlier hunk can make a later rewrite silently stop matching.

## NIT

- Make refusal diagnostics identify the requested layout and the unsupported tuple, including operation and architecture. This makes missing gqa12 gate-quant coverage distinguishable from an unsupported layout.
- A table-driven selftest is useful, but it should derive expected offsets and byte counts independently enough to catch a bad descriptor; generating implementation and expected values from the same table can reproduce the same mistake on both sides.

## Answers to the numbered questions

**Q1.** The descriptor plus geometry-set-row approach is sound for shared storage/layout metadata and row construction. It breaks if the set collapses geometry-specific kernel tuning into one configuration. Keep tile shape, LDS bank skew, launch parameters, and any register-pressure-driven specialization per geometry (or per explicitly defined tuning class). Only share them when measurements and correctness checks support that sharing.

**Q2.** Make gate-plus-quantizer a distinct operation/layer kind whose rows consume a layout descriptor. Generate only supported operation/layout/geometry combinations. This preserves a generic layout axis while representing the fused epilogue’s different semantics.

**Q3.** Source generation from the shared template-instantiation list is the strongest option if the build permits it. Otherwise, use a scoped rewrite per template with exact expected counts and a post-rewrite inventory check. The silent failure is a missed or wrongly matched instantiation that still compiles but leaves a runtime row absent or mismatched.

**Q4.** Two patches minimize risk: refactor and validate gqa6 first, then add gqa12 and plugin plumbing. The broad refactor is the risky part; combining it with new kernels makes a regression harder to localize and weakens the proof that the old rows stayed unchanged.

**Q5.** Capture the old and new row-table dumps and compare constraints, function pointers/identities, option bits, and selected instantiations. Build with pinned compiler inputs and compare relevant binaries where reproducible; run the same selftests and KLD cases with fixed inputs and environment. Preserve those artifacts with the patch so the comparison is reviewable.

**Q6.** Head dimension 128 is a useful test of whether the arithmetic is actually generic. MLA and other KV representations may need a different storage/attention contract; sliding window adds page-visibility semantics; quantized K may introduce its own scale and packing rules. Keep those out until required, and avoid assuming every layout is two independently addressable K/V halves.

**Q7.** The hard-coded per-plugin lists, the gqa6 guard, repeated byte-size arithmetic, and fused gate-quant rows are the reuse traps called out by the brief. I can’t determine whether there are additional traps in the merged implementation: the workspace search returned no source files, so this review is based on the supplied brief, and I could not inspect the actual patches, guards, table definitions, or rewrite rules.
