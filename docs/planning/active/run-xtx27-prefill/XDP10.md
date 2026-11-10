---
id: XDP10
order: 9
plan: run-xtx27-prefill
state: pending
created-at: '2026-11-11T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Hardware-selected multi-variant quantized model bundle and loader

## Purpose / ownership

Implement the owner's XDP03 design: derive several **immutable** encodings
from the same, fully fingerprinted Qwen3.8-27B BF16 tensors, keep variants
on disk, and select **only the assigned GPU's compatible variant at model
admission**. First choices to test are native W8A8 INT8 for gfx1100 XTX
ranks and native E4M3/block-FP8 on gfx1201 R9700, against common W8A8 on
all three. IQ6R is a later independently calibrated candidate for more
resident KV capacity. Hardware support does not prove kernel speed,
converter compatibility or language-model accuracy.

This item owns **model variant production, provenance, format-selection
ABI and load/admission semantics**, not the three-GPU collective or
unequal tensor sharding (VTP12/VTP13). XDP03 remains the owner of the
original BF16-derived quant recipes, per-layer candidate selection and
whole-layer stage-placement alternative (XDP02). VTP06 owns BF16
calibration, IQ6R quality and held-out evaluation.

## Phase 1: validate native producer/consumer contracts (card-free first)

1. Inventory RAD2 `rad-convert` recipe capabilities and real gfx1100
   libr3 / gfx1201 libr4d kernel rows for each **actual** proposed E4M3
   variant (plain FP8 versus block FP8 are different layouts), W8A8,
   optional IQ6R, GDN, complete attention, embeddings/head and MTP.
   Require concrete converter `--plan-only` and kernel-layout evidence
   for each operator, shard and GPU. Do not infer support from an FP8
   WMMA instruction or from a synthetic GEMM.
2. Derive candidate variants **directly from original BF16**, never
   re-quantize the existing Q4 or INT8 file. Record BF16 full-shard
   hashes, tokenizer/config, converter/build/plugin ABI, recipe/hash,
   tensor name, original shape, split offsets/shape, quant layout and
   scale metadata, expected kernel row and numerical output contract.
3. Explicitly distinguish (a) different quant for **whole-layer
   placement** and (b) different quant for **same-layer TP shards**.
   Case (b) requires a single declared activation ABI and common partial
   output/accumulation semantics. Current qwen35_w8a8 global activation
   mode does NOT license arbitrary per-rank FP8+INT8 mixing. Reject it
   pending a valid mixed-mode graph/producer plan (VTP12).

## Phase 2: multi-variant package, start with sidecar composition

Use a fingerprinted immutable **bundle manifest plus separate
existing .rad artifacts**, not a speculative RAD2 format change in v1.
A manifest entry identifies logical tensor + exact offset/range and its
compatible alternatives: `variant_id`, source BF16 digest,
representation/scale/activation ABI, kernel ABI and architecture set,
exact file and per-payload SHA-256, quality status and required bytes.
The placement profile selects one variant **per physical executor's
owned shard**; whole-layer stage mappings are separately expressible.
Two XTX ranks must use identical shared-layer representations unless a
separate numerical/protocol qualification permits otherwise.

At startup: resolve device UUID/architecture and pinned kernel plugin;
check the actual allocation and tensor census; reject missing,
ambiguous, overlapping, incomplete or incompatible shard sets, invalid
offsets, absent converter planes, unapproved quality, wrong source
hash, unsupported FP8 layout or activation producer. No silent quant
conversion, downgrade, host spill, fallback or mid-request re-selection.
The resolved immutable manifest/receipt records precisely what each GPU
loaded. Unselected variants remain on disk and never consume VRAM.
Deduplicate common immutable metadata/artifacts by hash on disk, not by
assuming shared GPU memory. Start with restart-only changes; no live
migration or re-quantisation in v1.

## Phase 3: tests and measurement

- Card-free golden and adversarial fixtures for layer-range mapping,
  fused FFN gate/up/down offsets, attention/GDN head ownership, variant
  priority, duplicate/partial shards, missing hashes, mismatched BF16,
  plugin ABI, device order change, missing FP8, unsupported mixed
  activations, missing MTP and exact fail-closed messages.
- Real producer/converter checks on both hardware types, with separate
  independent numerical kernel oracles versus BF16-derived reference.
  Use actual Qwen3.8-27B shapes at M=1/4/8/64/128/512/2048; capture
  cold and warm GEMM, **activation quantisation/scale**, and layout
  conversion separately (VTP12 owns simultaneous cross-device timing).
- A preflight load using one selected variant per device, verify model
  tensor provenance and selected-device bytes; inspect measured
  physical VRAM weights/KV/GDN/scratch/graph reserves. No duplicate
  unused variants in GPU memory or hidden CPU transfer.
- BF16-vs-mixed held-out quality: per-tensor and full-model errors,
  KLD/top-1/PPL, sensitive layer/head checks, repeatability with
  speculation off, then on only if separately qualified. VTP06 owns
  approved corpus and gates. Never promote an unapproved tiny pilot.

## Acceptance criteria / decision

One genuinely heterogeneous per-device bundle **loads reproducibly or
fails explicitly** against the real GPU/plugin inventory, passes whole
tensor and numerical dispatch checks, records per-physical allocation,
and uses only selected variants in VRAM. End-to-end quality acceptance
and performance win are separate VTP13 gates. If RAD2/libr3 cannot
legally mix activation modes at a within-layer TP boundary, report a
bounded negative result and leave only whole-layer selection enabled.
Do not change production serving defaults or the pinned vendor tree.

Dependencies: XDP03, VTP06, VTP08, VTP12 and VTP13. No push or PR
without an owner instruction; use the existing experimental worktree.

## 2026-11-11 feasibility gate — defer heavy loader changes

VTP12 executed 72 same-build native INT8/FP8 timing-only selftest
invocations under exclusive GPU leases, A/B/B/A over 30/30/40,
~28/28/44 and 25/25/50 physical FFN partitions. Mixed FP8 on R9700
slowed the *ideal FFN decode* barrier by 9.6% for ~28/28/44 and
34.4% for 25/25/50 at M4; at M64 it improved the ideal barrier by
2.7%, **exactly matching** a hypothetical best-native-INT8-row
choice on the R9700. These are not numerical or model-serving results.

Do **not** promote XDP10's multi-variant loader based on this small
FP8 gain. First determine which actual mixed-architecture kernel rows
Radiance selects at M4/M64, benchmark native activation preparation,
model quality and three-card no-P2P collectives. Retain the immutable
sidecar variant contract and fail-closed tests as planned; native per-GPU
INT8 dispatch tuning is now a mandatory control before larger work.
See VTP12 for source receipts and corrected replay of the M64 narrow
versus tiled INT8 control.

## VTP12 42.65% split evidence (2026-11-11)

Eight additional exclusive-lease 24-case, synthetic warm-weight
ABBA runs (192 selftests total) compared symmetric 4992/4992/7424
with uneven 5120/4864/7424 physical FFN widths at M=1/4/8/64.
Unequal XTX widths remove the severe 128-aligned down-K decode
slowdown, keeping the ideal FFN barrier within 2.1% of 30/30/40
INT8 control. At M64, mixed R9700 FP8 improves the ideal bound
by <1%, matched by best-observed native INT8 dispatch. Neither
within-layer mixed activation graph nor simultaneous three-GPU
collective, GDN/attention, BF16 quality or real service speed
has been proved. **Defer loader implementation/promotion until
optimised all-INT8 dispatch and cross-device numerical/transport
are measured.** Preserve the static XDP10 artifact plan as a
separate eventual option. See VTP12 hardware receipt matrix.
