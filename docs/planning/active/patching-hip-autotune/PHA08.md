---
id: PHA08
order: 8
plan: patching-hip-autotune
state: pending
created-at: '2026-09-11T22:57:38.584940+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: L
---

# HIP Flash-Attention D=72 VLM aperture violation: root-cause before fallback (#28608/#28664)

## Description

Resolve the gfx1100 HIP tiled Flash-Attention D=72 aperture violation for VLM workloads. Prefer a kernel/root-cause fix; retain the CLIP-only matmul fallback as a narrow safety valve. Upstream PR #28664 closed without merge while issue #28608 retains evidence of the fault, so the workaround must not be assumed present merely because the PR existed.

## Steps

1. Reproduce on the current llama.cpp pin (`050439614`) on dual XTX with representative Qwen/Gemma mmproj lanes at 1024/1600/2048/2560px; record Q sequence length and exact `flash_attn_tile<D,D,cols_per_block,...>` instantiation.
2. Reproduce with one XTX as a topology control and `cols_per_block=32/64`. Treat tile width as diagnostic only: upstream evidence says 32 can move the crash earlier.
3. Instrument `ggml/src/ggml-cuda/fattn-tile.cuh` and helpers for Q/K/V/O bounds, shared-memory offsets, padded sequence extents, vectorized D=72 tail loads/stores, launch dimensions, workspace sizes, and allocation byte ranges. Compare generated gfx1100 ISA/resource use against D=64/80 controls and gfx1201 where supported.
4. Test the D=72 non-power-of-two/vector-tail hypothesis with guarded tail-safe loads/stores for final lanes before considering broad kernel changes.
5. If no kernel fix is proven, retain only the #28664-style `tools/mtmd/clip.cpp::clip_graph::build_attn` fallback: HIP + CLIP + `d_head == 72`. Never disable main-model FA or all HIP FA.
6. Feed architecture/shape policy through existing HIP-autotune ownership; do not add a second dispatch registry. Retire a local fallback when upstream lands an equivalent targeted fix.

## Detailed Solution & Technical Design

Issue #28608 provides useful boundary evidence: the failure resolves in `flash_attn_tile<72,72,64,1,false>`, scales with image/sequence size, and forcing a narrower tile does not cure it. That makes D=72 tail geometry, storage sizing, vector-width assumptions, or another allocation/indexing invariant higher-value hypotheses than the 64-column selection itself.

The optimisation opportunity is to recover tiled FA for D=72 rather than permanently paying matmul-attention cost. Audit vectorized `D/pack` loops, final-pack masking, padded key/query lengths, and any storage sized by floor division while wider loads/stores are issued. The image-size threshold may expose an address error that exists at shorter lengths but remains in-bounds.

PHA08 owns only the D=72 correctness gate and narrow candidate fix. Generic FA architecture/shape tuning remains with `patching-hip-autotune`; generic kernel work remains in the normal patch/upstream path. Related aperture-violation reports are comparison evidence only unless traces resolve to the same defect.

## Code Samples & Guidance

Debug invariants should validate byte ranges, not only logical element indices:

```cpp
// debug-only pseudocode; use actual tensor strides/types
const size_t last_byte = base + logical_index * elem_size + vector_width_bytes;
GGML_ASSERT(last_byte <= allocation_end);
```

For a proven D-tail defect, prefer a compile-time/tail predicate inside the existing load/store helper over cloning a D=72 kernel.

## Files

`ggml/src/ggml-cuda/fattn-tile.cuh`; related FA helpers/dispatch; `tools/mtmd/clip.cpp`; `patches/<new-id>/**`; `tools/bigcherry/patch/**`; `tools/bigcherry/tuning/correctness_evidence.py`; `tools/tests/patch/**`; `tools/tests/tuning/**`; `docs/evidence/<run-id>/`.

## Validation

Current pin plus a suitable upstream control; 1x/2x XTX; gfx1201 comparison where supported; 1024/1600/2048/2560px; D=64/72/80 controls; cols-per-block 32/64 diagnostic only; FA candidate versus CLIP matmul fallback. Capture HSA fault absence, logits/embedding parity, image encode latency, peak VRAM/workspace, kernel time, VGPR/SGPR/spills, and text-only throughput. Run at least 20 repeated 2048/2560px encodes before accepting a memory-safety fix.

## Effort & Risk

Medium-high. A narrow fallback is low risk; a kernel fix is higher value but must prove the exact OOB/codegen predicate. Avoid speculative compiler workarounds without ISA/bounds evidence.

## Standards

Package-only changes remain fail-closed, idempotent, ancestry-pinned, benchmarked, and removable. No broad backend disablement.

## Acceptance Criteria

Preferred: a D=72 tiled-FA fix completes 20x 2048/2560px runs on gfx1100 with no aperture violation, numerical parity to matmul fallback, and >=5% VLM encode improvement versus fallback with no >2% healthy-shape regression. Otherwise, a narrow HIP+CLIP+D=72 fallback passes correctness/fault/non-regression gates. Current-pin non-reproduction requires the same stress matrix before closing the item.

## Notes

Provenance: shared ChatGPT conversation, 11 Sep 2026, '#28664 — direct 2×7900 XTX HIP Flash-Attention crash'; source https://github.com/ggml-org/llama.cpp/pull/28664. Existing PHA06 is a different completed item.

2026-10-05, folded in from BCOP21 (audit backfill) - root-cause before retaining the fallback: upstream #28664 closed without merge while #28608 keeps evidence of gfx1100 faults in `flash_attn_tile<72,72,64,1,false>`. Reproduce on current pin `050439614` with D=64/72/80 controls on gfx1100/gfx1201; inspect vector/tail bounds, padded extents, shared-memory offsets, workspace sizing, generated ISA and resources; keep only the narrow HIP+CLIP+d_head==72 fallback until root cause is known.

Sources: https://github.com/ggml-org/llama.cpp/issues/28608 ; https://github.com/ggml-org/llama.cpp/pull/28664

## Change Log

- 2026-09-11T22:57:38.584940+00:00 (created-by): Created by agent
- 2026-10-05: BCOP21 audit backfill added root-cause gate.
- 2026-10-05: Transplanted the structured D=72 root-cause/tail-geometry plan from `automation-qfp-indexer-20261004`, preserving current-pin audit guidance.

## Ledger-events

- chg_20260911_225756_added-six-provenance-rich-buil_2622
- 2026-09-11T22:57:56.719160+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-05T04:55:02.911053+00:00 (updated-by): Updated: section:notes
