---
id: BCOP23
order: 23
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T05:08:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: S
---

# Qualify single-source MMQ J selection before further Flash-Next tuning

## Description

Follow-through from the QFP10 implementation audit. Current BigCherry pin `050439614` predates upstream #29941 and current #29953. QFP10 already identifies the correctness requirement; this item records the concrete action needed to make it executable without creating another MMQ selector.

The important upstream mechanism is #29953: resolve `J_best` once with the real MMQ config/shared-memory checks, use that same value to size Q8_1 padding and padded `ids_dst`, then pass it into launch. This eliminates the existing split ownership where allocation estimates J separately from execution. BigCherry patch `0300_mmq_forced_j` already owns forced-J experimentation and must be adapted to consume the resolved native J rather than grow a second estimator.

## Steps

1. Backport #29941 or move the pin beyond it before any MoE MMQ performance qualification.
2. Prototype #29953's single-source `J_best` invariant against the current pin in an isolated composition with `0300_mmq_forced_j`; do not create a new architecture/shape table.
3. In `0300`, preserve forced-J only as an experiment override. Native/default mode must obtain one resolved J from the production selector and use it for allocation and launch.
4. Add host-fixture assertions that allocation-J == launch-J for dense and MoE shapes around J boundaries `{7,8,9,15,16,17,31,32,33,63,64,65,127,128,129}` and that invalid forced J fails closed.
5. Run the existing patch apply/idempotence/build validation, then HIP execution on gfx1100 and gfx1201 with the lopsided Flash-Next expert shape. Require no OOB/illegal access across repeated multi-request runs.
6. Only after this gate passes resume QFP10 VDR/direct-F32 performance work. Treat #29948 FFN gate/up+GLU fusion as mechanism evidence only: its current match explicitly requires NVIDIA MMA and is not an AMD candidate without a separate measured AMD accumulator/writeback design.

## Files

- `patches/0300_mmq_forced_j/patch.py`
- `patches/0300_mmq_forced_j/validation/*`
- `patches/0300_mmq_forced_j/README.md`
- upstream/current-pin `ggml/src/ggml-cuda/mmq.cu`
- upstream/current-pin `ggml/src/ggml-cuda/mmq.cuh`
- `docs/planning/active/patching-qwen-flash-next/QFP10.md`

## Validation

Correctness gate: one J value owns scratch padding, ids padding and launch; zero allocation/launch disagreement; no memory fault/OOB on gfx1100/gfx1201; deterministic multi-request output remains valid.

Performance is explicitly secondary for this item. Record PP/TG before/after only to reject a pathological correctness backport (>2% regression on representative lanes). Forced-J performance experiments remain owned by patch 0300/QFP10 after the invariant is proven.

## Acceptance Criteria

- #29941-equivalent MoE padding safety is present.
- #29953-equivalent single-source J selection passes host fixture, real build and HIP execution.
- `0300_mmq_forced_j` contains no independent native-J estimator in its default path.
- QFP10 is unblocked for performance qualification without duplicate selector ownership.

## Related

QFP10; patch 0300; upstream llama.cpp #29941, #29953. #29948 is not an AMD implementation target in its current form.
