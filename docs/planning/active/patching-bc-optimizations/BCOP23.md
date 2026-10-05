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

Rebased on the current b11402-era BigCherry line. Upstream #29941 is already baseline history for this pin family and must no longer be treated as a backport task. The remaining correctness/design gate is open upstream #29953: allocation currently can derive a different effective J from the J used by execution; #29953 resolves one `J_best`, sizes Q8_1/ids padding from it, and passes that exact value to launch.

The important BigCherry-specific finding is that patch `0300_mmq_forced_j` already owns a second copy of the native J scan through `ggml_cuda_mmq_native_j_best(...)`, while its host proof explicitly compiles that lifted scan. That was correct for the older upstream shape, but becomes duplicate policy when #29953 lands because upstream moves native J selection ahead of allocation and stores the resolved J in `mmq_args`. Do not preserve the duplicate scan after that transition.

## Implementation

1. Treat merged #29941 as baseline correctness. Verify the current composed source contains the MoE `ne12` padding fix; fail the qualification if it does not, but do not create another backport unless the actual pin lacks it.
2. Prototype #29953 against the current composed source as a disposable qualification patch. Preserve upstream's single resolution point: `J_best` is chosen once using config availability, shared-memory limit, precision, fallback and RDNA3/RDNA4 MoE average-column heuristic; the same value owns scratch padding, `ids_dst` padding and launch.
3. Rebase `0300_mmq_forced_j` onto that ownership model. Native/default mode consumes `args.J_best` and must not call `ggml_cuda_mmq_native_j_best`. Forced mode may replace `args.J_best` only after `ggml_cuda_mmq_config_is_eligible` plus the real launch-width tail-safety check passes.
4. Delete the lifted native scan and its duplicate policy tests once the #29953-shaped path is adopted. Keep the forced-J switch, hard eligibility and forced/native noise-canary behavior.
5. Update the 0300 host fixture so native tests inject a resolved `args.J_best`; assert forced `J == args.J_best` reaches exactly the same launcher. Add sizing/launch invariants around J boundaries `{7,8,9,15,16,17,31,32,33,63,64,65,127,128,129}` for dense and MoE shapes.
6. Run patch apply/idempotence and the emitted-transform host proof, then a real HIP build. On gfx1100/gfx1201 run the lopsided `MUL_MAT_ID` reproduction plus Flash-Next expert shapes, repeated multi-request execution and MTP on/off controls. No performance tuning is valid before this passes.

## Exact ownership / files

- `patches/0300_mmq_forced_j/patch.py`: remove `ggml_cuda_mmq_native_j_best` ownership after #29953-shaped integration; preserve forced override/eligibility only.
- `patches/0300_mmq_forced_j/validation/checks.py`: replace the lifted-native-scan fixture with resolved-J injection and allocation/launch equality assertions.
- `patches/0300_mmq_forced_j/README.md`: document that upstream owns native J selection and 0300 only overrides a resolved candidate experimentally.
- composed `ggml/src/ggml-cuda/mmq.cu` / `mmq.cuh`: one `J_best` determines padding and launch.
- `docs/planning/active/patching-qwen-flash-next/QFP10.md`: remains technical owner for subsequent Flash-Next IQ expert tuning.

## Mock / cheap discriminator

The existing 0300 host fixture is already the correct mock surface: it compiles emitted patch blocks against a minimal config table. Rework that fixture before touching GPU tuning. The mock must prove:

- native path launches the supplied resolved J without rescanning;
- forced J replaces only the launch choice, not allocation ownership;
- invalid/undefined/shared-memory-ineligible forced J fails closed;
- tail safety uses real `ncols_max`;
- dense and MoE allocation-J equals launch-J at every boundary case.

This is hardware-free correctness evidence only; it cannot qualify HIP performance or memory safety.

## Validation / gates

Correctness: exactly one resolved native J owns Q8_1 padding, ids padding and launch; no second native scan remains in 0300; no OOB/illegal access; deterministic multi-request output remains valid. The host fixture, real build, gfx1100 and gfx1201 execution must all pass.

Performance is secondary. Record PP/TG before/after only to reject a pathological correctness integration (>2% representative-lane regression). Forced-J performance experiments resume under QFP10/0300 only after the invariant passes.

Stop rule: if upstream #29953 materially changes before adoption, rebase to its final ownership boundary rather than maintaining this prototype. Do not carry a permanent BigCherry copy of upstream's native J selector.

## Acceptance Criteria

- Current pin is explicitly verified for #29941-equivalent MoE padding safety.
- #29953-equivalent single-source J selection passes host fixture, real HIP build and gfx1100/gfx1201 execution.
- `0300_mmq_forced_j` owns only experimental override/eligibility; default/native J policy is upstream-owned.
- Allocation-J == launch-J for dense/MoE boundary and real Flash-Next shapes.
- QFP10 is unblocked without a duplicate selector, dispatch table or padding estimator.

## Related

QFP10; patch 0300; upstream llama.cpp #29941 (merged), #29953 (open). #29948 remains NVIDIA-MMA-specific mechanism evidence, not an AMD implementation target.
