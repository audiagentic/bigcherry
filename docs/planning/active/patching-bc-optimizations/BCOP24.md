---
id: BCOP24
order: 24
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T06:14:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# Track AMD Q2_K spill and Q1_0 native-permute qualification

## Description

Action record for the 2026-10-05 deep audit of BCOP14. The audit converted the two upstream AMD quant candidates from a broad investigation into bounded, implementation-ready qualification work. BCOP14 remains the technical owner; this item records what must happen for the audit to be acted on and prevents another plan/dispatch mechanism being created.

## Steps

1. Execute BCOP14's exact-upstream A/B for llama.cpp #29910 on gfx1100/gfx1201, including ub16-512, resource/ISA evidence and `MUL_MAT`/`MUL_MAT_ID` correctness.
2. Execute BCOP14's #29927 exhaustive Q1_0 selector reference test, then MMQ/MMVQ backend-op and pp/TG A/B on gfx1100/gfx1201.
3. Classify each candidate independently as adopt-now, wait-for-upstream, reject, or already-upstream using BCOP14's promotion/stop thresholds.
4. If accepted before upstream merge, carry only a disposable exact-upstream patch. Reuse existing HIP architecture/config ownership for any proven crossover; do not create a BCOP-owned dispatch table.
5. Close this item when both candidates have classifications recorded in BCOP14 or become part of the normal llama.cpp pin.

## Detailed Solution & Technical Design

No separate implementation is owned here. See BCOP14. This item exists solely to ensure the audit is executed and resolved without duplicating technical design.

## Files

- `docs/planning/active/patching-bc-optimizations/BCOP14.md` — authoritative technical qualification plan.
- Upstream affected files are listed in BCOP14.

## Validation

BCOP14 acceptance criteria are authoritative. Do not mark complete from compiler/spill evidence alone; local gfx1100/gfx1201 correctness and performance classification is required unless the changes arrive through a subsequently validated normal pin bump.

## Effort & Risk

Small follow-through item. Avoid expanding scope into new quant formats, new dispatch registries or unrelated MMQ tuning.

## Standards

Follow existing BigCherry patch/build/ABBA and performance-integrity standards. Prefer upstream source verbatim for attribution.

## Acceptance Criteria

- #29910 and #29927 each have a recorded disposition backed by BCOP14 evidence, or are validated as part of a newer upstream pin.
- No duplicate technical plan or production dispatch mechanism was introduced.
- Disposable local qualification patches are removed once no longer needed.

## Related

BCOP14; llama.cpp #29910, #29927.

## Change Log

- 2026-10-05: Created from the BCOP14 deep audit to track execution/disposition only.
