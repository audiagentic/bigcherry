# 1245_gp11_mmvq_fusion_ncols_gate

Plan item GP11 (superseded by PGC02, `patching-gpu-collectives`). See
`SUMMARY.md` for the full real-hardware negative result -- do not duplicate
it here.

## Status

**Negative result, real hardware (2026-09-04).** Widening the MMVQ
MUL_MAT+GLU fusion gate beyond `ncols_dst == 1` compiled cleanly, fired
8,450 real fused dispatches (0 in control), and removed 25,350 total
dispatches -- but MMVQ kernel time increased +17.9% and end-to-end
throughput dropped -9.4%. The fused kernel's added per-dispatch cost
swamps the quantize/SwiGLU savings the dispatch reduction bought. Retained
as a documented negative result specifically so this gate is not
rediscovered and re-attempted.

PGC02 explicitly inherits this as retained negative evidence: "retain
patch 1245 as negative/dispositioned evidence; do not reopen it."

## Disposition

`state` stays `"untested"` (meaning "not contract-qualified", not
"unmeasured" -- see `SUMMARY.md`'s Evidence status section). Do not
promote. Do not add to any patch-set. Do not reopen without new evidence
that changes the structural picture SUMMARY.md documents.
