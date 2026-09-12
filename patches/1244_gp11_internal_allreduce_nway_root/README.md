# 1244_gp11_internal_allreduce_nway_root

Plan item GP11 (superseded by PGC02, `patching-gpu-collectives`). See
`SUMMARY.md` for the full mechanism, bug history, and real-hardware
evidence gathered so far -- do not duplicate it here.

## Status

Landed with positive evidence for tested regimes: two real integration bugs
(missing per-block arrival offset, cross-device mapped-pointer alias reuse)
found and fixed; subsequent evidence includes clean high-repetition harness
validation, real full-stack MTP correctness, alternate-root validation, and
a positive N=3 decode result.

Not yet qualified for promotion. PGC02
(`docs/planning/active/patching-gpu-collectives/PGC02.md`) owns the
remaining gates: a controlled long soak, supported root/topology/size
coverage, provider+threshold telemetry (there is a real, still-open -1% to
-2% pp1024-4096 softening that needs internal-call-count instrumentation
before sign-off), and a controlled comparison against both RCCL and an
unmodified baseline.

## Disposition

`state` stays `"untested"`. Do not promote or generalize the measured
result outside the qualified envelope until PGC02's gates are run.

## Requires

`1001_hip_internal_allreduce`.
