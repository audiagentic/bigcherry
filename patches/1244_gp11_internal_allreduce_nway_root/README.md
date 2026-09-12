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

## Real hardware finding (2026-09-12): topology-matrix coverage beyond the proven N=3 point is deferred-hardware

Brutus (this project's only real GPU host) has 4 GPUs across 3 different
architectures (2x gfx1100, 1x gfx1201, 1x gfx1030) -- only 2 GPUs share an
architecture, so no real same-architecture N>=3 topology exists to widen
this patch's coverage matrix on. Attempting a heterogeneous 3-GPU
`-sm tensor` run (2x XTX + the gfx1201 card) segfaults immediately, but a
real discriminator test -- reproducing the identical crash on a completely
stock, patch-free `bigcherry-native` build at the same pin/topology --
confirmed this is a pre-existing HIP-runtime limitation in generic
cross-device tensor copy (`ggml_backend_cuda_cpy_tensor_async`, inside
`libamdhip64.so`), not a defect in this patch or `0840`. GPT-approved
disposition (`req_fa5e1027390941fe`): deferred-hardware for the
topology-matrix gate; do not reject or demote on this evidence. Full detail
in PGC02's plan item.

## Disposition

`state` stays `"untested"`. Do not promote or generalize the measured
result outside the qualified envelope until PGC02's gates are run.

## Requires

`1001_hip_internal_allreduce`.
