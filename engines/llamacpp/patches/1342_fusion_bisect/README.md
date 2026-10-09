# 1342_fusion_bisect

## Promotion record

Promotion record (QFP18 lightweight evidence-reuse tier, pin b11474 / b9acf138, 2026-10-08). A diagnostic, promoted as a
neutral enabler: it prints only when its flag is set and changes nothing otherwise, so it can be used on the production
binary without a special build.

- Build and run: experiment build `b-metamem-mig-fuse` (production + 1342) compiled clean. Flash-Next production
  topology, ctx 245760 (runs `metamem-mig-fuse`, `metamem-mig-fuse-skip`).
- Activation, counters at exit under `BIGCHERRY_META_MEM=1`: fusions taken per family - ADD 137, GATED_DELTA_NET 1512,
  MUL 1902 (36138 nodes elided), MUL_MAT 291, MUL_MAT_ID 1314, RMS_NORM 10719, SCALE 8347, SOFT_MAX 1979.
- The switch works: with `BIGCHERRY_FUSION_SKIP_OPS=MUL` the MUL family is gone from the counters (the nodes it would
  have taken are picked up by others: ADD 2185, SSM_CONV 1620, ...) and the greedy text changes (94012812de5c against
  2571b60b1505) - the MUL-start fusion is one of the families whose result is not bit-identical (FKE01).
- Neutral without the flag: greedy text equals production (md5 2571b60b1505).

## Native llama.cpp comparison

Native llama.cpp has no counterpart: the patch only adds reporting behind its flag. With the flag unset the build behaves as without the patch.
