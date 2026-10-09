# 1320_meta_compute_timing

## Promotion record

Promotion record (QFP18 lightweight evidence-reuse tier, pin b11474 / b9acf138, 2026-10-08). A diagnostic, promoted as a
neutral enabler: it prints only when its flag is set and changes nothing otherwise, so it can be used on the production
binary without a special build.

- Build and run: experiment build `b-metamem-mig-diag` (production + 1319 + 1320 + 1325 + 1346) compiled clean; the
  patch applies in the production composition at b11474 (it anchors next to the per-device arena code of 1339 / 1340).
- Activation with `BIGCHERRY_SUBMIT_TIMING=1` (run `metamem-mig-diag`): 108 `BIGCHERRY_META_TIMING` lines; per
  512-token chunk the tensor split spends 9.6 ms rebuilding subgraphs, 20.6 ms launching 97 subgraphs and 2.1 ms
  enqueueing AllReduce (32.3 ms of host time).
- Neutral: greedy text with the flag on equals the flag-off arm (md5 fe307bdfb7e1).

## Native llama.cpp comparison

Native llama.cpp has no counterpart: the patch only adds reporting behind its flag. With the flag unset the build behaves as without the patch.
