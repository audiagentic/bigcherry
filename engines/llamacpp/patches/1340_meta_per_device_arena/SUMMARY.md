# 1340_meta_per_device_arena

**Status:** validated
**Plan item:** MSM01/MSM02

## What it does

Single package for Meta per-device compute arenas plus the `BIGCHERRY_META_MEM` memory-report diagnostic formerly packaged as 1339. The arena mechanism remains default-on with `BIGCHERRY_META_PER_DEVICE_ARENA=0` as the native/common-size control; `BIGCHERRY_META_MEM=1` remains diagnostic-only.

## PA44-E packaging proof

1339 and 1340 were adjacent and 1340 already required 1339. Their edit/env-doc order is retained inside this package; the former 1339 README is preserved verbatim in README.md.

## PA44-B owned definitions

The six file-scope definition blocks formerly pasted into upstream sources now live in BigCherry-owned include-once headers:

- `ggml/src/bc-meta-arena-flag.h`
- `ggml/src/bc-meta-arena-simple-tensors.h`
- `ggml/src/bc-meta-arena-graph.h`
- `ggml/src/bc-meta-arena-vbuffer.h`
- `ggml/src/bc-meta-arena-reserve.h`
- `ggml/src/bc-meta-arena-bind.h`

Headers are intentional. At b11552, `ggml/src/CMakeLists.txt` explicitly enumerates `ggml-alloc.c` and `ggml-backend-meta.cpp`; a new translation unit would require a CMake edit. More importantly, these definitions use file-local structs/static helpers from those translation units. Three include sites are therefore used in each upstream source so each block is included only after its local dependencies are defined. Existing edits inside upstream functions remain edits; state, flags, evidence and env docs are unchanged.

Identity proof over the composed production source: the six moved blocks contain 446 physical lines / 403 non-blank lines. Comparing the multiset of non-blank, whitespace-stripped lines across all touched and created files before vs after this ownership refactor gives moved 403/403, removed-only 0, and added-only 6: the six `#include` lines.
