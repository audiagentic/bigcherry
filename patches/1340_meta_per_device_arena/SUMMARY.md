# 1340_meta_per_device_arena

**Status:** validated
**Plan item:** MSM01/MSM02

## What it does

Single package for Meta per-device compute arenas plus the `BIGCHERRY_META_MEM` memory-report diagnostic formerly packaged as 1339. The arena mechanism remains default-on with `BIGCHERRY_META_PER_DEVICE_ARENA=0` as the native/common-size control; `BIGCHERRY_META_MEM=1` remains diagnostic-only.

## PA44-E packaging proof

1339 and 1340 were adjacent and 1340 already required 1339. Their edit/env-doc order is retained inside this package; the former 1339 README is preserved verbatim in README.md.
