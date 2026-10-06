# 1342_fusion_bisect

**Status:** untested
**Plan item:** FKE01

Kind: diagnostic, variable `BIGCHERRY_FUSION_SKIP_OPS` (unset by default).

The GPU backend has a single switch for all kernel fusion. This patch lets fusion be switched off per family,
identified by the op of the fusion's first node (`BIGCHERRY_FUSION_SKIP_OPS=MUL_MAT_ID,RMS_NORM,...`), and counts
the fusions taken per family at exit under `BIGCHERRY_META_MEM=1`. It exists to find which fused launch is not
bit-identical to the node sequence it replaces and what each family is worth in speed. Nothing changes without the
variables.

## Evidence

- Offline mechanics test: `tools/tests/patch/test_1342_fusion_bisect.py`. Hardware: pending.
