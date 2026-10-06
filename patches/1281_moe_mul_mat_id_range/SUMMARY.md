# 1281_moe_mul_mat_id_range

**Status:** validated
**Plan item:** MET02

Kind: enhancement, a new ggml primitive. Nothing uses it yet; ordinary `ggml_mul_mat_id` is unchanged.

`ggml_mul_mat_id_range(ctx, as, b, ids, id_base)` is MUL_MAT_ID over a tensor that holds only the experts
`[id_base, id_base + as->ne[2])` of a larger set. `ids` stay global. A lane whose id is in the range is computed with
the local expert; a lane whose id is outside it is an exact +0 and reads no expert weight. A tier graph (MET03) and
whole-expert parallelism (MET04) are sums of such ops, one per device.

Phase A is the semantic primitive only: constructor and accessors, the CPU implementation, and a reference test
(`tests/test-mul-mat-id-range.cpp`). The HIP backend refuses the range variant in `supports_op`, so the scheduler
runs it on the CPU and no global id reaches a GPU kernel. Phase B (HIP translation at the existing grouping) and
phase C (compact dispatch) follow in MET02's order.

## Evidence

- Offline mechanics test: pending.
- Reference test on Brutus (CPU): pending.
