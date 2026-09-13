# 1258: RD12 paired MUL_MAT correctness test case

Diagnostic support for the `RD12-PAIRED-MMVQ-DUAL` correctness producer.

RD12 cannot be exercised by the ordinary `test-backend-ops --test-file`
path: one test-file line constructs one `MUL_MAT`, while patch 1205's
production selector requires two distinct adjacent `MUL_MAT` nodes with
different `src0` weights and the exact same `src1` activation tensor.

This patch registers one whole-graph Q6_K decode case:

- weight shape: `[2560, 1024]`
- shared activation: `[2560, 1]`
- K output: `[1024, 1]`
- V output: `[1024, 1]`
- terminal ADD keeps both projection nodes live
- `fusion_test_nodes()` compares `rd12_k_out` and `rd12_v_out`
  independently

It is diagnostic infrastructure only. It is present identically in the
RD12 control and subject compositions and carries no Experiment Contract
binding of its own.

Required support:

- `1222_hi67_deterministic_test_backend_ops_seed`
- `1223_hi67_machine_readable_correctness_metrics`

Patch 1236 is not required because this graph contains no `MUL_MAT_ID`
routing tensor.
