# rd25-block08-review: upstream rdna-boosts block 08 under review

Plan item: RD25
Status: active
Owner: RD25 (patching-patch-system)
Question state: open

## Question

What does upstream `rdna-boosts` block 08 (commit `5efcd85f`, "fused-core
prefill kernels and GPU bit-identical") change, and does it warrant
adoption/rebase for RD25? This topic holds the upstream diff under review —
not a BigCherry patch.

## Inputs

- `block08.diff` — the upstream `rdna-boosts` block 08 commit touching
  `ggml/src/ggml-cuda/common.cuh` (per-graph cache of quantized Q8_1 matmul
  inputs: the decode `mmvq` launcher quantizes `src1` to Q8_1 before every
  matmul; matmuls sharing the same `src1` data quantize once and reuse the
  result, keyed by view root + data pointer + quantize layout).

## Outputs

The reviewed upstream diff itself (no generated output).

## Runtime

GPU required: no
Real compilation required: no
Mutates canonical BigCherry state: no

## Safety

- A read-only review artifact of an upstream diff; not production tooling,
  not a BigCherry patch, and not evidence authority.

## Disposition

Retained (TRANSITIONAL) as the RD25 upstream block 08 review artifact;
diagnostic review input, not a maintained tool.
