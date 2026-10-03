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

- `block08.diff` — the upstream `rdna-boosts` block 08 commit touching 10
  files under `ggml/src/ggml-cuda/`, spanning two themes:
  - **Fused-core prefill kernels**: a new/expanded fused prefill path in
    `mul_mat_vec_q` (`mmvq.cu`, +466 lines in a single hunk) plus
    warp/row-per-block calc and fused-add changes; new `l2_norm`/`group_norm`
    kernels (`norm.cu`); a new gated unary-op kernel (`unary.cu`;
    `unary.cuh` adds `ggml_cuda_op_xielu`); RoPE/set-rows fusion and the
    `ggml_cuda_try_fuse` expansion in `ggml-cuda.cu`; attention kernel
    selection (`fattn.cu`) + tile switch (`fattn-tile.cuh`); declarations in
    `mmvq.cuh`/`norm.cuh` (`ggml_cuda_op_rms_norm_fused_add`).
  - **GPU bit-identity**: `common.cuh` adds a per-graph cache of quantized
    Q8_1 matmul inputs — the decode `mmvq` launcher quantizes `src1` to Q8_1
    before every matmul; matmuls sharing the same `src1` data (e.g. the
    qkv/z/alpha/beta projections of one layer, or views of the same tensor)
    quantize once and reuse the result, keyed by view root + data pointer +
    quantize layout — making decode mmvq reproducible/bit-identical; plus
    `ggml_cuda_mm_fusion_args_host`/`_device` struct changes.

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
