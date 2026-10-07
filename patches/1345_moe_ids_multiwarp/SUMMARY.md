# 1345_moe_ids_multiwarp

**Status:** untested
**Plan item:** QFP36

## What it does

For batches of 128 tokens and more, the MoE routing helper (`ggml_cuda_launch_mm_ids_helper`, `ggml-cuda/mmid.cu`)
runs 8 warps per expert block where the native helper runs one. Each warp walks its own contiguous slice of the
tokens with the native per-warp code; the warps exchange their counts once and write their rows at the right
offsets. On by default; `BIGCHERRY_MOE_IDS_MULTIWARP=0` restores the native helper.

## Why

The native helper is one warp per expert walking every token in sequence (256 dependent iterations for 512 tokens at
top-10), for each of the 512 experts. In the Flash-Next prefill profile on the production build `mm_ids_helper<10>`
is about 3.8% of all kernel time on the three target cards (run `gate0-d24576`).

## Scope

The rows of an expert stay in ascending token order, so `ids_src1` (forward and inverse form), `ids_dst` and
`expert_bounds` are byte-identical to the native helper's; nothing downstream changes. Smaller batches, the generic
top-k path, and devices whose warp size is not the compiled one use the native helper. Works on the global ids and on
1281's translated local ids (an id of INT_MAX is in no expert's list, as in the native helper).

## Activation

`BIGCHERRY_PATCH_TRACE=1` prints `BIGCHERRY_PATCH_HIT patch=1345_moe_ids_multiwarp used= experts= tokens= warps=
inverse=` once.
