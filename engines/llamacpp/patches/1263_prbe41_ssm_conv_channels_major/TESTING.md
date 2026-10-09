# Testing — PRBE41 SSM_CONV channels-major

## Lane applicability
Changes `ggml_ssm_conv` to support channels-major input and makes the Qwen delta-net/GDN graph feed convolution without the physical transpose; CPU and CUDA/HIP implement the layout. This can affect gfx1100 `-sm tensor` Qwen3.8-27B-Q8_0 because the model is hybrid GDN/SSM+attention: prefill is the primary target and recurrent/MTP decode can also traverse SSM_CONV. **Applicable to this lane: yes.** Recurrent conv-state layout changes, so do not reuse state across control/subject builds.

## Hardware-free
- `PYTHONPATH=tools python -m bigcherry patch-rebase-check --source bigcherry --focal-overlay 1263_prbe41_ssm_conv_channels_major`
- Unit mechanics: apply/idempotence; missing anchor fails closed; channels-major CPU/CUDA backend-op cases; graph asserts Qwen delta-net selects channels-major and removes the redundant transpose while unsupported backends reject/fallback.

## Activation
With `BIGCHERRY_PATCH_TRACE=1`, require exact marker: `BIGCHERRY_PATCH_HIT patch=1263_prbe41 path=ssm_conv_channels_major`. Marker must appear on subject and not control during the measured workload.

## Correctness
Dual 7900 XTX/gfx1100, `-sm tensor`, Qwen3.8-27B-Q8_0, MTP `n_max=4` (verify batch 5 columns). Compare control/subject MTP logprobs; max abs diff must be <= `5e-4`. Report any greedy-token divergence. Require MTP draft-acceptance parity against the standard server-bench control reference `0.90101` (record accepted/drafted counts and ratio).

## Performance
Run order-balanced `bigcherry ab-benchmark --server-config` A/B with patch presence as the only variable; report `pp1024`, `pp4096`, `tg512`, `tg2048`, pair order/raw observations/deltas and Mann-Whitney. Expect strongest opportunity in prefill from transpose removal; do not infer benefit from activation alone.

## Promotion
Promotion requires a contract campaign for `PRBE41-SSM-CONV-CHANNELS-MAJOR` producing `patch-verify-evidence` validated-evidence with architecture/workload identity, activation, correctness and statistical performance evidence. Ad-hoc A/B is diagnostic only and is not promotion evidence.
