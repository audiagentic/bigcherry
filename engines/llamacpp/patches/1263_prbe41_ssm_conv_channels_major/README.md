# 1263 (PRBE41 / AMD-SSM-001): channels-major SSM_CONV input

## Scope

Ports nasone commit `33611a98a53af8a327f4a0e01a42631eb5fcd576` (AMD):
`ggml_ssm_conv` gains a channels-major input mode (CPU and CUDA/HIP; other
backends reject it so the scheduler falls back to CPU), and the Qwen3.5/3.6
delta-net graph (`qwen35`, `qwen35moe`, `qwen3next`) feeds it directly
instead of transposing the conv input. Generated with
`bigcherry.patch.port_diff`; `delta-net-base.cpp` hand-merged onto b11126
(the fork's pre-image differed); Metal's rejection not ported.

The recurrent conv-state memory layout changes (time-major to
channels-major): states saved without the patch cannot be loaded with it.

## Validation

Contract `PRBE41-SSM-CONV-CHANNELS-MAJOR`, producer `prbe41`: full-vocab
backend reference on `tierA-qwen4b-q6k`, activation marker
`BIGCHERRY_PATCH_HIT patch=1263_prbe41 path=ssm_conv_channels_major`, pp512
prefill positive, tg128 control on `tierM-gptoss20b-q6k`. The fork's
test-backend-ops SSM_CONV channels-major cases are added too.

## Status

untested.
