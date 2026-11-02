# 1209_rd22_integrated_gpu_host_buffer_backout

## Summary

Forces integrated=false inside ggml_cuda_init()'s GGML_USE_HIP branch, backing out upstream PR #24233's restoration of prop.integrated on HIP, which enabled a zero-copy UMA host-buffer path on APUs.

## Outcome

Recorded state: `superseded` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

The patch package's [SUMMARY.md](SUMMARY.md) records the rationale, scope, and available evidence. See [patch.toml](patch.toml) for the authoritative lifecycle state and dependencies.
