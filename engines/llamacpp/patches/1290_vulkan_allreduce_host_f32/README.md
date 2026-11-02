# 1290_vulkan_allreduce_host_f32

## Summary

Exposes `ggml_backend_comm_init` / `comm_allreduce_tensor` / `comm_free` from the Vulkan backend registry so the meta (tensor-split) backend can call a Vulkan-owned AllReduce.

## Outcome

Recorded state: `evaluated` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

The patch package's [SUMMARY.md](SUMMARY.md) records the rationale, scope, and available evidence. See [patch.toml](patch.toml) for the authoritative lifecycle state and dependencies.
