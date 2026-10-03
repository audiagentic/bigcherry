# 1290_vulkan_allreduce_host_f32

**Status:** evaluated
**Plan item:** PRVP03

## What it does

Exposes `ggml_backend_comm_init` / `comm_allreduce_tensor` / `comm_free` from the Vulkan backend
registry so the meta (tensor-split) backend can call a Vulkan-owned AllReduce. Phase 0 reference
provider: enabled only with `BIGCHERRY_VK_ALLREDUCE=host-f32`; contiguous F32 only, synchronous host
reduction in fixed rank order; anything else returns false and meta's stock fallback handles it. With the
env unset, `comm_init` returns null and behaviour is stock. Activation marker (with
`BIGCHERRY_PATCH_TRACE=1`): `BIGCHERRY_PATCH_HIT patch=1290_vulkan_allreduce_host_f32 path=host-f32`.
