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

## 2026-10-11 qualification caveat (TRVP16 / BCOP120)

Pinned meta fallback zero-fills ranks whose final node lacks `GGML_TENSOR_FLAG_COMPUTE`; this 1290 host provider reads every rank unconditionally. Do **not** claim D=3/D=4 or uneven/zero-slice correctness from the dual-XTX screen. PRVP03 must fail closed before writes on inactive ranks (or prove an equivalent zero-source/zero-destination implementation). Returning false after partial output mutation is unsafe because meta immediately executes its fallback. Upstream [llama.cpp #25051](https://github.com/ggml-org/llama.cpp/pull/25051) is open and contains a separate mapped-host/timeline Vulkan implementation; evaluate it before authoring another transport. No new build or GPU result.
