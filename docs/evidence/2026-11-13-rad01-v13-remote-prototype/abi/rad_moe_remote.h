/* rad_moe_remote.h -- optional whole-expert compute hand-off for routed MoE blocks. */
#pragma once
#include <stdint.h>
#include "rad_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif
#define RAD_MOE_REMOTE_ABI 1u
typedef void* RadMoeRemoteHandle;
typedef struct RadMoeRemoteRoute {
    int32_t token;
    int32_t expert;
    uint16_t weight_bf16;
    uint16_t reserved;
} RadMoeRemoteRoute;
typedef struct RadMoeRemoteHooks {
    uint32_t abi_version;
    void* user;
    RadMoeRemoteHandle (*submit)(void* user, int layer, int rank,
                                 int64_t rows, int64_t n_embd,
                                 const RadMoeRemoteRoute* route, int64_t n_route,
                                 const void* activation, int64_t activation_stride_bytes);
    const void* (*wait)(void* user, RadMoeRemoteHandle handle);
} RadMoeRemoteHooks;
/* Register before architecture declaration. An already-loaded plugin may call this directly.
 * R3_MOE_REMOTE_SO may instead name a .so exporting
 *   const RadMoeRemoteHooks* rad_moe_remote_hooks(void)
 * which the core loads during architecture declaration. */
int rad_moe_remote_register(const RadMoeRemoteHooks* hooks);
int rad_moe_remote_wanted(int64_t step_rows);
RadMoeRemoteHandle rad_moe_remote_submit(RadCtx* c, int layer,
                                         int64_t step_rows, int64_t rows, int64_t n_embd,
                                         int64_t top_k, int64_t expert_base,
                                         const rad_weight* gate_up, const rad_weight* down,
                                         int64_t n_expert,
                                         int32_t* ids, const uint16_t* weight_bf16,
                                         const void* activation, int64_t activation_stride_bytes);
/* Waits through the registered hook, copies its BF16 partial into dst on the engine's current
 * stream, and drains that stream before returning so the hook may reuse its returned storage. */
int rad_moe_remote_wait_into(RadCtx* c, RadMoeRemoteHandle handle, void* dst, int64_t bytes);
#ifdef __cplusplus
}
#endif
