// bigcherry: telemetry-only SPLIT_REDUCE observation (HI58).
//
// This API records what the existing multi-device path actually did. It does
// not select a provider, alter fallback behavior, or participate in any
// dispatch/replay identity.
#pragma once

#include <stddef.h>

#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH)

struct ggml_tensor;
void ggml_hip_reduce_telemetry_provider(
        const int * devices,
        size_t device_count,
        ggml_tensor ** tensors,
        const char * requested_provider,
        bool provider_succeeded);

void ggml_hip_reduce_telemetry_fallback(
        const int * devices,
        size_t device_count,
        ggml_tensor ** tensors,
        const char * requested_provider,
        const char * handoff,
        size_t fallback_depth);

#endif
