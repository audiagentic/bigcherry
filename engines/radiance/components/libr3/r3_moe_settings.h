#pragma once
#include "bc_helper_config.h"
#include <cstdlib>

// The helper installs its immutable configuration before any callback. Only an
// engine process reaches these environment reads; RSC07 owns their migration.
inline BcR4dHelperConfig r3_moe_settings() {
    if (const auto* c = bc_r4d_helper_config()) return *c;
    const char* native = std::getenv("R3_MOE_NATIVE");
    const char* fp16 = std::getenv("R3_MOE_FP16");
    const char* ahalf = std::getenv("R3_MOE_AHALF");
    return {sizeof(BcR4dHelperConfig),
            uint32_t(native && std::atoi(native) != 0),
            uint32_t(fp16 && std::atoi(fp16) != 0),
            uint32_t(ahalf && std::atoi(ahalf) != 0),
            0u, 0u};   // the timing-only probe (R3_MOE_PROBE) is the helper config's, never the environment's
}
