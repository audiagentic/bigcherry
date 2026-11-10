#pragma once
#include "bc_helper_config.h"
#include "../serving/bc_engine_settings.h"

// Helper settings are installed before callbacks. The engine reads the typed
// manifest snapshot exported by its core. Neither path reads R3_* variables.
inline BcR4dHelperConfig r3_moe_settings() {
    if (const auto* c = bc_r4d_helper_config()) return *c;
    const auto s = bc_engine_settings_snapshot();
    return {sizeof(BcR4dHelperConfig), s.moe_native, s.moe_fp16, s.moe_ahalf, 0u, 0u};
}
