#pragma once
#include "bc_helper_config.h"
#include <cstdlib>

// The helper installs its immutable configuration before any callback. Only an
// engine process reaches these environment reads; RSC07 owns their migration.
inline BcR4dHelperConfig r3_moe_settings() {
    if (const auto* c = bc_r4d_helper_config()) return *c;
    // An unset switch takes its declared default (component.toml: "1" for all three, the validated forms the helpers run);
    // "0" turns it off. The engine used to read an unset switch as off, so a gfx1100 engine rank (a TP layout) ran the
    // compatibility MoE path while its helpers ran the native one.
    // (Each variable is read by name, std::getenv("..."), so the declared-settings check finds it.)
    const auto on = [](const char* v) { return uint32_t(!v || std::atoi(v) != 0); };
    return {sizeof(BcR4dHelperConfig),
            on(std::getenv("R3_MOE_NATIVE")),
            on(std::getenv("R3_MOE_FP16")),
            on(std::getenv("R3_MOE_AHALF")),
            0u, 0u};   // the timing-only probe (R3_MOE_PROBE) is the helper config's, never the environment's
}
