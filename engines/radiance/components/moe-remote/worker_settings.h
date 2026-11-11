#pragma once
// The one immutable, validated configuration of this helper process.
// Installed at entry before queue setup, model access or HIP; deep helper
// call sites use settings() and abort if accessed before installation.
#include "../serving/v2/helper_settings.h"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace r3_serving {
inline std::unique_ptr<const bc::serving::HelperSettings> configured;
inline void install(bc::serving::HelperSettings value) {
    if (configured) throw std::logic_error("helper settings installed twice");
    configured = std::make_unique<const bc::serving::HelperSettings>(std::move(value));
}
inline const bc::serving::HelperSettings& settings() {
    if (!configured) {
        std::fputs("r3_moe_worker: helper settings read before initialization\n", stderr);
        std::abort();
    }
    return *configured;
}
} // namespace r3_serving
