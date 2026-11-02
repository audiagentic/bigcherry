#pragma once
#include "../serving/serving.h"
#include "bc_helper_config.h"
#include <dlfcn.h>
#include <cstdio>

namespace r3_serving {
inline bool configure_plugin(void* library, const bc::serving::HelperSettings& s) {
    using Configure = int (*)(const BcR4dHelperConfig*);
    dlerror();
    const auto configure = reinterpret_cast<Configure>(dlsym(library, "bc_r4d_configure_helper"));
    const char* error = dlerror();
    if (!configure || error) {
        std::fprintf(stderr, "r3_moe_worker: plugin lacks bc_r4d_configure_helper: %s\n",
                     error ? error : "null");
        return false;
    }
    const BcR4dHelperConfig config{sizeof(BcR4dHelperConfig), s.native, s.fp16, s.ahalf,
                                 s.moe_probe, s.refusal_trace};
    if (configure(&config) != 0) {
        std::fputs("r3_moe_worker: plugin refused helper configuration\n", stderr);
        return false;
    }
    return true;
}
} // namespace r3_serving
