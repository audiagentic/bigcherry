// libr3moe kernel table: the remote-MoE mailbox rows (publish, merge, shadow, mask-ref) and the hook that
// owns the shared segment. A plugin of its own, built for gfx1100 and gfx1201, so the same hand-off runs
// beside libr3 on the XTX ranks and beside libr4d on an R9700 engine (RAD02).
#include "rad_abi.h"

extern "C" {
int r3_moe_mailbox_enabled(void);
int r3_moe_mailbox_schema_count(void);
const RadOpSchema* r3_moe_mailbox_schema_at(int i);
const RadKernelInfo* r3_moe_mailbox_row(int i);
}

namespace {
int live_count() {
    static const int n = [] {
        int k = 0;
        if (r3_moe_mailbox_enabled())
            while (r3_moe_mailbox_row(k)) ++k;
        return k;
    }();
    return n;
}
}  // namespace

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info = {
        RAD_PLUGIN_KERNEL, "libr3moe", "0.0.1",
        "remote-MoE mailbox: GPU publish/merge rows for an out-of-process expert helper", R3MOE_TARGETS,
    };
    return &info;
}
extern "C" int rad_kernel_schema_count(void) { return r3_moe_mailbox_schema_count(); }
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) { return r3_moe_mailbox_schema_at(i); }
extern "C" int rad_kernel_count(void) { return live_count(); }
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    return i >= 0 && i < live_count() ? r3_moe_mailbox_row(i) : nullptr;
}
extern "C" int rad_kernel_concurrent(int) { return 0; }
