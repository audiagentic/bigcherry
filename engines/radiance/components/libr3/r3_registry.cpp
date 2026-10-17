// libr3 kernel table: upstream gfx11 rows plus opt-in host helpers. The remote-MoE mailbox rows live in
// their own plugin, libr3moe (helper/moe_mailbox_plugin.cpp).
#include "rad_abi.h"

#include <vector>

extern "C" {
int r4d_up_kernel_schema_count(void);
const RadOpSchema* r4d_up_kernel_schema_at(int i);
int r4d_up_kernel_count(void);
const RadKernelInfo* r4d_up_kernel_at(int i);
int r4d_up_kernel_concurrent(int i);
int r4d_up_kernel_collective(int i);
const RadPluginInfo* r4d_up_plugin_info(void);
extern const char r3_stub_begin[];
extern const char r3_stub_end[];
int r3_ngram_enabled(void);
const RadKernelInfo* r3_ngram_row(void);
}

namespace {
bool is_stub(const void* fn) {
    const char* p = static_cast<const char*>(fn);
    return p >= r3_stub_begin && p < r3_stub_end;
}
struct LiveRow { int upstream = -1; const RadKernelInfo* custom = nullptr; };
const std::vector<LiveRow>& live() {
    static const std::vector<LiveRow> rows = [] {
        std::vector<LiveRow> out;
        const int n = r4d_up_kernel_count();
        for (int i = 0; i < n; ++i) {
            const RadKernelInfo* k = r4d_up_kernel_at(i);
            if (k && k->launch && !is_stub(reinterpret_cast<const void*>(k->launch))) out.push_back({i, nullptr});
        }
        if (r3_ngram_enabled()) out.push_back({-1, r3_ngram_row()});
        return out;
    }();
    return rows;
}
}  // namespace

extern "C" const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info = {
        RAD_PLUGIN_KERNEL, "libr3", "0.0.1",
        "libr4d gfx11 kernels plus opt-in host helpers", R3_TARGETS,
    };
    return &info;
}
extern "C" int rad_kernel_schema_count(void) {
    return r4d_up_kernel_schema_count();
}
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) {
    return r4d_up_kernel_schema_at(i);
}
extern "C" int rad_kernel_count(void) { return static_cast<int>(live().size()); }
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    const auto& rows = live();
    if (i < 0 || i >= static_cast<int>(rows.size())) return nullptr;
    return rows[i].custom ? rows[i].custom : r4d_up_kernel_at(rows[i].upstream);
}
extern "C" int rad_kernel_concurrent(int i) {
    const auto& rows = live();
    if (i < 0 || i >= static_cast<int>(rows.size()) || rows[i].custom) return 0;
    return r4d_up_kernel_concurrent(rows[i].upstream);
}
// ABI 17 (rad_abi.h, RadKernelCollectiveFn): the engine indexes this library's live table, so the answer is mapped
// from the live index to the upstream index like rad_kernel_concurrent. 1 is forwarded as 1; 2 (the row ends the
// link hold itself through link_gate) is clamped to 1: libr3's gfx11 forms do not decrement the gate, and a wrong 2
// leaves it odd so every yielding DMA list on that card waits forever. Answering 1 is always safe.
extern "C" int rad_kernel_collective(int i) {
    const auto& rows = live();
    if (i < 0 || i >= static_cast<int>(rows.size()) || rows[i].custom) return 0;
    return r4d_up_kernel_collective(rows[i].upstream) != 0 ? 1 : 0;
}
extern "C" int r3_rows_present(void) { return static_cast<int>(live().size()); }
extern "C" int r3_rows_upstream(void) { return r4d_up_kernel_count(); }
