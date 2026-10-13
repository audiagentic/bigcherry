// libr3 kernel table: upstream gfx11 rows plus opt-in host helpers.
#include "rad_abi.h"

#include <vector>

extern "C" {
int r4d_up_kernel_count(void);
const RadKernelInfo* r4d_up_kernel_at(int i);
int r4d_up_kernel_concurrent(int i);
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
        "libr4d gfx11 kernels plus opt-in host helpers for non-rank devices", R3_TARGETS,
    };
    return &info;
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
extern "C" int r3_rows_present(void) { return static_cast<int>(live().size()); }
extern "C" int r3_rows_upstream(void) { return r4d_up_kernel_count(); }
