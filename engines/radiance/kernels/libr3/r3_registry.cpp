// libr3's kernel table: libr4d's own rows, minus the ones whose kernels are not built for this card.
//
// libr4d's declaration sources (the row table, shapes, layouts, fusion claims) are compiled into libr3 unchanged,
// with their registry exports renamed to r4d_up_*. A unit libr3 does not build yet leaves its entry points
// undefined; the build gives each a stub in one section (r3_gen_stubs.py), and a row whose launch lands in that
// section is left out here. So the engine and rad-kbench see exactly the rows that have a kernel behind them, and a
// unit moves from "absent" to "present" by coming off exclude.txt, with no table to edit.
#include "rad_abi.h"

#include <vector>

extern "C" {
int r4d_up_kernel_count(void);
const RadKernelInfo* r4d_up_kernel_at(int i);
int r4d_up_kernel_concurrent(int i);
const RadPluginInfo* r4d_up_plugin_info(void);
extern const char r3_stub_begin[];
extern const char r3_stub_end[];
}

namespace {

bool is_stub(const void* fn) {
    const char* p = static_cast<const char*>(fn);
    return p >= r3_stub_begin && p < r3_stub_end;
}

// upstream index of every row libr3 offers, in upstream order (priority ties break on declaration order)
const std::vector<int>& live() {
    static const std::vector<int> rows = [] {
        std::vector<int> out;
        const int n = r4d_up_kernel_count();
        for (int i = 0; i < n; ++i) {
            const RadKernelInfo* k = r4d_up_kernel_at(i);
            if (k && k->launch && !is_stub(reinterpret_cast<const void*>(k->launch))) out.push_back(i);
        }
        return out;
    }();
    return rows;
}

}  // namespace

extern "C" const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info = {
        RAD_PLUGIN_KERNEL, "libr3", "0.0.1",
        "libr4d's kernels built for RDNA3 through a compatibility layer; units not ported yet are absent",
        R3_TARGETS,
    };
    return &info;
}

extern "C" int rad_kernel_count(void) { return static_cast<int>(live().size()); }

extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    const std::vector<int>& rows = live();
    return i >= 0 && i < static_cast<int>(rows.size()) ? r4d_up_kernel_at(rows[i]) : nullptr;
}

extern "C" int rad_kernel_concurrent(int i) {
    const std::vector<int>& rows = live();
    return i >= 0 && i < static_cast<int>(rows.size()) ? r4d_up_kernel_concurrent(rows[i]) : 0;
}

// For tools and the log: how many of libr4d's rows are present.
extern "C" int r3_rows_present(void) { return static_cast<int>(live().size()); }
extern "C" int r3_rows_upstream(void) { return r4d_up_kernel_count(); }
