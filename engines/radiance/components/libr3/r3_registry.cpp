// libr3 kernel table: upstream gfx11 rows. The n-gram gather is its own plugin, libr3ngram (components/ngram-helper). The remote-MoE mailbox rows live in
// their own plugin, libr3moe (worker/moe_mailbox_plugin.cpp).
#include "rad_abi.h"

#include <algorithm>
#include <array>
#include <cstring>
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
}

namespace {
bool is_stub(const void* fn) {
    const char* p = static_cast<const char*>(fn);
    return p >= r3_stub_begin && p < r3_stub_end;
}
struct LiveRow { int upstream = -1; const RadKernelInfo* custom = nullptr; };

// The gfx11-only IQ6R M=64 crossover. The upstream narrow row has M<=64
// and priority 60; its tiled peer permits any M but priority 40. At M=64
// model-shape cold tests, the narrow path is >4x slower on both XTXs.
// Publish a new higher-priority row ONLY when the exact upstream ABI and
// legality predicates match. The original narrow row is not changed: a
// small N that is legal at 16 columns but not 128 is never orphaned.
struct IqrM64Overlay {
    RadKernelInfo row{};
    std::array<RadConstraint, 5> constraints{};
    bool enabled = false;
};
IqrM64Overlay& iqr_m64() {
    static IqrM64Overlay overlay;
    return overlay;
}
// A second disjoint row preserves the individually proven M=64
// implementation while accelerating M=17..63 ONLY at measured
// K=5120, N=5120/10240. M<=16 and unmeasured K/N retain narrow.
struct IqrEarlyOverlay {
    RadKernelInfo row{};
    std::array<RadConstraint, 8> constraints{};
    bool enabled = false;
};
IqrEarlyOverlay& iqr_early() {
    static IqrEarlyOverlay overlay;
    return overlay;
}
bool safe_iqr_m64_overlay(const RadKernelInfo* narrow, const RadKernelInfo* tiled) {
    if (!narrow || !tiled || !narrow->op || !tiled->op ||
        !narrow->constraints || !tiled->constraints ||
        !tiled->launch || is_stub(reinterpret_cast<const void*>(tiled->launch)) ||
        !tiled->scratch || is_stub(reinterpret_cast<const void*>(tiled->scratch)) ||
        !tiled->layout || is_stub(reinterpret_cast<const void*>(tiled->layout)) ||
        !tiled->relayout || is_stub(reinterpret_cast<const void*>(tiled->relayout)) ||
        narrow->n_constraints != 5 || tiled->n_constraints != 4 ||
        narrow->priority < tiled->priority || narrow->priority >= 1000000 ||
        std::strcmp(narrow->op, "gemm_nt_q") ||
        std::strcmp(tiled->op, "gemm_nt_q")) return false;
    // Preserve every upstream tiled legality check; only ADD M == 64.
    const RadConstraint* n = narrow->constraints;
    const RadConstraint* t = tiled->constraints;
    return n[0].key && !std::strcmp(n[0].key, "M") &&
           n[0].op == RAD_C_LE && n[0].ival == 64 &&
           t[0].key && !std::strcmp(t[0].key, "K") &&
           t[0].op == RAD_C_DIV && t[0].ival == 256 &&
           t[1].key && !std::strcmp(t[1].key, "N") &&
           t[1].op == RAD_C_DIV && t[1].ival == 128 &&
           t[2].key && !std::strcmp(t[2].key, "group") &&
           t[2].op == RAD_C_EQ && t[2].ival == 128 &&
           t[3].key && !std::strcmp(t[3].key, "dtype") &&
           t[3].op == RAD_C_IN && t[3].sval &&
           !std::strcmp(t[3].sval, "i8a8 i8a8h");
}

const std::vector<LiveRow>& live() {
    static const std::vector<LiveRow> rows = [] {
        std::vector<LiveRow> out;
        const int n = r4d_up_kernel_count();
        int narrow_idx = -1, tiled_idx = -1;
        for (int i = 0; i < n; ++i) {
            const RadKernelInfo* k = r4d_up_kernel_at(i);
            if (!k || !k->launch || is_stub(reinterpret_cast<const void*>(k->launch))) continue;
            out.push_back({i, nullptr});
            if (k->name && !std::strcmp(k->name, "gemm_iqr_nt_m64")) narrow_idx = i;
            if (k->name && !std::strcmp(k->name, "gemm_iqr_tiled")) tiled_idx = i;
        }
        if (narrow_idx >= 0 && tiled_idx >= 0) {
            const RadKernelInfo* narrow = r4d_up_kernel_at(narrow_idx);
            const RadKernelInfo* tiled = r4d_up_kernel_at(tiled_idx);
            if (safe_iqr_m64_overlay(narrow, tiled)) {
                IqrM64Overlay& patch = iqr_m64();
                std::copy_n(tiled->constraints, 4, patch.constraints.begin());
                patch.constraints[4] = {"M", RAD_C_EQ, 64, nullptr};
                patch.row = *tiled;
                patch.row.name = "gemm_iqr_tiled_m64_gfx11";
                patch.row.priority = narrow->priority + 1;
                patch.row.constraints = patch.constraints.data();
                patch.row.n_constraints = static_cast<int>(patch.constraints.size());
                out.push_back({tiled_idx, &patch.row});
                patch.enabled = true;
                IqrEarlyOverlay& early = iqr_early();
                std::copy_n(tiled->constraints, 4, early.constraints.begin());
                early.constraints[4] = {"M", RAD_C_GE, 17, nullptr};
                early.constraints[5] = {"M", RAD_C_LE, 63, nullptr};
                // Only geometry verified by the VTP09 27B FFN crossover.
                // Small K/N regress; keep the original narrow row there.
                early.constraints[6] = {"K", RAD_C_EQ, 5120, nullptr};
                early.constraints[7] = {"N", RAD_C_IN, 0, "5120 10240"};
                early.row = *tiled;
                early.row.name = "gemm_iqr_tiled_m17_63_gfx11";
                early.row.priority = narrow->priority + 1;
                early.row.constraints = early.constraints.data();
                early.row.n_constraints = static_cast<int>(early.constraints.size());
                out.push_back({tiled_idx, &early.row});
                early.enabled = true;
            }
        }
        return out;
    }();
    return rows;
}
}  // namespace

// Explicit queryable proof: never infer this dispatch override from mere
// presence of gemm_iqr_tiled in the plugin binary.
extern "C" int r3_iqr_m64_tiled_override(void) {
    (void)live();
    return iqr_m64().enabled ? 1 : 0;
}
extern "C" int r3_iqr_m17_63_tiled_override(void) {
    (void)live();
    return iqr_early().enabled ? 1 : 0;
}
extern "C" const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info = {
        RAD_PLUGIN_KERNEL, "libr3", "0.0.1",
        "libr4d gfx11 kernels", R3_TARGETS,
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
    if (i < 0 || i >= static_cast<int>(rows.size())) return 0;
    return r4d_up_kernel_concurrent(rows[i].upstream);
}
// ABI 17 (rad_abi.h, RadKernelCollectiveFn): the engine indexes this library's live table, so the answer is mapped
// from the live index to the upstream index like rad_kernel_concurrent. 1 is forwarded as 1; 2 (the row ends the
// link hold itself through link_gate) is clamped to 1: libr3's gfx11 forms do not decrement the gate, and a wrong 2
// leaves it odd so every yielding DMA list on that card waits forever. Answering 1 is always safe.
extern "C" int rad_kernel_collective(int i) {
    const auto& rows = live();
    if (i < 0 || i >= static_cast<int>(rows.size())) return 0;
    return r4d_up_kernel_collective(rows[i].upstream) != 0 ? 1 : 0;
}
extern "C" int r3_rows_present(void) { return static_cast<int>(live().size()); }
extern "C" int r3_rows_upstream(void) { return r4d_up_kernel_count(); }
