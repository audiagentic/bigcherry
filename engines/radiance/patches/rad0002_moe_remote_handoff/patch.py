"""Radiance 1.7.0: GPU-lane remote whole-expert MoE hand-off, as thin hooks into one BigCherry-owned extension point.

All policy lives in files this patch creates (abi/bc_moe_remote.h, core/place/bc_moe_remote.cpp,
core/place/bc_moe_remote_place.h, arch/common/bc_moe_remote_block.h) and in the libr3moe plugin. The edits to upstream
files are one-line guarded call sites into them. The shadow compare and mask-only diagnostics are rad0004.
"""

from bigcherry.patcher import Edit, FilePatch

_BC_MOE_REMOTE_H = r"""/* bc_moe_remote.h -- BigCherry's extension point for GPU-lane remote whole-expert MoE compute.
 *
 * The engine asks four questions of it: is this step handed off (wanted), does the helper own this routed
 * expert so the planner may leave it unplaced (skip_unit), what is each layer's runtime expert-to-slot map
 * (layer_map), and is a diagnostic mode on (shadow, mask_only; diagnostic builds only). The libr3moe plugin
 * registers its two callbacks once through bc_moe_remote_register. Every function is exported by the engine
 * binary and bound by the arch plugins and libr3moe at dlopen. */
#ifndef BC_MOE_REMOTE_H
#define BC_MOE_REMOTE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Filled by libr3moe. struct_size must equal sizeof(BcMoeRemotePlugin) of the engine's header. */
typedef struct BcMoeRemotePlugin {
    uint32_t struct_size;
    /* A layer's runtime slot map (slot_of_expert[e] < 0: expert e is a protected, locally served expert). */
    void (*layer_map)(int layer, int n_expert, const int32_t* slot_of_expert);
    /* Whether a helper owns routed expert (layer, expert). */
    int (*owns_expert)(int layer, int expert);
    /* An inference burst begins (1: the engine loop has a step after being idle) or ends (0: the loop found no
     * step). Called from the engine's host scheduler loop, so it fires whether a pass is recorded or replayed;
     * a helper arms its GPU waits only inside a burst, and its queues drain to empty between bursts. */
    void (*burst)(int active);
} BcMoeRemotePlugin;
/* Register the plugin (NULL unregisters). Layer maps published before registration are replayed to it.
 * Returns 0, or -1 when struct_size does not match. */
int bc_moe_remote_register(const BcMoeRemotePlugin* plugin);
/* Runtime step gate (R3_MOE_REMOTE_MIN_ROWS / R3_MOE_REMOTE_MAX_ROWS, set by libr3moe once its mapped segment
 * is pinned and ready; an unset minimum means remote MoE is off). */
int bc_moe_remote_wanted(int64_t step_rows);
int bc_moe_remote_wanted_upto(int64_t max_step_rows);
/* Diagnostics: always 0 unless the engine is built with BC_MOE_REMOTE_DIAG (patch rad0004). A production build
 * that finds R3_MOE_WORKER_SHADOW or R3_MOE_WORKER_MASK_ONLY set answers "no remote rows" and says so once. */
int bc_moe_remote_shadow(void);
int bc_moe_remote_mask_only(void);
int bc_moe_remote_diag_build(void);
/* The engine loop calls engine_step for every step it runs and engine_idle when it finds no step; the
 * transitions between the two are forwarded to the plugin's burst callback, once each. */
void bc_moe_remote_engine_step(void);
void bc_moe_remote_engine_idle(void);
/* Block declare publishes a layer's map here; it is forwarded to the plugin and kept for replay. */
void bc_moe_remote_layer_map(int layer, int n_expert, const int32_t* slot_of_expert);
/* Placement (R3_MOE_WORKER_SKIP_OWNED): whether the engine may leave a routed expert unplaced because the
 * helper computes it and nothing local ever reads it. */
int bc_moe_remote_skip_unit(int layer, int expert, int64_t max_step_rows);
#ifdef __cplusplus
}
#endif
#endif /* BC_MOE_REMOTE_H */
"""

_BC_MOE_REMOTE_CPP = r"""/* bc_moe_remote.cpp -- policy behind abi/bc_moe_remote.h: the decode-sized step gate, the plugin registry,
 * the layer-map store and the SKIP_OWNED placement rule. */
#include "bc_moe_remote.h"
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <vector>

#ifdef BC_MOE_REMOTE_DIAG
static constexpr bool kDiagBuild = true;
#else
static constexpr bool kDiagBuild = false;
#endif

namespace {
bool env_on(const char* name) {
    const char* e = std::getenv(name);
    return e && *e && !(e[0] == '0' && e[1] == 0);
}

int64_t rows_env(const char* name) {
    const char* s = std::getenv(name);
    if (!s || !*s) return (int64_t)LLONG_MAX;
    char* end = nullptr;
    const long long n = std::strtoll(s, &end, 10);
    return end && *end == 0 && n >= 0 ? (int64_t)n : (int64_t)LLONG_MAX;
}

int64_t min_rows() {
    static const int64_t v = rows_env("R3_MOE_REMOTE_MIN_ROWS");
    return v;
}
int64_t max_rows() {
    static const int64_t v = rows_env("R3_MOE_REMOTE_MAX_ROWS");
    return v;
}

/* The diagnostic modes exist only in a diagnostic build. Asked for in a production build they must not be
 * silently ignored (that would measure the wrong thing): the extension point answers "no remote rows", so every
 * expert computes locally, and says so once. */
bool diag_refused() {
    if (kDiagBuild) return false;
    static const bool requested = env_on("R3_MOE_WORKER_SHADOW") || env_on("R3_MOE_WORKER_MASK_ONLY");
    if (!requested) return false;
    static std::atomic<bool> said{false};
    if (!said.exchange(true))
        std::fprintf(stderr, "radiance: R3_MOE_WORKER_SHADOW / R3_MOE_WORKER_MASK_ONLY need the diagnostic build "
                             "(BC_MOE_REMOTE_DIAG, patch rad0004); remote MoE hand-off is off and every expert "
                             "computes locally\n");
    return true;
}

std::mutex g_mu;
BcMoeRemotePlugin g_plugin{};   /* struct_size 0: none registered */
std::map<int, std::vector<int32_t>> g_layer_maps;
}

extern "C" int bc_moe_remote_wanted(int64_t step_rows) {
    if (diag_refused()) return 0;
    const int64_t lo = min_rows(), hi = max_rows();
    return lo != LLONG_MAX && step_rows >= lo && step_rows <= hi;
}

extern "C" int bc_moe_remote_wanted_upto(int64_t max_step_rows) {
    if (diag_refused()) return 0;
    const int64_t lo = min_rows(), hi = max_rows();
    return lo != LLONG_MAX && lo <= hi && lo <= max_step_rows;
}

extern "C" int bc_moe_remote_shadow(void) {
    static const bool v = kDiagBuild && env_on("R3_MOE_WORKER_SHADOW");
    return v;
}

extern "C" int bc_moe_remote_mask_only(void) {
    static const bool v = kDiagBuild && env_on("R3_MOE_WORKER_MASK_ONLY");
    return v;
}

extern "C" int bc_moe_remote_diag_build(void) {
    return kDiagBuild ? 1 : 0;
}

/* Layer maps are published at declare, which can run before the plugin registers (libr3moe registers when it
 * loads). Keep every published map and replay them on registration, so no layer's map is lost to that ordering. */
extern "C" int bc_moe_remote_register(const BcMoeRemotePlugin* plugin) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!plugin) {
        g_plugin = BcMoeRemotePlugin{};
        return 0;
    }
    if (plugin->struct_size != sizeof(BcMoeRemotePlugin)) return -1;
    g_plugin = *plugin;
    if (g_plugin.layer_map)
        for (const auto& [layer, slots] : g_layer_maps)
            g_plugin.layer_map(layer, (int)slots.size(), slots.data());
    return 0;
}

namespace {
std::atomic<bool> g_burst_active{false};
void forward_burst(int active) {
    void (*burst)(int) = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        burst = g_plugin.burst;
    }
    if (burst) burst(active);
}
}

/* Per-step cost is one atomic exchange; the plugin hears only the transitions. */
extern "C" void bc_moe_remote_engine_step(void) {
    if (!g_burst_active.exchange(true, std::memory_order_acq_rel)) forward_burst(1);
}

extern "C" void bc_moe_remote_engine_idle(void) {
    if (g_burst_active.exchange(false, std::memory_order_acq_rel)) forward_burst(0);
}

extern "C" void bc_moe_remote_layer_map(int layer, int n_expert, const int32_t* slot_of_expert) {
    if (n_expert <= 0 || !slot_of_expert) return;
    std::lock_guard<std::mutex> lock(g_mu);
    g_layer_maps[layer].assign(slot_of_expert, slot_of_expert + n_expert);
    if (g_plugin.layer_map) g_plugin.layer_map(layer, n_expert, slot_of_expert);
}

/* R3_MOE_WORKER_SKIP_OWNED=1: a helper-owned expert is not placed at all -- no VRAM, no host pool -- which is
 * what lets one engine rank (the R9700) hold a model whose experts it could not hold alone. Only when nothing
 * local will ever read the unit: every step is handed off (the hand-off cap covers the largest step), no
 * diagnostic recomputes owned experts locally, and the layer's runtime map gives the expert a regular slot with
 * no protected expert in the layer (those layers stay local; the helper drops ownership on exactly the same
 * test). */
extern "C" int bc_moe_remote_skip_unit(int layer, int expert, int64_t max_step_rows) {
    static const bool on = env_on("R3_MOE_WORKER_SKIP_OWNED");
    if (!on || diag_refused() || bc_moe_remote_shadow() || bc_moe_remote_mask_only()) return 0;
    if (min_rows() == (int64_t)LLONG_MAX || min_rows() > 1 || max_rows() < max_step_rows) return 0;
    int (*owns)(int, int) = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        owns = g_plugin.owns_expert;
        const auto it = g_layer_maps.find(layer);
        if (it == g_layer_maps.end() || expert < 0 || expert >= (int)it->second.size()) return 0;
        for (int32_t slot : it->second)
            if (slot < 0) return 0;
    }
    return owns && owns(layer, expert) ? 1 : 0;
}
"""

_BC_MOE_REMOTE_PLACE_H = r"""/* bc_moe_remote_place.h -- the placement and loading side of the BigCherry remote-expert extension point
 * (abi/bc_moe_remote.h). Header-only templates, so the planner and the bring-up call them with their own types
 * and need no further includes. */
#ifndef BC_MOE_REMOTE_PLACE_H
#define BC_MOE_REMOTE_PLACE_H

#include "bc_moe_remote.h"
#include "rad_core.h"

#include <cstdint>
#include <vector>

namespace rad {

/* A routed expert the remote helper computes, when the engine never reads it (bc_moe_remote_skip_unit): no
 * slot, no pool and no copy. It stays on the mapped tier as a placeholder the loader publishes without
 * reading; the zero-copy site keeps it out of the host-site and mixed-operand logic. Returns whether the unit
 * was taken (the planner then leaves it alone). */
template <class MU>
inline bool bc_moe_remote_pin_unit(MU& mu, int64_t max_tok) {
    if (mu.layer < 0 || mu.expert < 0 || !bc_moe_remote_skip_unit(mu.layer, mu.expert, max_tok)) return false;
    mu.tier = Tier::Mapped;
    mu.movable = false;
    mu.site = Site::DeviceZeroCopy;
    mu.reason = "computed by the remote MoE helper; never read on this rank";
    return true;
}

/* The pre-plan budget makes the same call pin_unit does: a helper-owned expert costs this rank nothing. The
 * step-cap condition is the planner's and bring-up's to enforce (0 skips it here); a cap below the largest
 * step only makes this budget generous, and the plan then reports what did not fit. */
template <class U>
inline void bc_moe_remote_mark_footprint(std::vector<U>& units) {
    for (U& mu : units)
        if (mu.layer >= 0 && mu.expert >= 0 && bc_moe_remote_skip_unit(mu.layer, mu.expert, 0)) mu.mapped = true;
}

/* Bring-up: publish a unit the planner left on the mapped tier as a never-read placeholder (the container
 * address, never dereferenced) instead of loading it. Its stored form never matters because this rank never
 * reads it, so the byte-count and direct-read checks do not apply. Returns whether it was published. */
template <class W, class E, class F, class M, class I>
inline bool bc_moe_remote_publish_placeholder(int64_t max_tok, W& w, const E* e, const F& file, M& mover, I index) {
    if (!e || w.tier != Tier::Mapped || w.decl.group.layer < 0 || w.decl.group.expert < 0 ||
        !bc_moe_remote_skip_unit(w.decl.group.layer, w.decl.group.expert, max_tok))
        return false;
    w.ptr = (void*)file.data(file.plane(*e, w.sel[0]));
    w.host_ptr = w.ptr;
    mover.publish_mapped(index, w.ptr);
    return true;
}

}  /* namespace rad */

#endif /* BC_MOE_REMOTE_PLACE_H */
"""

_BC_MOE_REMOTE_BLOCK_H = r"""/* bc_moe_remote_block.h -- the MoE block's side of the BigCherry remote-expert extension point
 * (abi/bc_moe_remote.h). MoeFP8 holds one BcMoeRemote and calls it at five places; HyperConn asks
 * bc_moe_remote_hc_open. Everything else (what is handed off, when, and how the helper's rows rejoin the
 * block's output) is decided here, in the engine's C ABI and in the libr3moe plugin.
 *
 * Hand-off of a step: the routed ids are published to the helper (the publish op masks the helper-owned routes
 * out of the sort in place), the scatter is re-run so the local grouped GEMMs skip them, and moe_mailbox_add
 * replaces moe_gather: it sums the local rows and the helper's rows in slot order. The standalone all-reduce
 * then runs before the residual write (BcMoeRemote::active makes MoeFP8::issues_ar say so), and HyperConn
 * leaves its fused gather / reducing write for that step. With the mailbox ops absent or the step excluded, no
 * call changes anything. */
#ifndef BC_MOE_REMOTE_BLOCK_H
#define BC_MOE_REMOTE_BLOCK_H

#include "rad_fp8.h"
#include "bc_moe_remote.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace rad {
namespace arch {

/* HyperConn::takes_ar: the connection fed by a MoE gather source (top_k > 0) does not take the reduction while
 * the step is handed off; attention and delta-net connections keep their fused writes. */
inline bool bc_moe_remote_hc_open(int64_t moe_top_k, int64_t T) {
    return moe_top_k > 0 && bc_moe_remote_wanted(T) && !bc_moe_remote_shadow();
}

struct BcMoeRemote {
    rad_op publish = 0, add = 0;

    /* The step `T` is handed off (shadow keeps the local path and only publishes). */
    bool policy(int64_t T) const { return publish && add && bc_moe_remote_wanted(T); }
    bool active(int64_t T) const { return policy(T) && !bc_moe_remote_shadow(); }

    /* Block declare, after the quantised experts' order `reg` is final: publish the layer's runtime slot map,
     * and declare the two mailbox ops when the step gate admits any step. The gate comes from libr3moe, so an
     * absent op while the gate is open is an error, never a silent local fallback. */
    template <class Moe>
    int declare(RadBuilder* b, const Moe& m, const std::vector<int64_t>& reg, int layer) {
        const auto& c = m.c;
        const auto& g = m.g;
        const auto& w = m.w;
        const int64_t ne = c.n_expert;
        std::vector<int32_t> slot((size_t)ne, -1);
        for (size_t s = 0; s < reg.size(); ++s)
            if (reg[s] >= 0 && reg[s] < ne) slot[(size_t)reg[s]] = (int32_t)s;
        bc_moe_remote_layer_map(layer, (int)ne, slot.data());

        if (!bc_moe_remote_wanted_upto(g.max_tok)) return RAD_OK;
        const rad_buf q = c.expert_rot ? w.hr.q : w.h.q;
        const rad_buf s = c.expert_rot ? w.hr.s : w.h.s;
        publish = rw(b, RAD_OP(b, "moe_mailbox_publish",
                               RAD_PARAMS(RAD_INT("layer", layer), RAD_RANGE("M", 1, c.rows),
                                          RAD_INT("n", g.n_embd), RAD_INT("top_k", c.top_k),
                                          RAD_INT("expert_base", c.expert_base),
                                          RAD_INT("n_expert", c.n_expert)),
                               RAD_NOWEIGHTS),
                     {w.ids, w.ew, w.h.x, q, s, q}, {w.ids});
        add = RAD_OP(b, "moe_mailbox_add",
                     RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd), RAD_INT("top_k", c.top_k)),
                     RAD_NOWEIGHTS);
        if (add) {
            const rad_buf rd[5] = { w.sorted, w.edn, w.ew, w.sout, w.sgate };
            const rad_buf wr[1] = { w.h.x };
            rad_op_reads(b, add, rd, m.gfold ? 5 : 3);
            rad_op_writes(b, add, wr, 1);
        }
        if (!publish || !add) {
            std::fprintf(stderr, "radiance: MoE remote hand-off is on but its mailbox ops are not declared at layer "
                         "%d (publish %d merge %d; q %d s %d rot %d); is libr3moe loaded?\n", layer,
                         publish != 0, add != 0, q != 0, s != 0, (int)c.expert_rot);
            return RAD_E_INVAL;
        }
        return RAD_OK;
    }

    /* After the router and scatter of a pass, before the grouped GEMMs: publish the routes and re-scatter, so
     * the helper-owned routes are masked out of the local sort. In mask-only (diagnostic) the publish comes
     * later, from the diagnostic hook. */
    template <class Moe>
    void after_route(RadCtx* c_, const Moe& m, int64_t T, int64_t r0, int64_t rows, const RadOperand& ecnt) const {
        if (!policy(T) || bc_moe_remote_mask_only()) return;
        const auto& c = m.c;
        const auto& g = m.g;
        const auto& w = m.w;
        const int64_t k = c.top_k;
        const rad_buf q = c.expert_rot ? w.hr.q : w.h.q;
        const rad_buf s = c.expert_rot ? w.hr.s : w.h.s;
        RAD_ISSUE_N(c_, publish, rows,
                    brow_slice(w.ids, r0, rows, k),
                    brow_slice(w.ew, r0, rows, k),
                    brow_slice(w.h.x, r0, rows, g.n_embd),
                    brow_slice(q, r0, rows, g.n_embd),
                    brow_slice(s, r0, rows, g.n_embd / RAD_FP8_BLOCK),
                    brows(q, T));
        if (active(T))
            RAD_ISSUE_N(c_, m.op_scatter, rows, brow_slice(w.ids, r0, rows, k),
                        brows(w.sorted, rows * k), brows(w.eoff, c.n_expert + 1), ecnt);
    }

    /* In place of the gather: one slot-order merge of the local rows and the helper's rows (and the shared arm
     * when folded) into the block's output. Returns whether it ran; false leaves the normal gather. */
    template <class Moe>
    bool merge(RadCtx* c_, const Moe& m, int64_t T, int64_t r0, int64_t rows) const {
        if (!active(T)) return false;
        const auto& c = m.c;
        const auto& g = m.g;
        const auto& w = m.w;
        const int64_t k = c.top_k;
        RAD_ISSUE_N(c_, add, rows,
                    brows(w.sorted, rows * k), brows(w.edn, rows * k),
                    brow_slice(w.ew, r0, rows, k),
                    m.gfold ? brow_slice(w.sout, r0, rows, g.n_embd) : RAD_NONE,
                    m.gfold ? brow_slice(w.sgate, r0, rows, 1) : RAD_NONE,
                    brow_slice(w.h.x, r0, rows, g.n_embd));
        return true;
    }
};

}  /* namespace arch */
}  /* namespace rad */

#endif /* BC_MOE_REMOTE_BLOCK_H */
"""

PATCHES = (
    FilePatch(
        path="abi/bc_moe_remote.h",
        create=True,
        description="Create the BigCherry remote-MoE extension-point ABI.",
        edits=(Edit(
            id="create-bc-moe-remote-abi",
            anchor=r"\A",
            text=_BC_MOE_REMOTE_H,
            guard=r"^int bc_moe_remote_wanted\(int64_t step_rows\);$",
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any upstream collision.",
        ),),
    ),
    FilePatch(
        path="core/place/bc_moe_remote.cpp",
        create=True,
        description="Create the remote-MoE policy: step gate, plugin registry, layer maps, SKIP_OWNED rule.",
        edits=(Edit(
            id="create-bc-moe-remote-core",
            anchor=r"\A",
            text=_BC_MOE_REMOTE_CPP,
            guard=r'^extern "C" int bc_moe_remote_wanted\(',
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any upstream collision.",
        ),),
    ),
    FilePatch(
        path="core/place/bc_moe_remote_place.h",
        create=True,
        description="Create the planner / bring-up helpers of the remote-MoE extension point.",
        edits=(Edit(
            id="create-bc-moe-remote-place",
            anchor=r"\A",
            text=_BC_MOE_REMOTE_PLACE_H,
            guard=r"^inline bool bc_moe_remote_pin_unit\(",
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any upstream collision.",
        ),),
    ),
    FilePatch(
        path="arch/common/bc_moe_remote_block.h",
        create=True,
        description="Create the MoE block's side of the remote-MoE extension point.",
        edits=(Edit(
            id="create-bc-moe-remote-block",
            anchor=r"\A",
            text=_BC_MOE_REMOTE_BLOCK_H,
            guard=r"^struct BcMoeRemote \{$",
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any upstream collision.",
        ),),
    ),
    FilePatch(
        path="arch/common/rad_block_moe_fp8.h",
        description="Call the remote-MoE extension point from the MoE block: declare, publish, merge, all-reduce.",
        edits=(
            Edit(
                id="include-bc-moe-remote-block",
                anchor=r'^#include "rad_device\.h"$',
                text='\n#include "bc_moe_remote_block.h"',
                guard=r'^#include "bc_moe_remote_block\.h"$',
                expect_matches=1,
                rationale="The device include is unique in this header and sits with architecture dependencies.",
            ),
            Edit(
                id="bc-remote-member",
                anchor=r"^[ \t]*rad_op op_ar = 0, op_add = 0;$",
                text="\n    BcMoeRemote bc_remote;",
                guard=r"^[ \t]*BcMoeRemote bc_remote;$",
                expect_matches=1,
                rationale="The output op declaration is the unique home of the block's per-layer mailbox state.",
            ),
            Edit(
                id="bc-remote-declare",
                anchor=r"^[ \t]*\{w\.x, w\.h\.x\}, \{w\.x\}\);$",
                text="\n    RAD_ARCH_TRY(bc_remote.declare(b, *this, reg, l));",
                guard=r"RAD_ARCH_TRY\(bc_remote\.declare\(b, \*this, reg, l\)\);",
                expect_matches=1,
                rationale="The end of the residual add declaration: reg[] (the runtime slots) is final and the gathered output buffers exist.",
            ),
            Edit(
                id="bc-remote-after-route",
                anchor=r"^[ \t]*const int64_t n4\[4\] = \{",
                mode="insert_before",
                text="    bc_remote.after_route(c_, *this, T, r0, rows, ecnt);\n",
                guard=r"bc_remote\.after_route\(c_, \*this, T, r0, rows, ecnt\);",
                expect_matches=1,
                rationale="n4 is the first routed-GEMM code after the router, scatter and rotated quantisation, so routes and the exact q/s operand are ready.",
            ),
            Edit(
                id="bc-remote-merge",
                anchor=r"^[ \t]*if \(!gather_taken\(g, T, gather_out_rows, gather_out_rows6\)\)$",
                mode="replace",
                text="""    if (!bc_remote.merge(c_, *this, T, r0, rows) &&
        !gather_taken(g, T, gather_out_rows, gather_out_rows6))""",
                guard=r"if \(!bc_remote\.merge\(c_, \*this, T, r0, rows\) &&",
                expect_matches=1,
                rationale="The pass's single gather guard: a handed-off step merges instead of gathering.",
            ),
            Edit(
                id="bc-remote-issues-ar",
                anchor=r"^[ \t]*return op_ar && !ar_taken\(g, ar_out_take != kArOutNorm \? T : rows, ar_out, ar_out_take\);$",
                mode="replace",
                text="        return op_ar && (bc_remote.active(T) || !ar_taken(g, ar_out_take != kArOutNorm ? T : rows, ar_out, ar_out_take));",
                guard=r"return op_ar && \(bc_remote\.active\(T\) \|\|",
                expect_matches=1,
                rationale="issues_ar is the one place every caller (pass and the pipelined prefill callers) asks whether the block runs its own all-reduce; a handed-off step must, since the merged row is not a fused reduction input.",
            ),
        ),
    ),
    FilePatch(
        path="arch/common/rad_block_hc.h",
        description="Keep HyperConn's fused gather / reducing write off while a MoE-fed connection is handed off.",
        edits=(
            Edit(
                id="include-bc-moe-remote-block-hc",
                anchor=r'^#include "rad_fp8\.h"$',
                text='\n#include "bc_moe_remote_block.h"',
                guard=r'^#include "bc_moe_remote_block\.h"$',
                expect_matches=1,
                rationale="The fp8 include is the unique architecture dependency line in this header.",
            ),
            Edit(
                id="bc-remote-takes-ar",
                anchor=r"^[ \t]*bool takes_ar\(int64_t T\) const \{ return ar_write\(T\) != 0; \}$",
                mode="replace",
                text="    bool takes_ar(int64_t T) const { return ar_write(T) != 0 && !bc_moe_remote_hc_open(gsrc.top_k, T); }",
                guard=r"!bc_moe_remote_hc_open\(gsrc\.top_k, T\)",
                expect_matches=1,
                rationale="write() reaches the gather form through takes_gather() = takes_ar() && ..., and the reducing write through takes_ar(), so this one predicate closes both for a MoE-fed connection.",
            ),
        ),
    ),
    FilePatch(
        path="core/place/planner.cpp",
        description="Leave helper-owned expert units unplaced (mapped, never read) when nothing local reads them.",
        edits=(
            Edit(
                id="include-bc-moe-remote-place-planner",
                anchor=r'^#include "planner\.h"$',
                text='\n#include "bc_moe_remote_place.h"',
                guard=r'^#include "bc_moe_remote_place\.h"$',
                expect_matches=1,
                rationale="The planner's own header include is its unique first include line.",
            ),
            Edit(
                id="bc-remote-pin-unit",
                anchor=r"^[ \t]*bool host_only = false, stored = true;$",
                mode="insert_before",
                text="        if (bc_moe_remote_pin_unit(mu, in_.config->max_tok)) continue;\n",
                guard=r"if \(bc_moe_remote_pin_unit\(mu, in_\.config->max_tok\)\) continue;",
                expect_matches=1,
                rationale="This declaration opens pin_mapped's per-unit loop body; a helper-owned unit is decided before any host or n-gram logic and leaves the loop.",
            ),
            Edit(
                id="bc-remote-mark-footprint",
                anchor=r"^    for \(U& mu : units\) mu\.mapped = mapped_unit\(mu\.mapped, container && mu\.direct, host_backend\);$",
                text="\n    bc_moe_remote_mark_footprint(units);",
                guard=r"bc_moe_remote_mark_footprint\(units\);",
                expect_matches=1,
                rationale="weight_footprint's unit pass ends with this mapped resolution; the budgets read mu.mapped right after it.",
            ),
            Edit(
                id="rank-units-skip-preplaced",
                anchor=r"^    for \(int32_t i = 0; i < \(int32_t\)p->units\.size\(\); \+\+i\) order->push_back\(i\);$",
                mode="replace",
                text="""    /* A preplaced unit is never filled (the rule every assignment pass states), and the expert fill treats every
     * conditional unit as fillable, so a helper-owned expert must not be ranked. */
    for (int32_t i = 0; i < (int32_t)p->units.size(); ++i)
        if (!p->units[(size_t)i].preplaced()) order->push_back(i);""",
                guard=r"if \(!p->units\[\(size_t\)i\]\.preplaced\(\)\) order->push_back\(i\);",
                expect_matches=1,
                rationale="rank_units' order is what the expert-tiered fill walks; leaving preplaced units out keeps a helper-owned expert on the mapped tier. This is the one policy line that cannot move out of the core.",
            ),
        ),
    ),
    FilePatch(
        path="core/engine_bringup.cpp",
        description="Publish helper-owned expert units as never-read placeholders instead of loading them.",
        edits=(
            Edit(
                id="include-bc-moe-remote-place-bringup",
                anchor=r'^#include "engine_priv\.h"$',
                text='\n#include "place/bc_moe_remote_place.h"',
                guard=r'^#include "place/bc_moe_remote_place\.h"$',
                expect_matches=1,
                rationale="engine_priv.h is the bring-up's unique first include line.",
            ),
            Edit(
                id="bc-remote-publish-placeholder",
                anchor=r"^[ \t]*if \(\(w\.tier == Tier::Mapped \|\| w\.tier == Tier::SSD\) && \(!direct \|\| !e\)\) \{$",
                mode="insert_before",
                text="            if (bc_moe_remote_publish_placeholder(cfg_.max_tok, w, e, *file_, r->mover, (rad_weight)i)) continue;\n",
                guard=r"if \(bc_moe_remote_publish_placeholder\(cfg_\.max_tok, w, e, \*file_, r->mover, \(rad_weight\)i\)\) continue;",
                expect_matches=1,
                rationale="The single tier / stored-form check in the weight publish loop; a helper-owned placeholder is published before it, since nothing reads its bytes and its relayouted form would fail the check.",
            ),
        ),
    ),
    FilePatch(
        path="core/engine.cpp",
        description="Tell the extension point where an inference burst begins and ends (host scheduler loop, replay-safe).",
        edits=(
            Edit(
                id="include-bc-moe-remote-engine",
                anchor=r'^#include "engine_priv\.h"$',
                text='\n#include "bc_moe_remote.h"',
                guard=r'^#include "bc_moe_remote\.h"$',
                expect_matches=1,
                rationale="engine_priv.h is engine.cpp's unique first include line.",
            ),
            Edit(
                id="bc-remote-engine-idle",
                anchor=r"^            step_batch_\.store\(nullptr, std::memory_order_release\);$",
                text="\n            bc_moe_remote_engine_idle();",
                guard=r"bc_moe_remote_engine_idle\(\);",
                expect_matches=1,
                rationale="The engine loop's idle branch clears the step batch here when the scheduler has no step: every earlier step has joined, so a burst ends.",
            ),
            Edit(
                id="bc-remote-engine-step",
                anchor=r"^        idle = 0;$",
                mode="insert_before",
                text="        bc_moe_remote_engine_step();\n",
                guard=r"bc_moe_remote_engine_step\(\);",
                expect_matches=1,
                rationale="The loop resets its idle backoff exactly when the scheduler returned a step, before the ranks are released or any recorded pass is replayed.",
            ),
        ),
    ),
)
