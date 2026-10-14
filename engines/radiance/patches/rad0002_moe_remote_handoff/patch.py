"""Radiance 1.3.0: GPU-lane remote whole-expert MoE hand-off."""

from bigcherry.patcher import Edit, FilePatch

_RAD_MOE_REMOTE_H = r"""/* rad_moe_remote.h -- policy gate for GPU-lane remote whole-expert compute. */
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Runtime step gate. The libr3 mailbox component sets the bounds only after its mapped segment
 * is pinned and ready; an unset minimum means remote MoE is disabled. */
int rad_moe_remote_wanted(int64_t step_rows);
int rad_moe_remote_wanted_upto(int64_t max_step_rows);
int rad_moe_remote_shadow(void);
int rad_moe_remote_mask_only(void);
typedef void (*rad_moe_remote_layer_map_hook)(
    int layer, int n_expert, const int32_t* slot_of_expert);
void rad_moe_remote_set_layer_map_hook(rad_moe_remote_layer_map_hook hook);
void rad_moe_remote_layer_map(
    int layer, int n_expert, const int32_t* slot_of_expert);
#ifdef __cplusplus
}
#endif
"""

_MOE_REMOTE_CPP = r"""/* moe_remote.cpp -- decode-sized policy gate for GPU-lane remote MoE. */
#include "rad_moe_remote.h"
#include <atomic>
#include <climits>
#include <cstdlib>
#include <map>
#include <mutex>
#include <vector>

namespace {
std::atomic<rad_moe_remote_layer_map_hook> g_layer_map_hook{nullptr};

int64_t min_rows() {
    static const int64_t v = [] {
        const char* s = std::getenv("R3_MOE_REMOTE_MIN_ROWS");
        if (!s || !*s) return (int64_t)LLONG_MAX;
        char* end = nullptr;
        const long long n = std::strtoll(s, &end, 10);
        return end && *end == 0 && n >= 0 ? (int64_t)n : (int64_t)LLONG_MAX;
    }();
    return v;
}
int64_t max_rows() {
    static const int64_t v = [] {
        const char* s = std::getenv("R3_MOE_REMOTE_MAX_ROWS");
        if (!s || !*s) return (int64_t)LLONG_MAX;
        char* end = nullptr;
        const long long n = std::strtoll(s, &end, 10);
        return end && *end == 0 && n >= 0 ? (int64_t)n : (int64_t)LLONG_MAX;
    }();
    return v;
}
}

extern "C" int rad_moe_remote_wanted(int64_t step_rows) {
    const int64_t lo = min_rows(), hi = max_rows();
    return lo != LLONG_MAX && step_rows >= lo && step_rows <= hi;
}

extern "C" int rad_moe_remote_wanted_upto(int64_t max_step_rows) {
    const int64_t lo = min_rows(), hi = max_rows();
    return lo != LLONG_MAX && lo <= hi && lo <= max_step_rows;
}

extern "C" int rad_moe_remote_shadow(void) {
    const char* e = std::getenv("R3_MOE_HELPER_SHADOW");
    return e && *e && !(e[0] == '0' && e[1] == 0);
}

extern "C" int rad_moe_remote_mask_only(void) {
    const char* e = std::getenv("R3_MOE_HELPER_MASK_ONLY");
    return e && *e && !(e[0] == '0' && e[1] == 0);
}

/* Layer maps are published at declare, which can run before the plugin installs its hook (libr3 installs
 * it when the mailbox device is first created). Keep every published map and replay them when the hook
 * arrives, so no layer's map is lost to that ordering. */
namespace {
std::mutex g_layer_map_mu;
std::map<int, std::vector<int32_t>> g_layer_maps;
}

extern "C" void rad_moe_remote_set_layer_map_hook(rad_moe_remote_layer_map_hook hook) {
    std::lock_guard<std::mutex> lock(g_layer_map_mu);
    g_layer_map_hook.store(hook, std::memory_order_release);
    if (hook)
        for (const auto& [layer, slots] : g_layer_maps)
            hook(layer, (int)slots.size(), slots.data());
}

extern "C" void rad_moe_remote_layer_map(
    int layer, int n_expert, const int32_t* slot_of_expert) {
    if (n_expert <= 0 || !slot_of_expert) return;
    std::lock_guard<std::mutex> lock(g_layer_map_mu);
    g_layer_maps[layer].assign(slot_of_expert, slot_of_expert + n_expert);
    const auto hook = g_layer_map_hook.load(std::memory_order_acquire);
    if (hook) hook(layer, n_expert, slot_of_expert);
}
"""

PATCHES = (
    FilePatch(
        path="abi/rad_moe_remote.h",
        create=True,
        description="Create the GPU-lane remote-MoE policy ABI.",
        edits=(Edit(
            id="create-remote-moe-abi",
            anchor=r"\A",
            text=_RAD_MOE_REMOTE_H,
            guard=r"^int rad_moe_remote_wanted\(int64_t step_rows\);$",
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any upstream collision.",
        ),),
    ),
    FilePatch(
        path="core/place/moe_remote.cpp",
        create=True,
        description="Create decode-sized remote-MoE policy gating; no CPU hand-off path exists.",
        edits=(Edit(
            id="create-remote-moe-core",
            anchor=r"\A",
            text=_MOE_REMOTE_CPP,
            guard=r'^extern "C" int rad_moe_remote_wanted',
            expect_matches=1,
            rationale="Created files are anchored at the empty upstream file; create=True rejects any upstream collision.",
        ),),
    ),
    FilePatch(
        path="arch/common/rad_block_moe_fp8.h",
        description="Publish remote experts and merge local/helper rows with exact moe_gather arithmetic.",
        edits=(
            Edit(
                id="include-remote-moe-abi",
                anchor=r'^#include "rad_device\.h"$',
                text='\n#include "rad_moe_remote.h"',
                guard=r'^#include "rad_moe_remote\.h"$',
                expect_matches=1,
                rationale="The device include is unique in this header and sits with architecture dependencies.",
            ),
            Edit(
                id="remote-ops",
                anchor=r"^[ \t]*rad_op op_ar = 0, op_add = 0;$",
                mode="replace",
                text="""    rad_op op_ar = 0, op_add = 0, op_remote_publish = 0, op_remote_add = 0, op_remote_shadow = 0;
    rad_op op_remote_mask_capture = 0, op_remote_mask_restore = 0;""",
                guard=r"op_remote_publish = 0",
                expect_matches=1,
                rationale="The output op declaration is the unique home for the pre-reduction mailbox ops.",
            ),
            Edit(
                id="publish-runtime-layer-map",
                anchor=r"^[ \t]*if \(std::find\(prot\.begin\(\), prot\.end\(\), e\) == prot\.end\(\)\) reg\.push_back\(e\);$",
                text="""
    std::vector<int32_t> remote_slot((size_t)ne, -1);
    for (int64_t s = 0; s < n_reg; ++s)
        remote_slot[(size_t)reg[(size_t)s]] = (int32_t)s;
    if (!prot.empty() && rad_moe_remote_shadow()) {
        std::string pl;
        for (int64_t e : prot) pl += (pl.empty() ? "" : ",") + std::to_string(e);
        fprintf(stderr, "radiance: MoE remote shadow: layer %d protected [%s] n_reg %lld ncls %d\\n",
                l, pl.c_str(), (long long)n_reg, c.ff_hi4[0] > 0 ? 4 : 2);
    }
    rad_moe_remote_layer_map(l, (int)ne, remote_slot.data());""",
                guard=r"rad_moe_remote_layer_map\(l, \(int\)ne, remote_slot\.data\(\)\);",
                expect_matches=1,
                rationale="The reg[] fill line is the first point where runtime protected-expert removal and compact regular slots are both final.",
            ),
            Edit(
                id="declare-remote-mailbox-ops",
                anchor=r"^[ \t]*\{w\.x, w\.h\.x\}, \{w\.x\}\);$",
                text="""
    if (rad_moe_remote_wanted_upto(g.max_tok)) {
        const rad_buf remote_q = c.expert_rot ? w.hr.q : w.h.q;
        const rad_buf remote_s = c.expert_rot ? w.hr.s : w.h.s;
        op_remote_publish = rw(b, RAD_OP(b, "moe_mailbox_publish",
                                   RAD_PARAMS(RAD_INT("layer", l), RAD_RANGE("M", 1, c.rows),
                                              RAD_INT("n", g.n_embd), RAD_INT("top_k", c.top_k),
                                              RAD_INT("expert_base", c.expert_base),
                                              RAD_INT("n_expert", c.n_expert)),
                                   RAD_NOWEIGHTS),
                               {w.ids, w.ew, w.h.x, remote_q, remote_s, remote_q}, {w.ids});
        op_remote_add = RAD_OP(b, "moe_mailbox_add",
                               RAD_PARAMS(RAD_RANGE("M", 1, c.rows),
                                          RAD_INT("n", g.n_embd),
                                          RAD_INT("top_k", c.top_k)),
                               RAD_NOWEIGHTS);
        if (op_remote_add) {
            const rad_buf rd[5] = { w.sorted, w.edn, w.ew, w.sout, w.sgate };
            const rad_buf wr[1] = { w.h.x };
            rad_op_reads(b, op_remote_add, rd, gfold ? 5 : 3);
            rad_op_writes(b, op_remote_add, wr, 1);
        }
        op_remote_shadow = rw(b, RAD_OP(b, "moe_mailbox_shadow",
                                  RAD_PARAMS(RAD_RANGE("M", 1, c.rows),
                                             RAD_INT("n", g.n_embd),
                                             RAD_INT("top_k", c.top_k)),
                                  RAD_NOWEIGHTS),
                              {w.sorted, w.edn}, {w.edn});
        op_remote_mask_capture = rw(b, RAD_OP(b, "moe_mailbox_mask_ref",
                                        RAD_PARAMS(RAD_RANGE("M", 1, c.rows),
                                                   RAD_INT("n", g.n_embd),
                                                   RAD_INT("top_k", c.top_k),
                                                   RAD_INT("restore", 0)),
                                        RAD_NOWEIGHTS),
                                    {w.sorted, w.edn}, {w.edn});
        op_remote_mask_restore = rw(b, RAD_OP(b, "moe_mailbox_mask_ref",
                                        RAD_PARAMS(RAD_RANGE("M", 1, c.rows),
                                                   RAD_INT("n", g.n_embd),
                                                   RAD_INT("top_k", c.top_k),
                                                   RAD_INT("restore", 1)),
                                        RAD_NOWEIGHTS),
                                    {w.sorted, w.edn}, {w.edn});
        if (!op_remote_publish || !op_remote_add || !op_remote_shadow ||
            !op_remote_mask_capture || !op_remote_mask_restore) {
            std::fprintf(stderr, "radiance: MoE remote ops not declared at layer %d (publish %d merge %d shadow %d mask %d/%d; "
                         "q %d s %d rot %d)\\n", l, op_remote_publish != 0, op_remote_add != 0,
                         op_remote_shadow != 0, op_remote_mask_capture != 0,
                         op_remote_mask_restore != 0, remote_q != 0, remote_s != 0,
                         (int)c.expert_rot);
            return RAD_E_INVAL;
        }
    }""",
                guard=r'moe_mailbox_publish',
                expect_matches=1,
                rationale="The final residual declaration is immediately before the collective declarations and has the gathered token output available.",
            ),
            Edit(
                id="remote-pass-state",
                anchor=r"^[ \t]*const bool shared = op_sgate != 0;$",
                text="""
    const bool remote_policy =
        op_remote_publish && op_remote_add && op_remote_shadow &&
        op_remote_mask_capture && op_remote_mask_restore && rad_moe_remote_wanted(rows);
    const bool remote_shadow = remote_policy && rad_moe_remote_shadow();
    const bool remote_active = remote_policy && !remote_shadow;
    const bool remote_mask_only =
        remote_active && rad_moe_remote_mask_only();""",
                guard=r"remote_policy =",
                expect_matches=1,
                rationale="The shared-arm state is the first per-pass policy declaration after row geometry is resolved.",
            ),
            Edit(
                id="publish-remote-and-rescatter",
                anchor=r"^[ \t]*const int64_t n4\[4\] = \{",
                mode="insert_before",
                text="""    if (remote_policy && !remote_mask_only) {
        const rad_buf remote_q = rot ? w.hr.q : w.h.q;
        const rad_buf remote_s = rot ? w.hr.s : w.h.s;
        RAD_ISSUE_N(c_, op_remote_publish, rows,
                    brow_slice(w.ids, r0, rows, k),
                    brow_slice(w.ew, r0, rows, k),
                    brow_slice(w.h.x, r0, rows, g.n_embd),
                    brow_slice(remote_q, r0, rows, g.n_embd),
                    brow_slice(remote_s, r0, rows, g.n_embd / RAD_FP8_BLOCK),
                    brows(remote_q, T));
        if (remote_active)
            RAD_ISSUE_N(c_, op_scatter, rows, brow_slice(w.ids, r0, rows, k),
                        brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt);
    }
""",
                guard=r"RAD_ISSUE_N\(c_, op_remote_publish, rows,",
                expect_matches=1,
                rationale="The n4 declaration is the first routed-GEMM code after op_hq, so the exact q/s operand is ready before publication.",
            ),
            Edit(
                id="shadow-compare-remote-down",
                anchor=r"^[ \t]*if \(shared\) sh_down\.step\(c_, w\.sff, w\.sout, T, r0, rows\);$",
                mode="insert_before",
                text="""    if (remote_mask_only) {
        RAD_ISSUE_N(c_, op_remote_mask_capture, rows,
                    brows(w.sorted, rows * k), brows(w.edn, rows * k));
        const rad_buf remote_q = rot ? w.hr.q : w.h.q;
        const rad_buf remote_s = rot ? w.hr.s : w.h.s;
        RAD_ISSUE_N(c_, op_remote_publish, rows,
                    brow_slice(w.ids, r0, rows, k),
                    brow_slice(w.ew, r0, rows, k),
                    brow_slice(w.h.x, r0, rows, g.n_embd),
                    brow_slice(remote_q, r0, rows, g.n_embd),
                    brow_slice(remote_s, r0, rows, g.n_embd / RAD_FP8_BLOCK),
                    brows(remote_q, T));
        RAD_ISSUE_N(c_, op_scatter, rows, brow_slice(w.ids, r0, rows, k),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt);
        RAD_ISSUE_N(c_, op_remote_mask_restore, rows,
                    brows(w.sorted, rows * k), brows(w.edn, rows * k));
    }
    if (remote_shadow)
        RAD_ISSUE_N(c_, op_remote_shadow, rows,
                    brows(w.sorted, rows * k), brows(w.edn, rows * k));
""",
                guard=r"RAD_ISSUE_N\(c_, op_remote_mask_capture, rows,",
                expect_matches=1,
                rationale="The shared-down issue is the first code line after every routed/protected down writer, so mask-only can snapshot engine rows before masking and restore local row order after the diagnostic rescatter.",
            ),
            Edit(
                id="force-plain-gather-for-remote",
                anchor=r"^[ \t]*if \(!gather_taken\(g, T, gather_out_rows, gather_out_rows6\)\)$",
                mode="replace",
                text="""    if (remote_active)
        RAD_ISSUE_N(c_, op_remote_add, rows,
                    brows(w.sorted, rows * k), brows(w.edn, rows * k),
                    brow_slice(w.ew, r0, rows, k),
                    gfold ? brow_slice(w.sout, r0, rows, g.n_embd) : RAD_NONE,
                    gfold ? brow_slice(w.sgate, r0, rows, 1) : RAD_NONE,
                    brow_slice(w.h.x, r0, rows, g.n_embd));
    else if (!gather_taken(g, T, gather_out_rows, gather_out_rows6))""",
                guard=r"else if \(!gather_taken\(g, T, gather_out_rows, gather_out_rows6\)\)",
                expect_matches=1,
                rationale="Normal remote mode replaces moe_gather with one slot-order merge, avoiding a local bf16 narrow followed by a second remote-add narrow.",
            ),
            Edit(
                id="add-remote-before-all-reduce",
                anchor=r"^[ \t]*if \(op_ar && !ar_taken\(g, ar_out_take != kArOutNorm \? T : rows, ar_out, ar_out_take\)\)$",
                mode="replace",
                text="""    if (op_ar && (remote_active ||
                  !ar_taken(g, ar_out_take != kArOutNorm ? T : rows, ar_out, ar_out_take)))""",
                guard=r"op_ar && \(remote_active \|\|",
                expect_matches=1,
                rationale="The remote merge already wrote the exact gathered bf16 row, so only force the standalone all-reduce before the residual write.",
            ),
        ),
    ),
    FilePatch(
        path="arch/common/rad_block_hc.h",
        description="Disable HyperConn fused gather/all-reduce writes while a remote-MoE pass uses the mailbox merge.",
        edits=(
            Edit(
                id="include-remote-moe-abi-hc",
                anchor=r'^#include "rad_fp8\.h"$',
                text='\n#include "rad_moe_remote.h"',
                guard=r'^#include "rad_moe_remote\.h"$',
                expect_matches=1,
                rationale="The fp8 include is the unique architecture dependency line in this header.",
            ),
            Edit(
                id="remote-hc-write-state",
                anchor=r"^[ \t]*if \(!op_write\) return;$",
                text="""
    const bool remote_active =
        gsrc.top_k > 0 && rad_moe_remote_wanted(rows) && !rad_moe_remote_shadow();""",
                guard=r"const bool remote_active =",
                expect_matches=1,
                rationale="Only a connection with a declared MoE gather source may open the remote window; attention and delta-net writes keep their normal fused reductions.",
            ),
            Edit(
                id="remote-disable-hc-gather-ar",
                anchor=r"^[ \t]*if \(takes_gather\(T\)\) \{$",
                mode="replace",
                text="    if (!remote_active && takes_gather(T)) {",
                guard=r"!remote_active && takes_gather\(T\)",
                expect_matches=1,
                rationale="Normal remote passes use the mailbox slot-order merge; shadow keeps HyperConn's original fused gather path.",
            ),
            Edit(
                id="remote-disable-hc-ar-write",
                anchor=r"^[ \t]*RAD_ISSUE_N\(c_, takes_ar\(T\) \? ar_write\(T\) : op_write, rows,$",
                mode="replace",
                text="    RAD_ISSUE_N(c_, (!remote_active && takes_ar(T)) ? ar_write(T) : op_write, rows,",
                guard=r"\(!remote_active && takes_ar\(T\)\) \? ar_write\(T\) : op_write",
                expect_matches=1,
                rationale="Normal remote passes already reduce in MoeFP8; shadow leaves HyperConn's original reduction untouched.",
            ),
        ),
    ),
)
