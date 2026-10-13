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
#ifdef __cplusplus
}
#endif
"""

_MOE_REMOTE_CPP = r"""/* moe_remote.cpp -- decode-sized policy gate for GPU-lane remote MoE. */
#include "rad_moe_remote.h"
#include <climits>
#include <cstdlib>

namespace {
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
        description="Publish remote experts and add their weighted results after the plain gather.",
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
                text="    rad_op op_ar = 0, op_add = 0, op_remote_publish = 0, op_remote_add = 0, op_remote_shadow = 0;",
                guard=r"op_remote_publish = 0",
                expect_matches=1,
                rationale="The output op declaration is the unique home for the pre-reduction mailbox ops.",
            ),
            Edit(
                id="declare-remote-mailbox-ops",
                anchor=r"^[ \t]*\{w\.x, w\.h\.x\}, \{w\.x\}\);$",
                text="""
    /* Only layers without protected (plain-bf16) experts hand off: removing protected experts from reg[]
     * renumbers the regular slots, and TP class/half follows the slot, not the expert id. Shadow showed
     * those layers (34/44/46/47 on Flash-Next) wrong while every other layer matched to ~2e-4. */
    if (rad_moe_remote_wanted_upto(g.max_tok) && n_reg == (int64_t)c.n_expert) {
        const rad_buf remote_q = c.expert_rot ? w.hr.q : w.h.q;
        const rad_buf remote_s = c.expert_rot ? w.hr.s : w.h.s;
        op_remote_publish = rw(b, RAD_OP(b, "moe_mailbox_publish",
                                   RAD_PARAMS(RAD_INT("layer", l), RAD_RANGE("M", 1, c.rows),
                                              RAD_INT("n", g.n_embd), RAD_INT("top_k", c.top_k),
                                              RAD_INT("expert_base", c.expert_base),
                                              RAD_INT("n_expert", c.n_expert)),
                                   RAD_NOWEIGHTS),
                               {w.ids, w.ew, w.h.x, remote_q, remote_s, remote_q}, {w.ids});
        op_remote_add = rw(b, RAD_OP(b, "moe_mailbox_add",
                               RAD_PARAMS(RAD_RANGE("M", 1, c.rows),
                                          RAD_INT("n", g.n_embd),
                                          RAD_INT("top_k", c.top_k)),
                               RAD_NOWEIGHTS),
                           {w.h.x}, {w.h.x});
        op_remote_shadow = rw(b, RAD_OP(b, "moe_mailbox_shadow",
                                  RAD_PARAMS(RAD_RANGE("M", 1, c.rows),
                                             RAD_INT("n", g.n_embd),
                                             RAD_INT("top_k", c.top_k)),
                                  RAD_NOWEIGHTS),
                              {w.sorted, w.edn}, {w.edn});
        if (!op_remote_publish || !op_remote_add || !op_remote_shadow) {
            std::fprintf(stderr, "radiance: MoE remote ops not declared at layer %d (publish %d add %d shadow %d; "
                         "q %d s %d rot %d)\\n", l, op_remote_publish != 0, op_remote_add != 0,
                         op_remote_shadow != 0, remote_q != 0, remote_s != 0, (int)c.expert_rot);
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
        op_remote_publish && op_remote_add && op_remote_shadow && rad_moe_remote_wanted(rows);
    const bool remote_shadow = remote_policy && rad_moe_remote_shadow();
    const bool remote_active = remote_policy && !remote_shadow;""",
                guard=r"remote_policy =",
                expect_matches=1,
                rationale="The shared-arm state is the first per-pass policy declaration after row geometry is resolved.",
            ),
            Edit(
                id="publish-remote-and-rescatter",
                anchor=r"^[ \t]*const int64_t n4\[4\] = \{",
                mode="insert_before",
                text="""    if (remote_policy) {
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
                text="""    if (remote_shadow)
        RAD_ISSUE_N(c_, op_remote_shadow, rows,
                    brows(w.sorted, rows * k), brows(w.edn, rows * k));
""",
                guard=r"RAD_ISSUE_N\(c_, op_remote_shadow, rows,",
                expect_matches=1,
                rationale="The shared-down issue is the first code line after every routed/protected down writer, before calibration and gather.",
            ),
            Edit(
                id="force-plain-gather-for-remote",
                anchor=r"^[ \t]*if \(!gather_taken\(g, T, gather_out_rows, gather_out_rows6\)\)$",
                mode="replace",
                text="    if (remote_active || !gather_taken(g, T, gather_out_rows, gather_out_rows6))",
                guard=r"remote_active \|\| !gather_taken\(g, T, gather_out_rows, gather_out_rows6\)",
                expect_matches=1,
                rationale="A remote pass needs a materialized token output between gather and collective, so the downstream fused gather+all-reduce cannot consume the gather.",
            ),
            Edit(
                id="add-remote-before-all-reduce",
                anchor=r"^[ \t]*if \(op_ar && !ar_taken\(g, ar_out_take != kArOutNorm \? T : rows, ar_out, ar_out_take\)\)$",
                mode="replace",
                text="""    if (remote_active)
        RAD_ISSUE_N(c_, op_remote_add, rows,
                    brow_slice(w.h.x, r0, rows, g.n_embd));
    if (op_ar && (remote_active ||
                  !ar_taken(g, ar_out_take != kArOutNorm ? T : rows, ar_out, ar_out_take)))""",
                guard=r"op_remote_add, rows",
                expect_matches=1,
                rationale="Add weighted helper contributions only after the plain gather produced token rows, then force the standalone all-reduce before the residual write.",
            ),
        ),
    ),
    FilePatch(
        path="arch/common/rad_block_hc.h",
        description="Disable HyperConn fused gather/all-reduce writes while a remote-MoE pass needs a post-gather add.",
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
                rationale="Normal remote passes must leave the gather to MoeFP8; shadow keeps the original fused output path.",
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
