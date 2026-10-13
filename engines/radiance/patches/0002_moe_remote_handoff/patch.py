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
        description="Publish and join remote experts as ordinary device ops on the rank stream.",
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
                text="    rad_op op_ar = 0, op_add = 0, op_remote_publish = 0, op_remote_join = 0;",
                guard=r"op_remote_publish = 0",
                expect_matches=1,
                rationale="The output op declaration is the unique home for the pre-reduction mailbox ops.",
            ),
            Edit(
                id="declare-remote-mailbox-ops",
                anchor=r"^[ \t]*\{w\.x, w\.h\.x\}, \{w\.x\}\);$",
                text="""
    if (rad_moe_remote_wanted_upto(g.max_tok)) {
        op_remote_publish = rw(b, RAD_OP(b, "moe_mailbox_publish",
                                   RAD_PARAMS(RAD_INT("layer", l), RAD_RANGE("M", 1, c.rows),
                                              RAD_INT("n", g.n_embd), RAD_INT("top_k", c.top_k),
                                              RAD_INT("expert_base", c.expert_base),
                                              RAD_INT("n_expert", c.n_expert)),
                                   RAD_NOWEIGHTS),
                               {w.ids, w.ew, w.h.x}, {w.ids});
        op_remote_join = rw(b, RAD_OP(b, "moe_mailbox_join",
                                RAD_PARAMS(RAD_RANGE("M", 1, c.rows),
                                           RAD_INT("n", g.n_embd),
                                           RAD_INT("top_k", c.top_k)),
                                RAD_NOWEIGHTS),
                            {w.sorted, w.edn}, {w.sorted, w.edn});
        if (!op_remote_publish || !op_remote_join) return RAD_E_INVAL;
    }""",
                guard=r'moe_mailbox_publish',
                expect_matches=1,
                rationale="The final residual declaration is immediately before the collective declarations and has the route/down buffers available.",
            ),
            Edit(
                id="remote-pass-state",
                anchor=r"^[ \t]*const bool shared = op_sgate != 0;$",
                text="""
    const bool remote_policy =
        op_remote_publish && op_remote_join && rad_moe_remote_wanted(T);""",
                guard=r"remote_policy =",
                expect_matches=1,
                rationale="The shared-arm state is the first per-pass policy declaration after row geometry is resolved.",
            ),
            Edit(
                id="publish-remote-and-rescatter",
                anchor=r"^[ \t]*/\* The rotated re-quantisation of the same input, when the expert plane carries a$"
                mode="replace",
                text="""    if (remote_policy) {
        RAD_ISSUE_N(c_, op_remote_publish, rows,
                    brow_slice(w.ids, r0, rows, k),
                    brow_slice(w.ew, r0, rows, k),
                    brow_slice(w.h.x, r0, rows, g.n_embd));
        /* Preserve the normal router/top-k arithmetic. Fused paths already scattered the
         * unmasked ids, so rebuild the cheap integer scatter after publish masks remote slots. */
        RAD_ISSUE_N(c_, op_scatter, rows, brow_slice(w.ids, r0, rows, k),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt);
    }

    /* The rotated re-quantisation of the same input, when the expert plane carries a""",
                guard=r"Preserve the normal router/top-k arithmetic",
                expect_matches=1,
                rationale="Publish after whichever normal router/top-k path ran, then rebuild only scatter from the masked ids so helper-on does not globally change router arithmetic.",
            ),
            Edit(
                id="join-remote-before-gather",
                anchor=r"^[ \t]*if \(shared\) sh_down\.step\(c_, w\.sff, w\.sout, T, r0, rows\);$",
                text="""
    if (remote_policy)
        RAD_ISSUE_N(c_, op_remote_join, rows,
                    brows(w.sorted, rows * k), brows(w.edn, rows * k));""",
                guard=r"op_remote_join, rows",
                expect_matches=1,
                rationale="All local down writers have finished here; restore helper rows and their sorted-slot mapping before the unchanged engine gather.",
            ),
        ),
    ),
)
