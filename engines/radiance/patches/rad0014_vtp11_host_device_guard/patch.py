"""VTP11: prevent GPU workloads from entering CPU kernels with device-only activations.

A host-capable weight format is not sufficient for a host execution site: the
architecture's activation buffers must be CPU accessible too. On a GPU backend
the ordinary RAD_DOMAIN_DEVICE activation arena is hipMalloc memory. A CPU
kernel writing it (e.g. AVX GDN conv_prep) segfaults. Declarations already
record exact buffer read/write sets; carry one conservative bit to planning.
All changes are opt-in, source-pinned and do not change production recipes.
"""

import re

from bigcherry.patcher import Edit, FilePatch


def replacement(id, old, new):
    marker = "BC_VTP11_HOST_DOMAIN_" + id.upper().replace("-", "_")
    assert "BC_VTP11_HOST_DOMAIN" in new
    new = new.replace("BC_VTP11_HOST_DOMAIN", marker, 1)
    return Edit(
        id=id, anchor=re.escape(old), mode="replace", text=new,
        guard=re.escape(marker),
        expect_matches=1,
        max_span_lines=max(10, old.count("\n") + 4),
        rationale="Exact pinned v1.17.0 source hunk; refuse upstream drift",
    )


PATCHES = (
    FilePatch(
        path="core/rad_core.h",
        description="Persist whether an op touches device-only activation buffers.",
        edits=(replacement(
            "vtp11-op-buffer-domain",
            """    std::vector<int32_t>  weight_opd;
    const RadOpSchema*    schema = nullptr;""",
            """    std::vector<int32_t>  weight_opd;
    /* BC_VTP11_HOST_DOMAIN: CPU kernels may not dereference a GPU activation arena.
     * Set from declared read/write sets before the placement planner runs. */
    bool                  host_unsafe_device_buffers = false;
    const RadOpSchema*    schema = nullptr;""",
        ),),
    ),
    FilePatch(
        path="core/build/rad_builder.cpp",
        description="Carry the op's activation-domain contract into the planner.",
        edits=(
            replacement(
                "vtp11-read-set-domain",
                """        op_bufs_[h].reads.push_back(b[i]);""",
                """        op_bufs_[h].reads.push_back(b[i]);
        /* BC_VTP11_HOST_DOMAIN: a host kernel cannot read hipMalloc data. */
        if (prog_.buffers[b[i]].decl.domain == RAD_DOMAIN_DEVICE)
            prog_.ops[h].host_unsafe_device_buffers = true;""",
            ),
            replacement(
                "vtp11-write-set-domain",
                """        op_bufs_[h].writes.push_back(b[i]);""",
                """        op_bufs_[h].writes.push_back(b[i]);
        /* BC_VTP11_HOST_DOMAIN: a host kernel cannot store to hipMalloc data. */
        if (prog_.buffers[b[i]].decl.domain == RAD_DOMAIN_DEVICE)
            prog_.ops[h].host_unsafe_device_buffers = true;""",
            ),
        ),
    ),
    FilePatch(
        path="core/place/planner.cpp",
        description="Stage dense weights on GPU when their operators use device-only activations.",
        edits=(
            replacement(
                "vtp11-site-signature",
                """static void resolve_sites(const Program& prog, std::vector<uint8_t>* host_ok,
                          std::vector<uint8_t>* device_ok) {""",
                """static void resolve_sites(const Program& prog, std::vector<uint8_t>* host_ok,
                          std::vector<uint8_t>* device_ok, bool host_backend) {
    /* BC_VTP11_HOST_DOMAIN: CPU backend allocations are host accessible. */""",
            ),
            replacement(
                "vtp11-site-eligibility",
                """        for (const Band& b : o.bands) {
            if (!b.dom[RAD_DOMAIN_HOST]) host = false;
            if (!b.dom[RAD_DOMAIN_DEVICE]) dev = false;
        }
        for (rad_weight w : o.weights) {""",
                """        for (const Band& b : o.bands) {
            if (!b.dom[RAD_DOMAIN_HOST]) host = false;
            if (!b.dom[RAD_DOMAIN_DEVICE]) dev = false;
        }
        /* BC_VTP11_HOST_DOMAIN: if a GPU kernel exists, stage weights rather
         * than run a CPU fallback against device-only activation buffers. If
         * none exists, keep the original host-only classification; the
         * bind-time guard refuses an unsafe launch with a clear diagnostic. */
        if (!host_backend && dev && o.host_unsafe_device_buffers) host = false;
        for (rad_weight w : o.weights) {""",
            ),
            replacement(
                "vtp11-budget-footprint",
                """    resolve_sites(prog, &host_ok, &device_ok);""",
                """    /* BC_VTP11_HOST_DOMAIN: the preflight budget must match the planner. */
    resolve_sites(prog, &host_ok, &device_ok, host_backend);""",
            ),
            replacement(
                "vtp11-planner-resolve",
                """    resolve_sites(prog, &host_ok_, &device_ok_);""",
                """    /* BC_VTP11_HOST_DOMAIN: the planner stages unsafe host-site layers. */
    resolve_sites(prog, &host_ok_, &device_ok_, in_.host_backend);""",
            ),
        ),
    ),
    FilePatch(
        path="core/runtime/ctx.cpp",
        description="Refuse unsafe host execution before GPU launch can dereference its buffers.",
        edits=(
            replacement(
                "vtp11-bind-op-guard",
                """            if (all_host) dom = RAD_DOMAIN_HOST;
        }
        op_domain_[i].store((uint8_t)dom, std::memory_order_relaxed);""",
                """            /* BC_VTP11_HOST_DOMAIN: the planner must never leave host weights
             * for an operator with GPU-only buffers. Refuse instead of
             * selecting AVX and crashing on a device output pointer. */
            if (all_host && !device_is_host() && oi.host_unsafe_device_buffers) {
                RAD_ERR("op '%s' has host-site weights but device-only activation buffers; "
                        "its GPU weights must be staged, not executed by a host kernel",
                        oi.op.c_str());
                return RAD_E_UNSUPPORTED;
            }
            if (all_host) dom = RAD_DOMAIN_HOST;
        }
        op_domain_[i].store((uint8_t)dom, std::memory_order_relaxed);""",
            ),
            replacement(
                "vtp11-explicit-domain-guard",
                """int Ctx::set_op_domain(rad_op h, int domain) {
    int s = check_domain(h, domain);""",
                """int Ctx::set_op_domain(rad_op h, int domain) {
    /* BC_VTP11_HOST_DOMAIN: an explicit override cannot bypass admission. */
    if (domain == RAD_DOMAIN_HOST && !device_is_host() && program_ &&
        h > 0 && (size_t)h < program_->ops.size() &&
        program_->ops[h].host_unsafe_device_buffers) return RAD_E_UNSUPPORTED;
    int s = check_domain(h, domain);""",
            ),
        ),
    ),
    FilePatch(
        path="core/runtime/issue.cpp",
        description="Fail closed if any host launch still names a GPU-only buffer.",
        edits=(replacement(
            "vtp11-launch-buffer-guard",
            """            tt = buf_tmpl_[bh];
            touches_host |= buf_host_[bh] != 0;""",
            """            /* BC_VTP11_HOST_DOMAIN: catch incomplete declarations or
             * later domain overrides before a host kernel touches GPU VRAM. */
            if (dom == RAD_DOMAIN_HOST && !device_is_host() && !buf_host_[bh])
                return abort_step_msg(h, RAD_E_UNSUPPORTED,
                                      fmt("host kernel operand %d uses device-only buffer '%s'",
                                          i, program_->buffers[bh].name.c_str()));
            tt = buf_tmpl_[bh];
            touches_host |= buf_host_[bh] != 0;""",
        ),),
    ),
)
