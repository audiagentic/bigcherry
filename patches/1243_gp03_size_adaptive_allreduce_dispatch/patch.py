"""GP03: size-adaptive per-call provider dispatch for AllReduce -- eliminate
the internal-path pp regression while keeping its tg decode win.

Requires 1001_hip_internal_allreduce (this patch's secondary NCCL bring-up
happens inside ggml_backend_cuda_comm_init_internal(), which only ever
succeeds -- ret->ar_pipeline non-null -- once 1001 makes the internal
pipeline buildable under GGML_USE_HIP at all; without 1001 this patch's
added code is unreachable dead weight, confirmed empirically during
development: ar_pipeline_init() returned nullptr on every call against an
unpatched base, so the secondary-init block below never executed).

patches/1001_hip_internal_allreduce/SUMMARY.md documents the internal
provider's real hardware tradeoff on this project's dual-XTX box: a
consistent decode win (tg128 +6.9%) but a severe prefill regression
(pp512/pp2048/pp4096 -32% to -34%) versus RCCL. Both numbers come from the
SAME committed-at-init provider choice (GGML_CUDA_ALLREDUCE=internal picks
one function pointer for every reduction in the session) -- there was no
per-call distinction between a tiny tg-sized reduction and a large pp-sized
one.

This patch does not touch the internal path's own algorithm. It adds a
per-call override on top of the existing GGML_HIP_DISPATCH reduce-plan
mechanism (ggml_backend_cuda_comm_reduce_plan / ..._try_reduce_plan,
already present in the pinned base, gated by #ifdef GGML_HIP_DISPATCH):

* ggml_backend_cuda_comm_init_internal() additionally brings up a secondary
  NCCL/RCCL communicator set (best-effort; a failure here must not affect
  the primary internal path, which stays the committed default via
  comm_ctx->try_allreduce) alongside the internal ar_pipeline, so a per-call
  "rccl" override actually has real communicators to use.
* ggml_backend_cuda_comm_reduce_plan() becomes size-adaptive: reductions at
  or above GGML_HIP_REDUCE_RCCL_THRESHOLD bytes (default 256 KiB) resolve to
  "rccl" instead of the internal-path default, everything smaller stays on
  "auto" (i.e. whatever comm_ctx->try_allreduce is -- internal, once 1001 is
  applied). GGML_HIP_REDUCE_PLAN keeps working as an explicit override,
  taking priority over the size heuristic.
* ggml_backend_cuda_comm_allreduce_tensor()'s call site now computes the
  real byte count of the reduction and passes it to the plan function
  instead of calling it with no arguments.

Real-hardware validation (Brutus, 2x RX 7900 XTX, HIP_VISIBLE_DEVICES=0,1,
GGML_CUDA_ALLREDUCE=internal, GGML_CUDA_AR_BF16_THRESHOLD=0, -sm tensor -fa
on -b 2048 -ub 512, r=3), against patch 1001's own SUMMARY.md baselines:

    metric   RCCL baseline   internal-only baseline   this patch (dispatch)
    pp512    1504.67         998.21  (-33.7%)          1502.49  (matches RCCL)
    pp2048   1447.58         980.64  (-32.3%)           1451.17  (matches RCCL)
    pp4096   1431.77         972.08  (-32.1%)           1435.27  (matches RCCL)
    tg128    33.69            36.01  (+6.9%)             37.23  (+10.5%, decode
                                                                  win retained
                                                                  and improved)

Also validated on the RCCL-viable heterogeneous 2-GPU subset (both
CPU-direct PCIe root ports, no device-3 involvement): {0,2} (XTX+R9700,
gfx1100+gfx1201) pp512=1205.82/tg32=29.58, {1,2} pp512=1283.78/tg32=30.46 --
both clean, dispatch worked as designed, no crash. Full detail:
docs/planning/active/gpu-collectives/GP03.md.

**Known unresolved safety gap -- why this patch ships STATE="untested" and
not "validated"**: the secondary NCCL communicator this patch brings up is
a second, independent RCCL entry point that does not consult patch 1225's
architecture guard (which only protects the ORIGINAL ggml_backend_cuda_comm
_init_nccl() call site) or any topology qualification list. Tested directly
against a topology including physical device 3 (RX 6900XT, the box's
chipset-routed GPU that HI138 confirmed permanently lacks PCIe AtomicOps
completion capability): ncclCommInitAll() reports spurious success (rc=0,
comms populated) because init alone doesn't exercise the atomics path, so
this patch's own admissibility check (comms.size() == backends.size())
passes -- and the process then hard-aborts inside
ggml_backend_cuda_comm_allreduce_nccl on the first real pp-sized reduction.
This is not a soft fallback; it is a crash, and it is strictly worse than
not having this patch at all on any device-3-inclusive topology (without
this patch, GGML_CUDA_ALLREDUCE=internal on such a topology just runs the
internal path the whole time and works fine).

This patch MUST NOT be promoted to a default patch-set until one of:
(a) docs/planning/active/gpu-collectives/GP02.md lands and this patch is
    made to consult the same guard before calling ncclCommInitAll here, or
(b) this patch gets its own equivalent topology check inlined before the
    secondary init.
See docs/reference/testing/RCCL_HETEROGENEOUS_RUNBOOK.md (2026-09-02
addendum) for the full evidence trail.
"""

GROUP = "upstream-fixes"
STATE = "untested"

from bigcherry.patcher import Edit, FilePatch

CU = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="size-adaptive per-call AllReduce provider dispatch: route "
                "large (pp-sized) reductions to RCCL, keep small (tg-sized) "
                "reductions on the internal path (GP03)",
    edits=(
        Edit(
            id="gp03-secondary-nccl-bringup",
            # Anchors match against a noise-stripped copy of the source where
            # string literals -- QUOTES INCLUDED -- are blanked to spaces of
            # the same length, so the "internal" literal below is matched as
            # a bare [^\n]* with no quote characters at all, not as a quoted
            # wildcard (same technique patches/1225 uses for a blanked
            # comment).
            anchor=(
                r"    if \(ret->ar_pipeline\) \{\n"
                r"        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;\n"
                r"        ret->provider_name = [^\n]*;\n"
                r"        return;\n"
                r"    \}"
            ),
            rationale="bring up a secondary NCCL communicator set alongside "
                      "the internal pipeline (not instead of it) so the "
                      "per-call size-adaptive override has real "
                      "communicators to use; the internal path stays the "
                      "committed default via comm_ctx->try_allreduce, and a "
                      "failure here must not affect it",
            mode="replace",
            text=(
                "    if (ret->ar_pipeline) {\n"
                "        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;\n"
                "        ret->provider_name = \"internal\";\n"
                "        // GP03 (pp-regression dispatch): best-effort bring up NCCL communicators\n"
                "        // ALONGSIDE the internal pipeline (not instead of it), so a real\n"
                "        // per-call size-based \"rccl\" override actually has comms to use.\n"
                "        // Internal stays the committed default (comm_ctx->try_allreduce);\n"
                "        // a failure here must not affect the primary internal path at all.\n"
                "        // KNOWN GAP (see GP02/GP03): this does not consult the device-3\n"
                "        // architecture guard -- do not enable on a topology that includes\n"
                "        // the box's PCIe-atomics-incapable device until GP02 lands.\n"
                "#ifdef GGML_USE_NCCL\n"
                "        const ggml_cuda_device_info & nccl_info = ggml_cuda_info();\n"
                "        if (nccl_info.device_count <= nccl_info.physical_device_count) {\n"
                "            const size_t n = ret->dev_ids.size();\n"
                "            ret->comms.resize(n);\n"
                "            ncclResult_t rc = ncclCommInitAll(ret->comms.data(), (int) n, ret->dev_ids.data());\n"
                "            if (rc != ncclSuccess) {\n"
                "                ret->comms.clear();\n"
                "                GGML_LOG_WARN(\"GP03: secondary NCCL init for size-adaptive dispatch failed (%s); \"\n"
                "                              \"large reductions will stay on the internal path\\n\",\n"
                "                              ncclGetErrorString(rc));\n"
                "            }\n"
                "        }\n"
                "#endif // GGML_USE_NCCL\n"
                "        return;\n"
                "    }"
            ),
            guard=r"GP03 \(pp-regression dispatch\)",
        ),
        Edit(
            id="gp03-size-adaptive-reduce-plan",
            # Same string-literal-blanking caveat as the edit above -- every
            # quoted literal in this anchor (quotes included) becomes a bare
            # [^\n]*.
            anchor=(
                r"static const char \* ggml_backend_cuda_comm_reduce_plan\(\) \{\n"
                r"    const char \* env = getenv\([^\n]*\);\n"
                r"    if \(env == nullptr \|\| strcmp\(env,[^\n]*\) == 0 \|\|\n"
                r"            strcmp\(env,[^\n]*\) == 0 \|\| strcmp\(env,[^\n]*\) == 0\) \{\n"
                r"        return env == nullptr \?[^\n]*: env;\n"
                r"    \}\n"
                r"    GGML_LOG_WARN\([^\n]*, env\);\n"
                r"    return [^\n]*;\n"
                r"\}"
            ),
            rationale="replace the always-'auto'-unless-overridden plan "
                      "function with a size-adaptive one -- large "
                      "reductions default to rccl, small ones keep the "
                      "existing auto (internal) behaviour; explicit "
                      "GGML_HIP_REDUCE_PLAN overrides still take priority",
            mode="replace",
            text=(
                "static size_t ggml_backend_cuda_comm_reduce_rccl_threshold() {\n"
                "    static const size_t threshold = [] () -> size_t {\n"
                "        const char * t = getenv(\"GGML_HIP_REDUCE_RCCL_THRESHOLD\");\n"
                "        if (t != nullptr) {\n"
                "            char * end = nullptr;\n"
                "            unsigned long long v = strtoull(t, &end, 10);\n"
                "            if (end != t && *end == 0) {\n"
                "                return (size_t) v;\n"
                "            }\n"
                "            GGML_LOG_WARN(\"invalid GGML_HIP_REDUCE_RCCL_THRESHOLD value: %s; using default\\n\", t);\n"
                "        }\n"
                "        return (size_t) (256 * 1024); // 256 KB default\n"
                "    }();\n"
                "    return threshold;\n"
                "}\n"
                "\n"
                "// GP03 (pp-regression fix): size-adaptive default. The internal AllReduce\n"
                "// path (patch 1001) is a real decode win (+17% tg) but a severe prefill\n"
                "// regression (-32% to -34% pp) on this hardware -- reductions at or above\n"
                "// the threshold route to rccl (if the secondary NCCL init succeeded)\n"
                "// instead of the committed internal provider. Validate the default\n"
                "// threshold before trusting it in production.\n"
                "static const char * ggml_backend_cuda_comm_reduce_plan(size_t nbytes) {\n"
                "    const char * env = getenv(\"GGML_HIP_REDUCE_PLAN\");\n"
                "    if (env != nullptr) {\n"
                "        if (strcmp(env, \"auto\") == 0 || strcmp(env, \"rccl\") == 0 || strcmp(env, \"meta\") == 0) {\n"
                "            return env;\n"
                "        }\n"
                "        GGML_LOG_WARN(\"unknown GGML_HIP_REDUCE_PLAN value: %s; using size-adaptive auto\\n\", env);\n"
                "    }\n"
                "    return nbytes >= ggml_backend_cuda_comm_reduce_rccl_threshold() ? \"rccl\" : \"auto\";\n"
                "}"
            ),
            guard=r"ggml_backend_cuda_comm_reduce_rccl_threshold",
        ),
        Edit(
            id="gp03-pass-nbytes-to-reduce-plan",
            anchor=(
                r"    const char \* requested_provider = ggml_backend_cuda_comm_reduce_plan\(\);"
            ),
            rationale="reduce_plan() now takes the real reduction byte "
                      "count to make its size-adaptive decision",
            mode="replace",
            text=(
                "    const size_t requested_nbytes = tensors[0] != nullptr ? ggml_nbytes(tensors[0]) : 0;\n"
                "    const char * requested_provider = ggml_backend_cuda_comm_reduce_plan(requested_nbytes);"
            ),
            guard=r"requested_nbytes",
        ),
    ),
)

PATCHES = [CU]
