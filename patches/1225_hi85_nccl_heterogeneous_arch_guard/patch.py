"""HI85/GP02: fail closed (not crash) when a NCCL/RCCL participant lacks
PCIe AtomicOps completion capability.

Real dual/multi-GPU hardware testing on Brutus (4 GPUs: RDNA2 gfx1030,
2x RDNA3 gfx1100, RDNA4 gfx1201) found that RCCL's collective kernel
dispatch hard-crashes when a communicator includes a device whose PCIe
path cannot complete AtomicOps -- confirmed with NCCL_DEBUG=INFO tracing
against a genuinely RCCL-enabled build (GGML_HIP_RCCL=ON), reproduced
identically on two ROCm versions (7.2.4 and 7.14.0), and reproduced with
a minimal BigCherry-free reproducer against completely stock llama.cpp
(so this is a real RCCL/ROCm/hardware limitation, not a BigCherry defect):

  - Per-rank topology/tuning setup succeeds; ncclCommInitAll_impl
    completes cleanly for every rank, reporting success.
  - The crash happens later, inside ncclGroupEnd() when the actual
    collective kernel is launched: "HIP failure: invalid device
    function" / hipErrorIllegalState on the PCIe-limited rank's device.
    This is a HIP-level hard abort, not a catchable ncclResult_t error
    code -- the caller has no way to recover from it once
    ncclCommInitAll has already returned success.

HI138's real-hardware follow-up localized the exact mechanism: RCCL's
generic device kernel declares a `hidden_hostcall_buffer` kernel argument,
and hostcall requires PCIe AtomicOps completion support from the target
device -- confirmed via `lspci -vvv` AtomicOpsCap/Ctl inspection and the
AMDGPU kernel driver's own boot-time self-test (dmesg: "PCIE atomic ops is
not supported"). This is a property of ONE SPECIFIC DEVICE'S PCIe path
(on this box: physical device 3, whose upstream root port is chipset-
routed), not a property of heterogeneous-architecture communicators in
general -- a real architecture-mismatched set (2x RDNA3 XTX + RDNA4
R9700, devices 0,1,2 on this box) passed 20/20 live RCCL AllReduce
repetitions cleanly once the PCIe-atomics-incapable device was excluded.

GP02 rewrite (2026-09-02, gpt-dev-agent review): the ORIGINAL version of
this guard used raw GPU-architecture inequality as a cheap proxy for the
real condition, because no better runtime signal was known at the time.
That proxy is now confirmed WRONG in a way that actively blocks real,
qualified work: it would reject the {0,2}/{1,2} XTX+R9700 pair that
HI138/GP06 spent real hardware evidence confirming is RCCL-safe, purely
because gfx1100 != gfx1201, with no relationship to the actual PCIe-
atomics limitation.

Real fix: HIP exposes the actual fact directly, per device, at runtime --
`hipDeviceAttributeHostNativeAtomicSupported` ("Link between the device
and host supports native atomic operations"). Verified on real hardware
(2026-09-02): returns 1 for devices 0/1/2 (all CPU-direct root ports) and
0 for device 3 (chipset-routed) on this exact box, with NO ordinal
hardcoded -- this queries each device's own real capability via the
standard HIP API, so it stays correct on different hardware/slots rather
than encoding "device 3 is bad" as a fact about this one machine (which
the project's own topology-identity rules explicitly forbid -- see
docs/reference/testing/RCCL_HETEROGENEOUS_RUNBOOK.md safety invariant 9).

Shared, reusable: the admission check is a standalone function defined
once, callable from EVERY ncclCommInitAll() entry point in the tree --
not just this guard's original call site (ggml_backend_cuda_comm_init_
nccl(), reached only via SPLIT_MODE_TENSOR / the META backend's comm_init
hook). Patch 0840 (GP03's consolidated hybrid dispatch) brings up its own
independent secondary NCCL communicator and calls this same function
before doing so -- confirmed necessary by direct testing: without this,
0840's own ncclCommInitAll() reports spurious success on a device-3-
inclusive topology and then hard-crashes on the first real collective,
identical to this guard's original finding, just through a different
entry point.

Per explicit user direction (2026-08-28, still applies): silently
substituting META's different reduction path when the user explicitly
asked for tensor-split is not acceptable -- correctness/performance
semantics differ, and a log-only warning is too easy to miss. This guard
fails closed with a hard, clearly-diagnosed abort (GGML_ABORT) instead of
the uncatchable, undiagnosed HIP SIGABRT the original investigation
found. See docs/reference/testing/RCCL_HETEROGENEOUS_RUNBOOK.md and
docs/planning/active/gpu-collectives/GP02.md for the full investigation
and real hardware evidence.
"""

GROUP = "core"
# Real hardware validated (2026-09-02): hipDeviceAttributeHostNativeAtomicSupported
# correctly distinguishes device 3 (=0) from devices 0/1/2 (=1) on Brutus,
# confirmed via a standalone HIP probe binary against real hardware, matching
# HI138's lspci/dmesg-based finding exactly. Promote from the prior
# architecture-inequality version's "untested" once a build with this guard
# confirms {0,3}/{0,1,3} fail closed with the new message and {0,2}/{1,2}
# (the topology this rewrite specifically unblocks) succeed.
STATE = "untested"

import re as _re

from bigcherry.patcher import Edit, FilePatch

# The patcher's anchor-matching runs against a noise-stripped copy of the
# source where string literals (including their quotes) are blanked to
# spaces of the same length -- so quoted literals in these anchors must be
# matched as "any non-newline content", not their literal text.
_VIRTUAL_DEVICE_ANCHOR_BLOCK = '''    if (info.device_count > info.physical_device_count) {
        GGML_LOG_WARN(LITERAL1
                      LITERAL2);
        ggml_backend_cuda_comm_init_internal(ret);
        return;
    }'''
_VIRTUAL_DEVICE_ANCHOR = (
    _re.escape(_VIRTUAL_DEVICE_ANCHOR_BLOCK)
    .replace(_re.escape("LITERAL1"), r"[^\n]*")
    .replace(_re.escape("LITERAL2"), r"[^\n]*")
)

_ADMISSION_FUNCTION = '''
// GP02: shared, reusable, ordinal-independent RCCL admission check.
// Queries each participating device's REAL PCIe-atomics capability via
// the standard HIP API (hipDeviceAttributeHostNativeAtomicSupported) --
// confirmed on real hardware (2026-09-02) to correctly identify the one
// PCIe-limited device on this box with no ordinal hardcoded, unlike this
// guard's original architecture-inequality proxy. Callable from every
// ncclCommInitAll() entry point in the tree; see HI85/HI138/GP02.
static bool ggml_backend_cuda_comm_rccl_admission_ok(const int * dev_ids, size_t n_devices) {
    for (size_t i = 0; i < n_devices; ++i) {
        int supported = 0;
        hipError_t rc = hipDeviceGetAttribute(&supported, hipDeviceAttributeHostNativeAtomicSupported, dev_ids[i]);
        if (rc != hipSuccess || !supported) {
            return false;
        }
    }
    return true;
}
'''

_GUARD = '''
    // bigcherry (HI85/HI138/GP02): NCCL/RCCL's collective kernel dispatch
    // hard-crashes when a communicator includes a device whose PCIe path
    // cannot complete AtomicOps (hostcall requirement) -- confirmed via
    // NCCL_DEBUG=INFO tracing against a genuinely RCCL-enabled build:
    // per-rank topology/tuning selection succeeds and ncclCommInitAll
    // reports success, but the actual collective kernel launched inside
    // ncclGroupEnd() aborts the process with a HIP "invalid device
    // function"/hipErrorIllegalState error on the PCIe-limited rank,
    // rather than returning a catchable NCCL error code. Check the REAL
    // per-device capability (ggml_backend_cuda_comm_rccl_admission_ok,
    // defined above) BEFORE calling ncclCommInitAll, not architecture
    // equality -- a real architecture-mismatched-but-PCIe-capable set
    // (2x RDNA3 + RDNA4) passed 20/20 live repetitions cleanly. This path
    // is only ever reached via SPLIT_MODE_TENSOR (the META backend's
    // comm_init hook), so reaching here always means the user explicitly
    // asked for tensor-split -- silently substituting a different
    // reduction path (internal/META) would change semantics behind the
    // user's back. Fail closed with a clear, named abort instead of both
    // the original uncatchable HIP SIGABRT and a silent fallback. See
    // docs/planning/active/hip-autotune/HI85.md, HI138.md, and
    // docs/planning/active/gpu-collectives/GP02.md for the full
    // investigation and real hardware evidence.
    if (!ggml_backend_cuda_comm_rccl_admission_ok(ret->dev_ids.data(), ret->dev_ids.size())) {
        GGML_LOG_ERROR("NCCL/RCCL admission check failed -- at least one participating "
                       "device lacks PCIe AtomicOps completion capability, which RCCL's "
                       "collective kernel requires. This combination was requested via "
                       "SPLIT_MODE_TENSOR and will not be silently downgraded to a "
                       "different reduction path. See HI85/HI138/GP02. Remediation: "
                       "force GGML_HIP_REDUCE_PLAN=meta explicitly, use -sm layer "
                       "instead, or restrict tensor-split to a PCIe-atomics-capable "
                       "device subset.\\n");
        GGML_ABORT("RCCL admission check failed: a participating device lacks PCIe AtomicOps completion capability (see HI85/HI138/GP02)");
    }
'''

CUDA = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="fail closed (not crash) when a NCCL/RCCL participant lacks "
                "PCIe AtomicOps completion capability -- real per-device "
                "runtime check, not an architecture-equality proxy (HI85/GP02)",
    edits=(
        Edit(
            id="gp02-admission-function",
            anchor=r"static void ggml_backend_cuda_comm_init_none\(ggml_backend_cuda_comm_context \* ret\) \{",
            mode="insert_before",
            rationale="defined once, before every comm_init_* function, so "
                      "both this guard's own call site and 0840's "
                      "independent hybrid init can reference it regardless "
                      "of patch application order",
            text=_ADMISSION_FUNCTION,
            guard=r"ggml_backend_cuda_comm_rccl_admission_ok\(",
        ),
        Edit(
            id="hi85-nccl-admission-guard",
            anchor=_VIRTUAL_DEVICE_ANCHOR,
            mode="insert_after",
            rationale="right after the existing virtual-device decline check, same function, same "
                       "pattern -- both checks must run before ncclCommInitAll is ever called",
            text=_GUARD,
            guard=r"ggml_backend_cuda_comm_rccl_admission_ok\(ret->dev_ids",
        ),
    ),
)

PATCHES = [CUDA]
