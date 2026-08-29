"""HI85: fail closed (not crash) when NCCL/RCCL's participants span
different GPU architectures.

Real dual/multi-GPU hardware testing on Brutus (4 GPUs: RDNA2 gfx1030,
2x RDNA3 gfx1100, RDNA4 gfx1201) found that RCCL's collective kernel
dispatch does not correctly handle a communicator whose ranks are on
different GPU architectures -- confirmed with NCCL_DEBUG=INFO tracing
against a genuinely RCCL-enabled build (GGML_HIP_RCCL=ON), reproduced
identically on two ROCm versions (7.2.4 and 7.14.0), and reproduced with
a minimal BigCherry-free reproducer against completely stock llama.cpp
(so this is a real RCCL/ROCm limitation, not a BigCherry defect):

  - Per-rank topology/tuning setup IS architecture-aware: NCCL_DEBUG
    shows a distinct "RCCL Tuning index" per rank matching each
    device's real architecture, and ncclCommInitAll_impl completes
    cleanly for every rank.
  - The crash happens later, inside ncclGroupEnd() when the actual
    collective kernel is launched: "HIP failure: invalid device
    function" on the non-matching rank's device. This is a HIP-level
    hard abort (SIGABRT), not a catchable ncclResult_t error code --
    the caller has no way to recover from it once ncclCommInitAll has
    already returned success and the collective is launched.

Every heterogeneous-architecture pair/group tested crashes this way,
regardless of device count (a 2-device pair crashes exactly like a
3-device group); only same-architecture participant sets (the real
production dual-XTX pair) work. Full findings:
docs/planning/active/hip-autotune/HI85.md.

This patch adds one guard, in the same place and following the same
pattern the existing virtual-device check already uses: before calling
ncclCommInitAll, compare every participating device's ggml_cuda_info()
compute-capability identifier (already populated from HIP's
prop.gcnArchName for AMD devices -- verified real values differ
correctly per architecture: gfx1100/gfx1201/gfx1030 each parse to a
distinct cc).

ggml_backend_cuda_comm_init_nccl() is only ever reached via the META
backend's comm_init hook (ggml-backend-meta.cpp), which llama.cpp only
constructs for LLAMA_SPLIT_MODE_TENSOR (llama-model.cpp's META device
wiring is exclusive to that split mode; -sm layer never touches
comm_ctx/META at all). So a heterogeneous-architecture participant
set reaching this guard always means the user explicitly requested
tensor-split -- there is no "auto, didn't really mean it" case here.

Per explicit user direction (2026-08-28): silently substituting
META's different reduction path when the user explicitly asked for
tensor-split is not acceptable -- correctness/performance semantics
differ, and a log-only warning is too easy to miss. So this guard
fails closed with a hard, clearly-diagnosed abort (GGML_ABORT) instead
of the uncatchable, undiagnosed HIP SIGABRT this item originally found
-- and instead of silently degrading to internal/META. The abort
message names the exact device architectures involved and points at
HI85 for the full explanation and remediation (force GGML_HIP_REDUCE_
PLAN=meta explicitly, or drop to -sm layer, or use a same-architecture
device subset).

Wording correction (2026-08-29, HI138/HI142): this guard's predicate is
raw architecture inequality, and its original message framed that as
the causal condition ("heterogeneous-architecture tensor-split is
unsupported"). HI138's real-hardware follow-up investigation proved
that framing is over-conservative: the actual RCCL failure mechanism is
a specific device's PCIe path lacking AtomicOps routing capability
(confirmed via AMDGPU kernel boot-time self-test + lspci + external
ROCm issue prior art), NOT architecture mismatch per se -- a real
architecture-mismatched set, {2x RDNA3 XTX + RDNA4 R9700} (devices
0,1,2 on this box), passed 20/20 live RCCL AllReduce repetitions
cleanly once the PCIe-atomics-incapable device (3, RDNA2 6900XT,
chipset-routed) was excluded. So architecture inequality is still used
as the trigger (it remains a reasonable, cheap proxy in the absence of
qualified per-topology evidence), but the abort message below is
reworded from "unsupported" to "unqualified ... prohibited" so it no
longer asserts a causal claim this investigation has since disproven.
This is a message-only correction -- the guard's behavior (abort on any
architecture mismatch) is intentionally unchanged: per HI142's design
(gpt-reviewed), production admission of a specific verified-safe
topology like {0,1,2} requires real P2 qualification evidence and an
explicit admission mechanism, not a loosened blanket predicate here.
See docs/reference/testing/RCCL_HETEROGENEOUS_RUNBOOK.md and
docs/planning/active/hip-autotune/HI142.md.
"""

GROUP = "core"
# Design verified against real source (ggml_cuda_device_info::cc is
# populated from prop.gcnArchName at ggml-cuda.cu:349, confirmed distinct
# per real architecture) before writing this patch. Not yet compiled or
# hardware-validated -- promote to "validated" once a Brutus build
# confirms a heterogeneous pair declines cleanly (falls back to
# internal/meta) instead of crashing under GGML_HIP_REDUCE_PLAN=auto.
STATE = "untested"

import re as _re

from bigcherry.patcher import Edit, FilePatch

# The patcher's anchor-matching runs against a noise-stripped copy of the
# source where string literals (including their quotes) are blanked to
# spaces of the same length -- so the two log-message string literals in
# this anchor must be matched as "any non-newline content", not their
# literal text (same technique patches/1222 uses for a blanked comment).
_ANCHOR_BLOCK = '''    if (info.device_count > info.physical_device_count) {
        GGML_LOG_WARN(LITERAL1
                      LITERAL2);
        ggml_backend_cuda_comm_init_internal(ret);
        return;
    }'''
_ANCHOR = (
    _re.escape(_ANCHOR_BLOCK)
    .replace(_re.escape("LITERAL1"), r"[^\n]*")
    .replace(_re.escape("LITERAL2"), r"[^\n]*")
)

_GUARD = '''
    // bigcherry (HI85/HI138): NCCL/RCCL's collective kernel dispatch does
    // not correctly handle a communicator whose ranks span different GPU
    // architectures ON THIS BOX'S DEVICE-3 (RDNA2, chipset-routed PCIe
    // path) -- confirmed via NCCL_DEBUG=INFO tracing against a genuinely
    // RCCL-enabled build: per-rank topology/tuning selection IS
    // architecture-aware (a distinct RCCL Tuning index per rank), but the
    // actual collective kernel launched inside ncclGroupEnd() aborts the
    // process with a HIP "invalid device function"/hipErrorIllegalState
    // error on the non-matching rank, rather than returning a catchable
    // NCCL error code. HI138's follow-up investigation localized the real
    // mechanism to a PCIe AtomicOps routing-capability gap on one specific
    // device's PCIe path (confirmed via AMDGPU kernel boot-time
    // self-test + lspci + external ROCm issue prior art), not architecture
    // mismatch per se -- a real architecture-mismatched set (2x RDNA3 +
    // RDNA4) passed 20/20 live repetitions once the PCIe-atomics-incapable
    // device was excluded. Architecture inequality remains this guard's
    // trigger (a cheap, still-useful proxy given no qualified per-topology
    // admission mechanism exists yet -- see HI142), but detect it BEFORE
    // calling ncclCommInitAll regardless. This path is only ever reached
    // via SPLIT_MODE_TENSOR (the META backend's comm_init hook), so
    // reaching here always means the user explicitly asked for
    // tensor-split -- silently substituting a different reduction path
    // (internal/META) would change semantics behind the user's back. Fail
    // closed with a clear, named abort instead of both the original
    // uncatchable HIP SIGABRT and a silent fallback. See
    // docs/planning/active/hip-autotune/HI85.md and HI138.md for the full
    // investigation and real hardware evidence.
    bool heterogeneous_arch = false;
    for (size_t i = 1; i < ret->dev_ids.size(); ++i) {
        if (info.devices[ret->dev_ids[i]].cc != info.devices[ret->dev_ids[0]].cc) {
            heterogeneous_arch = true;
            break;
        }
    }
    if (heterogeneous_arch) {
        GGML_LOG_ERROR("NCCL/RCCL heterogeneous-architecture tensor-split has no "
                       "qualified admission for this device set -- this combination "
                       "was requested via SPLIT_MODE_TENSOR and will not be silently "
                       "downgraded to a different reduction path. See HI85/HI138/"
                       "HI142. Remediation: force GGML_HIP_REDUCE_PLAN=meta "
                       "explicitly, use -sm layer instead, or restrict tensor-split "
                       "to a same-architecture device subset.\\n");
        GGML_ABORT("unqualified heterogeneous-architecture tensor-split is prohibited (see HI85/HI138)");
    }
'''

PATCH = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="fail closed instead of crashing when NCCL/RCCL participants span different GPU architectures (HI85)",
    edits=(
        Edit(
            id="hi85-nccl-heterogeneous-arch-guard",
            anchor=_ANCHOR,
            mode="insert_after",
            rationale="right after the existing virtual-device decline check, same function, same "
                       "pattern -- both checks must run before ncclCommInitAll is ever called",
            text=_GUARD,
            guard=r"heterogeneous_arch",
        ),
    ),
)

PATCHES = [PATCH]
