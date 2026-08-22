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
distinct cc). A mismatch logs a clear warning and declines to
internal AllReduce (which itself may further decline to META for
device counts it doesn't support) BEFORE the crash-prone
ncclCommInitAll call, rather than letting the process abort with no
diagnostic path. This makes AUTO's RCCL-first default safe for a
heterogeneous participant set instead of crashing the whole process;
forced GGML_HIP_REDUCE_PLAN=meta already worked correctly for these
same participant sets (HI18/HI84's own probe evidence) and remains
unaffected by this patch.
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
    // bigcherry (HI85): NCCL/RCCL's collective kernel dispatch does not
    // correctly handle a communicator whose ranks span different GPU
    // architectures -- confirmed on real heterogeneous hardware (RDNA2/
    // RDNA3/RDNA4 mix) via NCCL_DEBUG=INFO tracing against a genuinely
    // RCCL-enabled build: per-rank topology/tuning selection IS
    // architecture-aware (a distinct RCCL Tuning index per rank), but the
    // actual collective kernel launched inside ncclGroupEnd() aborts the
    // process with a HIP "invalid device function" error on the
    // non-matching rank, rather than returning a catchable NCCL error
    // code. Detect a mixed-architecture participant set BEFORE calling
    // ncclCommInitAll and decline explicitly -- the alternative is an
    // uncatchable process crash. See docs/planning/active/hip-autotune/
    // HI85.md for the full investigation and real hardware evidence.
    bool heterogeneous_arch = false;
    for (size_t i = 1; i < ret->dev_ids.size(); ++i) {
        if (info.devices[ret->dev_ids[i]].cc != info.devices[ret->dev_ids[0]].cc) {
            heterogeneous_arch = true;
            break;
        }
    }
    if (heterogeneous_arch) {
        GGML_LOG_WARN("NCCL disabled: participating devices have different GPU "
                      "architectures (NCCL/RCCL cannot reliably reduce across "
                      "mixed architectures on this hardware/driver -- see HI85); "
                      "falling back to internal AllReduce\\n");
        ggml_backend_cuda_comm_init_internal(ret);
        return;
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
