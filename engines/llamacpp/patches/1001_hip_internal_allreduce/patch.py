"""Upstream backport: enable the internal (non-RCCL) AllReduce on HIP.

Cherry-picked from an unmerged upstream PR
(https://github.com/ggml-org/llama.cpp/pull/27825, still ``OPEN`` as of this
writing). ``allreduce.cu`` implements a low-latency pinned-host-memory
AllReduce for two-GPU tensor-split inference, used as an alternative to the
meta backend's RCCL/NCCL collective path -- but the whole implementation is
compiled out under ``GGML_USE_HIP``, so it has only ever run on CUDA.

Real dual-XTX profiling on this project's own hardware (rocprofv3, decode-
window kernel trace, ``bigcherry completion-bench``) found NCCL/RCCL
consuming 9.9% of decode wall time (union-of-spans, not summed durations),
with only 3.3% of that overlapping matmul compute -- i.e. ~9.6% of decode
wall is pure inter-GPU communication with zero concurrent compute, at a call
rate (~5186 collectives/sec, ~19.2us union time/call) that is exactly the
small-collective-latency profile this internal path targets. The PR author's
own upstream numbers (2x AMD PCIe, RX 9070 + RX 6800XT) show +2.24% TG over
the meta/RCCL path.

Two functional gaps this patch closes:

* Host-mapped pinned-memory APIs (``cudaHostAlloc``/``cudaHostAllocPortable``/
  ``cudaHostAllocMapped``/``cudaHostGetDevicePointer``) -- these exist on HIP
  under their ``hip*`` names (confirmed against ROCm 7.2.4 docs), just
  weren't mapped in ``vendors/hip.h``.
* A device-side sleep for the cross-GPU spin-wait -- CUDA uses ``__nanosleep``
  (sm70+), which HIP lacks; the PR substitutes
  ``__builtin_amdgcn_s_sleep(4)`` (~100ns at a 2500MHz-clock assumption),
  available across RDNA/CDNA/Vega/GCN3+.

Deliberately **not** ported: the PR's several comment-only wording changes
(the file-header comment, the closing #endif's trailing comment, the
compute-capability-gate comment, "devices[] holds the GPU device IDs"
instead of "CUDA device IDs", etc.). Those touch no compiled behaviour, so
this patch stays scoped to the two functional edits above -- narrower than
the upstream diff, same effect. (They are also awkward to anchor: this
project's patch matcher blanks comment text before anchor matching, so a
comment-only anchor can't target them without matching on adjacent code
instead -- not worth the anchor fragility for a change with zero runtime
effect.)

``GGML_CUDA_ALLREDUCE=internal`` (an existing env selector already present in
the pinned base, see ``ggml-cuda.cu``) picks this path over RCCL at runtime,
so both providers can be A/B'd against the same binary once this patch makes
the internal path buildable on HIP at all.
"""

GROUP = "upstream-fixes"
STATE = "untested"

from bigcherry.patcher import Edit, FilePatch

CU = FilePatch(
    path="ggml/src/ggml-cuda/allreduce.cu",
    description="enable the internal AllReduce implementation under "
                "GGML_USE_HIP (upstream PR #27825)",
    edits=(
        Edit(
            id="enable-hip-compile-guard",
            anchor=r"^#if !defined\(GGML_USE_HIP\) && !defined\(GGML_USE_MUSA\)$",
            rationale="the top-of-file guard that currently compiles the "
                      "whole internal AllReduce out under HIP",
            mode="replace",
            text="#if !defined(GGML_USE_MUSA)",
            guard=r"^#if !defined\(GGML_USE_MUSA\)$",
        ),
        Edit(
            id="spin-wait-hip-sleep-intrinsic",
            anchor=(
                r"while \(ggml_cuda_ar_signal_get\(other_slot\) != token\) \{\n"
                r"#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA\n"
                r"            __nanosleep\(100\);\n"
                r"#else\n"
                r"            NO_DEVICE_CODE;\n"
                r"#endif"
            ),
            rationale="the cross-GPU spin-wait's device-side sleep, CUDA-only "
                      "before this patch",
            mode="replace",
            text=(
                "while (ggml_cuda_ar_signal_get(other_slot) != token) {\n"
                "#ifdef GGML_USE_HIP\n"
                "            // Equals ~100ns at 2500 MHz (sleeps for n * [1,64] clock cycles)\n"
                "            __builtin_amdgcn_s_sleep(4);\n"
                "#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA\n"
                "            __nanosleep(100);\n"
                "#else\n"
                "            NO_DEVICE_CODE;\n"
                "#endif"
            ),
            guard=r"#ifdef GGML_USE_HIP\n            // Equals ~100ns at 2500 MHz",
        ),
    ),
)

HIP_H = FilePatch(
    path="ggml/src/ggml-cuda/vendors/hip.h",
    description="map the host-mapped pinned-memory alloc APIs the internal "
                "AllReduce needs (upstream PR #27825)",
    edits=(
        Edit(
            id="map-host-alloc-apis",
            anchor=r"^#define cudaGetLastError hipGetLastError$",
            rationale="alongside the existing cudaGetLastError mapping, "
                      "ahead of the existing cudaHostRegister family",
            mode="insert_after",
            text=(
                "\n"
                "#define cudaHostAlloc hipHostMalloc\n"
                "#define cudaHostAllocPortable hipHostMallocPortable\n"
                "#define cudaHostAllocMapped hipHostMallocMapped\n"
                "#define cudaHostGetDevicePointer hipHostGetDevicePointer"
            ),
            guard=r"#define cudaHostAllocMapped hipHostMallocMapped",
        ),
    ),
)

PATCHES = [CU, HIP_H]
