"""HI18: standalone SPLIT_REDUCE correctness probe (test-hip-reduce).

Wires the native half of HI18's correctness-comparison gate into the build:
a standalone host executable that drives the REAL production META backend
(ggml_backend_meta_device -- the same construction llama.cpp's own
LLAMA_SPLIT_MODE_TENSOR path uses) through GGML_HIP_REDUCE_PLAN=auto|rccl|
meta with a minimal 2-node graph, and reports machine-readable execution
facts for tools/bigcherry/reduce_correctness.py to evaluate.

Design verified against real vendor source before writing this patch
(2026-08-22): ggml_backend_meta_device()/split-state API (ggml-backend.h),
the real axis0+axis0 MUL_MAT -> PARTIAL/MIRRORED rule and the PARTIAL-
triggered subgraph terminator (ggml-backend-meta.cpp's handle_mul_mat() and
its graph-compute partitioner), the COMPUTE-buffer-usage gate on whether
META invokes the split-state callback, and the private-vs-exported split
between ggml_backend_meta_buffer_simple_tensor (static, unreachable) and
ggml_backend_meta_simple_backend (exported) -- which is why per-device
output readback reuses the existing HI58 telemetry tensor-array capture
seam (hip-autotune-reduce-telemetry.h) rather than a new generic accessor.

The probe source lives in the overlay (src/tests/test-hip-reduce.cpp); this
patch only wires the build target, next to the other host tools that link
directly against ggml/ggml-backend (patches/1230's hip-autotune-inspect is
the precedent for this pattern -- llama_build()'s default `llama` +
`llama-common` linkage, plus one extra target_link_libraries for
vendor::hash, matches what nlohmann::json and the SHA-256 digest helper
this probe uses already need).

HI18 D=2 slice only -- this probe's own main() refuses to run with a device
count other than 2, even though its construction mechanics are device-
count-generic (see HI84 for the planned D=3/D=4 extension, which relaxes
that guard rather than rewriting the probe).
"""

GROUP = "core"
# Mechanism compiled and hardware-validated on Brutus (63/63 synthetic
# RCCL/META/AUTO executions across 6 numerical patterns, 3 seeds, and
# swapped device order -- see docs/planning/active/hip-autotune/HI18.md).
# STATE remains "untested" under the HI83 patch-validation contract until
# the required tracked validation record exists AND the real recorded
# production signatures (not this synthetic stand-in shape) have been run.
STATE = "untested"

from bigcherry.patcher import Edit, FilePatch

CMAKE = FilePatch(
    path="tests/CMakeLists.txt",
    description="build the standalone test-hip-reduce SPLIT_REDUCE correctness probe",
    edits=(
        Edit(
            id="hi18-reduce-probe-executable",
            anchor=(r"^if \(NOT GGML_BACKEND_DL\)$"
                    r"\n(?:.*\n)*?"
                    r"^endif\(\)$"),
            mode="insert_after",
            rationale="next to the other tests that link the backends directly and cannot "
                       "be built with dynamic loading",
            text=(
                "\n"
                "# bigcherry: HI18 SPLIT_REDUCE correctness probe (patches/1224). Drives\n"
                "# the real production META backend through GGML_HIP_REDUCE_PLAN; see\n"
                "# src/tests/test-hip-reduce.cpp for the full design rationale.\n"
                "if (NOT GGML_BACKEND_DL AND (GGML_HIP_AUTOTUNE OR GGML_HIP_DISPATCH_REPLAY))\n"
                "    llama_build(test-hip-reduce.cpp)\n"
                "    target_link_libraries(test-hip-reduce PRIVATE vendor::hash)\n"
                "    target_include_directories(test-hip-reduce PRIVATE\n"
                "        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/src)\n"
                "    # GGML_HIP_DISPATCH is only added at ggml/src/ggml-hip's directory\n"
                "    # scope (patches/0100_cmake_options/patch.py), which does not propagate to\n"
                "    # this sibling tests/ directory -- without this, the telemetry header's\n"
                "    # #if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH) guard hides\n"
                "    # the test-capture seam this probe depends on.\n"
                "    target_compile_definitions(test-hip-reduce PRIVATE GGML_HIP_DISPATCH)\n"
                "endif()\n"
            ),
            guard=r"add_executable\(test-hip-reduce|llama_build\(test-hip-reduce\.cpp\)",
        ),
    ),
)

PATCHES = [CMAKE]
