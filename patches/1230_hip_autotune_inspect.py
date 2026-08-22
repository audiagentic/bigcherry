"""Offline cache and registry inspector (HI15/HI16).

`hip-autotune-inspect` is a host executable that links the ggml-hip backend
and calls the same registry functions and the same replay loader a
production process uses. It is the C++ half of HI16's catalog/registry
agreement tests and the missing tool HI15's review update assigned to
both items: a Python reimplementation of the loader could disagree with
it, which is the one thing the inspector exists to check.

The tool source lives in the overlay (`src/ggml/src/ggml-cuda/`); this
patch only wires the build target into the ggml-hip backend, next to the
library it links.
"""

# Patch modules are loaded with ``tools`` on PYTHONPATH by the CLI.
# pyright: reportMissingImports=false
from bigcherry.patcher import Edit, FilePatch  # noqa: I001

GROUP = "core"
# Not yet "validated" under the HI83 evidence contract (no HI82 campaign
# record exists for this patch). Build/tool evidence IS complete, on Brutus
# production-replay config (REPLAY=ON, ROUTING=OFF) at pin 0adcc3bb:
#   - full CMake compile + link of the hip-autotune-inspect target
#   - --selftest: 10/10 cross-language digest vectors (C++ one-shot and
#     streaming vs Python hashlib.blake2b)
#   - registry self-check: 88 candidates, 0 anomalies, exit 0
#   - real dispatch-27b-v5.cache: 59/59 winner slots loaded via the
#     production loader, all usable on this build, exit 0
# The validated-state flip therefore only awaits an HI82 campaign record.
STATE = "untested"

CMAKE = FilePatch(
    path="ggml/src/ggml-hip/CMakeLists.txt",
    description="build the offline hip-autotune-inspect host tool",
    edits=(
        Edit(
            id="inspect-executable",
            anchor=r"^target_link_libraries\(ggml-hip PRIVATE ggml-base hip::host roc::rocblas roc::hipblas\)$",
            rationale="last link line of the ggml-hip backend; the inspector links that library",
            text=(
                "\n"
                "# bigcherry: offline cache/registry inspector (HI15/HI16).\n"
                "# Host executable that links the same loader and registry\n"
                "# production uses; its judgements are the real code's.\n"
                "if (GGML_HIP_AUTOTUNE OR GGML_HIP_DISPATCH_REPLAY)\n"
                "    # The inspector includes the same device-code header as the\n"
                "    # dispatch-layer .cpp files (replay.cpp et al), so compile it\n"
                "    # with their language (HIP, or CXX under the hipcc wrapper)\n"
                "    # rather than as a plain host TU.\n"
                "    set(_BC_INSPECT_SRC ../ggml-cuda/hip-autotune-inspect.cpp)\n"
                "    add_executable(hip-autotune-inspect ${_BC_INSPECT_SRC})\n"
                "    if (CXX_IS_HIPCC)\n"
                "        set_source_files_properties(${_BC_INSPECT_SRC} PROPERTIES LANGUAGE CXX)\n"
                "    else()\n"
                "        set_source_files_properties(${_BC_INSPECT_SRC} PROPERTIES LANGUAGE HIP)\n"
                "    endif()\n"
                "    # ggml-base carries ggml/include (where ggml.h lives) as a PUBLIC\n"
                "    # include dir; ggml-hip links it PRIVATE, so name it directly.\n"
                "    target_link_libraries(hip-autotune-inspect PRIVATE ggml-hip ggml-base)\n"
                "    # ggml-hip keeps its source include dirs private to the library,\n"
                "    # but the inspector's headers reach into them: ggml-impl.h lives\n"
                "    # in ggml/src and is pulled in through common.cuh.\n"
                "    target_include_directories(hip-autotune-inspect PRIVATE\n"
                "        ${CMAKE_CURRENT_SOURCE_DIR}/..)\n"
                "endif()\n"
            ),
            guard=r"add_executable\(hip-autotune-inspect",
        ),
    ),
)

PATCHES = [CMAKE]
