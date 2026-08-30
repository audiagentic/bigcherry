"""HI81: make CMAKE_HIP_FLAGS-gated options actually reach the compiler on
Windows Ninja+Clang HIP builds.

Root cause (confirmed by reading this file, not guessed): on Windows,
CXX_IS_HIPCC is forced TRUE unconditionally (this file's own comment: "CMake
on Windows doesn't support the HIP language yet"), so enable_language(HIP) is
NEVER called on Windows -- the CMake HIP *language* does not exist in this
build at all. The CXX_IS_HIPCC branch then sets every ROCm source file's
LANGUAGE property to CXX, not HIP (the `else()` branch that sets LANGUAGE HIP
only runs on Linux). CMAKE_HIP_FLAGS only feeds CMake's HIP-language compile
rule -- on Windows these files are compiled as plain CXX (with `-x hip
--offload-arch=...` supplied separately via the hip::device interface
target), so anything appended to CMAKE_HIP_FLAGS is a complete no-op on
Windows: the variable is set and correctly cached, but the CXX rule that
actually compiles these files never consults it. `CMakeCache.txt` shows the
flag; `ninja -t commands` never does. Confirmed identically for both
CMAKE_HIP_FLAGS consumers in this file: the pre-existing GGML_HIP_EXPORT_
METRICS option and this project's own GGML_HIP_UNSAFE_MATH option (patch
1002_hip_unsafe_math_opt_in).

Fix: under the CXX_IS_HIPCC branch, append the same flags to the ROCm
sources' real (CXX) COMPILE_FLAGS via set_property(... APPEND_STRING ...),
which composes with rather than clobbers the pre-existing Windows-Debug
workaround immediately above it (that workaround uses a plain, non-append
`PROPERTIES COMPILE_FLAGS "-O2 -g"`, so ordering matters: this edit's
APPEND_STRING must run after it). On Linux, CXX_IS_HIPCC is false (unless
CXX is literally hipcc), so this new block is inert there and the existing
CMAKE_HIP_FLAGS path -- already confirmed working on Linux/Brutus --
continues to be the only one that fires.
"""

GROUP = "core"
# Real source read (this file, both branches) confirms the mechanism; not
# yet hardware-validated on Windows with GGML_HIP_UNSAFE_MATH=ON --
# promote to "validated" once `ninja -t commands` on Windows shows
# -funsafe-math-optimizations in the real compile line.
STATE = "untested"

import re as _re

from bigcherry.patcher import Edit, FilePatch

# The anchor is a regex (patcher.py compiles edit.anchor directly), so the
# literal block below -- which contains regex metacharacters CMake syntax is
# full of ($, (), .) -- must be escaped, not passed raw. CMake's dialect
# keeps string literals as real anchor content (unlike C/C++, where they're
# noise-stripped), but it DOES blank `#`-comments to same-length spaces
# (csource.strip_noise), so the two comment lines can't be anchored on their
# literal text -- same LITERAL-placeholder technique patches/1222 and
# patches/1225 use for C++ comments/string literals.
_ANCHOR_BLOCK = '''    if (WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        COMMENT1
        COMMENT2
        set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES COMPILE_FLAGS "-O2 -g")
    endif()'''
_ANCHOR = (
    _re.escape(_ANCHOR_BLOCK)
    .replace(_re.escape("COMMENT1"), r"[^\n]*")
    .replace(_re.escape("COMMENT2"), r"[^\n]*")
)

_FIX = '''

    # bigcherry (HI81): CMAKE_HIP_FLAGS (set above for GGML_HIP_EXPORT_METRICS
    # / GGML_HIP_UNSAFE_MATH) only feeds CMake's HIP *language* compile rule.
    # enable_language(HIP) is never called on Windows (see CXX_IS_HIPCC
    # above), so these sources compile as plain CXX here and CMAKE_HIP_FLAGS
    # is silently never consulted -- confirmed via `ninja -t commands`
    # showing no trace of either flag despite CMakeCache.txt looking correct.
    # APPEND_STRING so this composes with (not clobbers) the Debug workaround
    # immediately above rather than replacing it.
    if (GGML_HIP_EXPORT_METRICS OR GGML_HIP_UNSAFE_MATH)
        set(_bigcherry_hipcc_cxx_flags "")
        if (GGML_HIP_EXPORT_METRICS)
            string(APPEND _bigcherry_hipcc_cxx_flags " -Rpass-analysis=kernel-resource-usage --save-temps")
        endif()
        if (GGML_HIP_UNSAFE_MATH)
            string(APPEND _bigcherry_hipcc_cxx_flags " -funsafe-math-optimizations")
        endif()
        set_property(SOURCE ${GGML_SOURCES_ROCM} APPEND_STRING PROPERTY COMPILE_FLAGS "${_bigcherry_hipcc_cxx_flags}")
    endif()'''

PATCH = FilePatch(
    path="ggml/src/ggml-hip/CMakeLists.txt",
    description="route CMAKE_HIP_FLAGS-gated options (GGML_HIP_EXPORT_METRICS, "
                "GGML_HIP_UNSAFE_MATH) through the real CXX compile rule on Windows, "
                "where CMake's HIP language is never enabled and CMAKE_HIP_FLAGS is "
                "silently inert (HI81)",
    edits=(
        Edit(
            id="hi81-windows-cxx-hipcc-flags",
            anchor=_ANCHOR,
            mode="insert_after",
            rationale="right after the existing Windows-Debug CXX COMPILE_FLAGS "
                      "workaround, same CXX_IS_HIPCC branch, before "
                      "target_link_libraries(ggml-hip PRIVATE hip::device)",
            text=_FIX,
            guard=r"_bigcherry_hipcc_cxx_flags",
        ),
    ),
)

PATCHES = [PATCH]
