"""HI02 - build options for HIP measured dispatch (serving/shared half).

PA27 split this patch by runtime ownership (dev-gpt-agent design,
req_42717536998244ea): this package now owns only what a replay/serving
build needs -- dispatch/replay enablement, generated registry inclusion,
production source list, and validation that makes an invalid replay build
fail closed. The campaign-only half (tune/record options, tuner source
inclusion, workload/record routing, campaign-only diagnostics) moved to
``patches/0110_campaign_tune_record_build``.

Two files are touched. The options and their validation go in
``ggml/CMakeLists.txt`` because that is where the backend switches live and,
more importantly, because the validation has to fire even when the HIP backend
directory is never processed -- ``GGML_HIP_DISPATCH_REPLAY=ON`` with
``GGML_HIP=OFF`` must fail at configure time, not produce a quietly inert
build.

The HIP backend's own ``CMakeLists.txt`` then turns the options into compile
definitions and pulls in the production dispatch/replay sources.

One rule from the standards is enforced here rather than at runtime, because
it describes builds that should not exist:

* section 4.3 / 6.1 -- ``GGML_CUDA_FORCE_MMQ`` and ``GGML_CUDA_FORCE_CUBLAS``
  hide legal families from measured dispatch. A replayed build combining
  either would replay against an incomplete candidate set and never know.

This package sets a non-cache sentinel, ``_BC_HIP_SERVING_BUILD_PLUMBING``,
in ``ggml/CMakeLists.txt`` (top level, NOT the HIP backend's own
``CMakeLists.txt`` -- ``add_subdirectory(src)`` runs after this file's own
option/validation blocks, so a sentinel set only in the backend file would
not exist yet when a top-level validation check reads it) that
``0110_campaign_tune_record_build`` reads to fail closed if its own options
are activated without this serving package applied. Reciprocally, this
package's own validation reads ``0110``'s
``_BC_HIP_CAMPAIGN_BUILD_PLUMBING`` sentinel to reject ``GGML_HIP_AUTOTUNE``
set without the campaign package present.
"""

GROUP = "core"
STATE = "validated"

from bigcherry.patcher import Edit, FilePatch

_OPTIONS = """
option(GGML_HIP_DISPATCH_REPLAY             "ggml: build HIP replay dispatch (no tuner)"      OFF)
set   (GGML_HIP_AUTOTUNE_VARIANT_SET "inventory" CACHE STRING
                                            "ggml: HIP autotune candidate set")
set_property(CACHE GGML_HIP_AUTOTUNE_VARIANT_SET PROPERTY STRINGS
             "inventory;workload-max;full-max;replay-full;replay-slim")
set   (GGML_HIP_AUTOTUNE_SIGNATURE_FILE "" CACHE STRING
                                            "ggml: inventory JSON driving workload-max")
set   (GGML_HIP_AUTOTUNE_GENERATED_DIR "" CACHE PATH
                                            "bigcherry: build-local generated compile inputs")

# Accept uppercase spellings from BUILD_PROFILES.md as aliases (B10).
set(__BC_UPPERCASE_SETS "NATIVE" "WORKLOAD_MAX" "FULL_MAX" "REPLAY_FULL" "REPLAY_SLIM")
if (GGML_HIP_AUTOTUNE_VARIANT_SET IN_LIST __BC_UPPERCASE_SETS)
    string(TOLOWER "${GGML_HIP_AUTOTUNE_VARIANT_SET}" _variant_lower)
    set(GGML_HIP_AUTOTUNE_VARIANT_SET "${_variant_lower}")
endif()

# bigcherry: non-cache marker read by the campaign tune/record build package
# (0110_campaign_tune_record_build) to fail closed if its own options are
# activated without this serving package present. Set HERE at top level, not
# in ggml/src/ggml-hip/CMakeLists.txt -- add_subdirectory(src) (which
# eventually reaches the HIP backend file) runs AFTER this top-level
# CMakeLists.txt's own option/validation blocks, so a sentinel set only in
# the backend file would not yet exist when 0110's top-level validation
# checks it (dev-gpt-agent review, req_4c330960a8db450f, BLOCKER 1).
set(_BC_HIP_SERVING_BUILD_PLUMBING ON)
"""

_VALIDATION = """
# ---- bigcherry: HIP measured dispatch validation (serving/shared) -----------
if (GGML_HIP_DISPATCH_REPLAY)
    if (NOT GGML_HIP)
        message(FATAL_ERROR
            "GGML_HIP_DISPATCH_REPLAY requires GGML_HIP=ON.")
    endif()

    # Standards 4.3: these flags remove legal families from consideration, so a
    # replayed build would be measuring a truncated candidate set.
    if (GGML_CUDA_FORCE_MMQ)
        message(FATAL_ERROR
            "GGML_CUDA_FORCE_MMQ is incompatible with HIP measured dispatch: "
            "it hides non-MMQ families from the candidate set.")
    endif()
    if (GGML_CUDA_FORCE_CUBLAS)
        message(FATAL_ERROR
            "GGML_CUDA_FORCE_CUBLAS is incompatible with HIP measured dispatch: "
            "it hides the MMQ/MMVQ/MMVF/MMF families from the candidate set.")
    endif()

    if (GGML_HIP_AUTOTUNE_VARIANT_SET STREQUAL "workload-max"
            AND GGML_HIP_AUTOTUNE_SIGNATURE_FILE STREQUAL "")
        message(FATAL_ERROR
            "GGML_HIP_AUTOTUNE_VARIANT_SET=workload-max requires "
            "GGML_HIP_AUTOTUNE_SIGNATURE_FILE to point at an inventory JSON.")
    endif()
endif()

# Reciprocal fail-closed check (dev-gpt-agent review, req_4c330960a8db450f,
# BLOCKER 2): this package does not declare GGML_HIP_AUTOTUNE, but CMake
# still accepts a raw, undeclared -DGGML_HIP_AUTOTUNE=ON with only this
# package applied -- that would enter this package's own dispatch/replay
# definitions block (`if (GGML_HIP_AUTOTUNE OR GGML_HIP_DISPATCH_REPLAY)`)
# without any of 0110_campaign_tune_record_build's tuner definitions or
# sources ever being compiled in. Reject it explicitly.
if (GGML_HIP_AUTOTUNE AND NOT _BC_HIP_CAMPAIGN_BUILD_PLUMBING)
    message(FATAL_ERROR
        "GGML_HIP_AUTOTUNE requires the HIP campaign tune/record build "
        "package (0110_campaign_tune_record_build) to be applied.")
endif()
# ---- end bigcherry -----------------------------------------------------------
"""

_HIP_DEFINITIONS = """
# ---- bigcherry: HIP measured dispatch (serving/shared) ----------------------
if (GGML_HIP_AUTOTUNE OR GGML_HIP_DISPATCH_REPLAY)
    # Partition BigCherry CUDA translation units out of upstream's broad glob;
    # they are appended explicitly below so each source enters the target once.
    list(FILTER GGML_SOURCES_ROCM EXCLUDE REGEX "hip-autotune-(dispatch|tuner|transform)\\.cu$")
    add_compile_definitions(GGML_HIP_DISPATCH)
    add_compile_definitions(GGML_HIP_AUTOTUNE_VARIANT_SET="${GGML_HIP_AUTOTUNE_VARIANT_SET}")

    if (GGML_HIP_DISPATCH_REPLAY)
        add_compile_definitions(GGML_HIP_DISPATCH_REPLAY)
    endif()

    # Campaign builds use an out-of-tree generated directory.  The empty
    # value remains a legacy compatibility path for the interactive command;
    # campaign/build services always pass the explicit directory.
    set(_BC_GENERATED_DIR "${GGML_HIP_AUTOTUNE_GENERATED_DIR}")
    if (_BC_GENERATED_DIR STREQUAL "")
        set(_BC_GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-cuda")
    endif()
    if (NOT EXISTS "${_BC_GENERATED_DIR}/hip-autotune-registry.inc")
        message(FATAL_ERROR
            "HIP dispatch is enabled but hip-autotune-registry.inc is missing. "
            "Run `python -m bigcherry generate` before configuring.")
    endif()
    include_directories("${_BC_GENERATED_DIR}")

    file(GLOB   SRCS "${_BC_GENERATED_DIR}/template-instances/mmvq-autotune-instance-*.cu")
    list(APPEND GGML_SOURCES_ROCM ${SRCS})

    # Upstream globs only *.cu from ggml-cuda/. Standards 12.1 names the host
    # parts of this layer .cpp -- signature, record, db, replay, metrics -- and
    # they are host C++ with no device code, so they should stay .cpp rather
    # than being renamed to compile as HIP. They therefore need their own glob.
    # Keep the production link graph explicit. A broad hip-autotune-* glob
    # pulled tuning, journal, SMI and record machinery into replay binaries.
    set(_BC_DISPATCH_SOURCES
        "../ggml-cuda/hip-autotune-dispatch.cu"
        "../ggml-cuda/hip-autotune-transform.cu"
        "../ggml-cuda/hip-autotune-reduce-telemetry.cpp"
        "../ggml-cuda/hip-autotune-signature.cpp"
        "../ggml-cuda/hip-autotune-blake2b.cpp")
    if (GGML_HIP_DISPATCH_REPLAY)
        list(APPEND _BC_DISPATCH_SOURCES
            "../ggml-cuda/hip-autotune-replay.cpp")
    endif()
    list(APPEND GGML_SOURCES_ROCM ${_BC_DISPATCH_SOURCES})
endif()
# ---- end bigcherry -----------------------------------------------------------
"""


OPTIONS_PATCH = FilePatch(
    path="ggml/CMakeLists.txt",
    description="HIP replay/serving build options and their validation",
    edits=(
        Edit(
            id="hip-autotune-options",
            # The last GGML_HIP_* option line. Anchoring to the option *block*
            # rather than a line number means upstream can add HIP options
            # above or below without breaking us; it only breaks if this
            # particular option is renamed or removed.
            anchor=r"^option\(GGML_HIP_EXPORT_METRICS.*$",
            rationale="the end of the GGML_HIP_* option block",
            text=_OPTIONS,
            guard=r"^option\(GGML_HIP_DISPATCH_REPLAY\b",
        ),
        Edit(
            id="hip-autotune-validation",
            # Placed after the whole backend option list so every option it
            # reads has already been declared.
            anchor=r"^option\(GGML_WEBGPU\b.*$",
            rationale="after the backend option list, where all referenced "
            "options are declared",
            text=_VALIDATION,
            guard=r"bigcherry: HIP measured dispatch validation",
        ),
    ),
)

HIP_BACKEND_PATCH = FilePatch(
    path="ggml/src/ggml-hip/CMakeLists.txt",
    description="HIP backend compile definitions and production dispatch/replay sources",
    edits=(
        Edit(
            id="hip-autotune-definitions",
            # Must land after GGML_SOURCES_ROCM exists (it appends to it) and
            # before the target is defined.
            anchor=r"^file\(GLOB\s+SRCS\s+\"\.\./ggml-cuda/template-instances/mmf\*\.cu\"\)\n"
            r"list\(APPEND GGML_SOURCES_ROCM \$\{SRCS\}\)$",
            rationale="the end of the GGML_SOURCES_ROCM glob block",
            text=_HIP_DEFINITIONS,
            guard=r"bigcherry: HIP measured dispatch",
        ),
    ),
)

PATCHES = [OPTIONS_PATCH, HIP_BACKEND_PATCH]
