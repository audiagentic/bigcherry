"""HI02 - build options for HIP measured dispatch (campaign tune/record half).

PA27 split ``patches/0100_cmake_options`` by runtime ownership (dev-gpt-agent
design, req_42717536998244ea). This package owns everything a tune/record
campaign build needs that a pure replay/serving build must not carry:
tuner engine, record mode, workspace metrics, hot-path diagnostics, and
sig-to-sig routing transformations. It is NOT declared as depending on
``0100_cmake_options`` via ``requires`` -- PA28's serving-core/campaign-support
composition needs to add this package to sets that also carry ``0100``
independently. Instead, every campaign-only branch is fail-closed at CMake
configure time against a non-cache sentinel
(``_BC_HIP_SERVING_BUILD_PLUMBING``) that ``0100`` sets in its own top-level
``ggml/CMakeLists.txt`` block (never in the HIP backend's own
``CMakeLists.txt`` -- ``add_subdirectory(src)`` runs after the top-level
file's own blocks, so a sentinel set only in the backend file would not
exist yet when this package's top-level validation checks it). Reciprocally,
this package sets its own ``_BC_HIP_CAMPAIGN_BUILD_PLUMBING`` sentinel that
``0100`` reads to reject ``GGML_HIP_AUTOTUNE`` set without this package
present: activating any campaign-only option without the serving package
applied, or activating a serving-only option without the campaign package
present while trying to use campaign-only flags, is a configure-time error,
not a silently broken build.

Two rules from the standards are enforced here rather than at runtime,
because both describe builds that should not exist:

* section 4.3 / 6.1 -- ``GGML_CUDA_FORCE_MMQ`` and ``GGML_CUDA_FORCE_CUBLAS``
  hide legal families from measured dispatch. A build combining either with
  tuning would tune against an incomplete candidate set and never know.
* section 6.2 -- the ``workload-max`` variant set is defined by an inventory
  file. Without one there is nothing to derive the candidate set from.

Record mode writes JSON Lines; ``python -m bigcherry.inventory`` builds the
database offline with the stdlib ``sqlite3`` module against the same
``sql/dispatch-db.sql`` schema. There is no SQLite link edit anywhere in this
package (PA27 removed the declaration-only, no-consumer
``GGML_HIP_AUTOTUNE_SQLITE`` option that 0100 used to carry).
"""

GROUP = "core"
STATE = "validated"

from bigcherry.patcher import Edit, FilePatch

_OPTIONS = """
option(GGML_HIP_AUTOTUNE                    "ggml: build the HIP dispatch autotuner"          OFF)
# bigcherry: hot-path diagnostics. OFF is the PRODUCTION shape -- the dispatch
# counters, the native-select sample timing and the per-launch coverage
# counting are all compiled out, not merely runtime-disabled. Coverage counting
# in particular was unconditional (two atomic RMWs per dispatch, ~382,000 per
# bench run), with the env var controlling only whether a report was WRITTEN.
# A build used for a final performance number must not carry any of it.
option(GGML_HIP_DISPATCH_DIAGNOSTICS        "ggml: HIP dispatch hot-path diagnostics"         OFF)
option(GGML_HIP_AUTOTUNE_RECORD             "ggml: build signature record mode"                OFF)
option(GGML_HIP_WORKSPACE_METRICS            "ggml: collect tune-only pool workspace metrics"   OFF)
option(GGML_HIP_ROUTING_TRANSFORM           "ggml: tune sig-to-sig routing transformations"    OFF)

# bigcherry: non-cache marker read by the serving/shared package
# (0100_cmake_options) to fail closed if GGML_HIP_AUTOTUNE is activated
# without this campaign package present. Set at top level, not in the HIP
# backend's own CMakeLists.txt -- see 0100_cmake_options's matching
# sentinel comment for why (dev-gpt-agent review, req_4c330960a8db450f,
# BLOCKER 1/2).
set(_BC_HIP_CAMPAIGN_BUILD_PLUMBING ON)
"""

_VALIDATION = """
# ---- bigcherry: HIP campaign tune/record build validation --------------------
# dev-gpt-agent review (req_4c330960a8db450f, BLOCKER 2): GGML_HIP_DISPATCH_REPLAY
# is included here (a foreign, 0100-owned option) so that a raw, undeclared
# -DGGML_HIP_DISPATCH_REPLAY=ON with only this package applied is also
# rejected -- it would otherwise enter this package's own dispatch/replay
# definitions block (`if (GGML_HIP_AUTOTUNE OR GGML_HIP_DISPATCH_REPLAY)`)
# without 0100's registry/generated-dir/production-source plumbing ever
# having run.
if (GGML_HIP_DISPATCH_REPLAY OR GGML_HIP_AUTOTUNE OR GGML_HIP_AUTOTUNE_RECORD
        OR GGML_HIP_DISPATCH_DIAGNOSTICS OR GGML_HIP_WORKSPACE_METRICS
        OR GGML_HIP_ROUTING_TRANSFORM)
    if (NOT _BC_HIP_SERVING_BUILD_PLUMBING)
        message(FATAL_ERROR
            "HIP campaign build plumbing (0110_campaign_tune_record_build) "
            "requires the serving dispatch/replay package "
            "(0100_cmake_options) to be applied.")
    endif()
endif()

if (GGML_HIP_AUTOTUNE)
    if (NOT GGML_HIP)
        message(FATAL_ERROR
            "GGML_HIP_AUTOTUNE requires GGML_HIP=ON.")
    endif()

    # Standards 4.3: these flags remove legal families from consideration, so a
    # tuned build would be measuring a truncated candidate set.
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

# Standards 9.1: production replay builds contain no tuner.
if (GGML_HIP_DISPATCH_REPLAY AND GGML_HIP_AUTOTUNE)
    message(FATAL_ERROR
        "GGML_HIP_DISPATCH_REPLAY and GGML_HIP_AUTOTUNE are mutually "
        "exclusive: a replay build must not carry the tuning engine.")
endif()

# HI29-HI31: the transform registry and launch helper are present, but the
# transform identity is not yet part of the record/replay ABI and the normal
# dispatcher does not yet select transformed bindings.  Do not let a partial
# build advertise the feature: it could tune a transformed path without being
# able to persist or replay the same decision.  Requiring the tuning dispatch
# and the record capability is the safe offline boundary until that ABI is
# wired end-to-end.
if (GGML_HIP_ROUTING_TRANSFORM AND
        (NOT GGML_HIP_AUTOTUNE OR NOT GGML_HIP_AUTOTUNE_RECORD))
    message(FATAL_ERROR
        "GGML_HIP_ROUTING_TRANSFORM requires both GGML_HIP_AUTOTUNE and "
        "GGML_HIP_AUTOTUNE_RECORD until transform recording and dispatch "
        "integration is complete.")
endif()

if (GGML_HIP_WORKSPACE_METRICS AND NOT GGML_HIP_AUTOTUNE)
    message(FATAL_ERROR
        "GGML_HIP_WORKSPACE_METRICS requires GGML_HIP_AUTOTUNE=ON.")
endif()
# ---- end bigcherry -----------------------------------------------------------
"""

_HIP_DEFINITIONS = """
# ---- bigcherry: HIP campaign tune/record build ------------------------------
if (GGML_HIP_AUTOTUNE OR GGML_HIP_DISPATCH_REPLAY)
    if (GGML_HIP_AUTOTUNE)
        add_compile_definitions(GGML_HIP_AUTOTUNE)
    endif()
    # Tuning and recording builds need the counters to do their job, so they
    # get diagnostics implicitly; a pure replay build does not and must be
    # able to be built clean for benchmarking.
    if (GGML_HIP_DISPATCH_DIAGNOSTICS OR GGML_HIP_AUTOTUNE OR GGML_HIP_AUTOTUNE_RECORD)
        add_compile_definitions(GGML_HIP_DISPATCH_DIAGNOSTICS)
    endif()
    # HI27. Global rather than per-source: the transform machinery is declared
    # in hip-autotune-types.h, which the tuner, the dispatcher and the replay
    # loader all include. Defining it for hip-autotune-transform.cu alone would
    # give that one file a different view of a struct the others also lay out.
    if (GGML_HIP_ROUTING_TRANSFORM)
        add_compile_definitions(GGML_HIP_ROUTING_TRANSFORM)
    endif()
    if (GGML_HIP_WORKSPACE_METRICS)
        if (NOT GGML_HIP_AUTOTUNE)
            message(FATAL_ERROR "GGML_HIP_WORKSPACE_METRICS requires GGML_HIP_AUTOTUNE=ON.")
        endif()
        add_compile_definitions(GGML_HIP_WORKSPACE_METRICS)
    endif()

    # Recording is a *separate capability* from tuning, and the distinction
    # matters: the inventory build records signatures with no tuner and no
    # benchmarking, and its output is what drives workload-max generation.
    # Folding record into GGML_HIP_AUTOTUNE would make the documented inventory
    # profile (DISPATCH_REPLAY=ON, AUTOTUNE=OFF, MODE=record) silently record
    # nothing -- and the whole pipeline would then start from an empty
    # inventory without any error.
    set(_BC_CAMPAIGN_SOURCES "")
    if (GGML_HIP_DISPATCH_DIAGNOSTICS OR GGML_HIP_AUTOTUNE OR GGML_HIP_AUTOTUNE_RECORD)
        list(APPEND _BC_CAMPAIGN_SOURCES
            "../ggml-cuda/hip-autotune-coverage.cpp")
    endif()
    if (GGML_HIP_AUTOTUNE_RECORD)
        list(APPEND _BC_CAMPAIGN_SOURCES
            "../ggml-cuda/hip-autotune-record.cpp")
    endif()
    if (GGML_HIP_AUTOTUNE)
        list(APPEND _BC_CAMPAIGN_SOURCES
            "../ggml-cuda/hip-autotune-tuner.cu"
            "../ggml-cuda/hip-autotune-io.cpp"
            "../ggml-cuda/hip-autotune-journal.cpp"
            "../ggml-cuda/hip-autotune-smi.cpp")
    endif()
    list(APPEND GGML_SOURCES_ROCM ${_BC_CAMPAIGN_SOURCES})

    # Kept separate from the source-list block above (not folded into the
    # GGML_HIP_AUTOTUNE_RECORD if-block that appends record.cpp) so the
    # patch-local validation check can extract a pure, self-contained
    # source-selection slice with no add_compile_definitions() call inside
    # it -- that command is unavailable in bare `cmake -P` script mode,
    # which the check uses to prove source selection without a real project.
    if (GGML_HIP_AUTOTUNE_RECORD)
        add_compile_definitions(GGML_HIP_AUTOTUNE_RECORD)
    endif()
endif()
# ---- end bigcherry -----------------------------------------------------------
"""


OPTIONS_PATCH = FilePatch(
    path="ggml/CMakeLists.txt",
    description="HIP campaign tune/record build options and their validation",
    edits=(
        Edit(
            id="campaign-tune-record-options",
            # A different, non-overlapping option line than the one
            # 0100_cmake_options anchors to, so both packages apply
            # independently to pristine upstream regardless of application
            # order.
            anchor=r"^option\(GGML_HIP_MMQ_MFMA.*$",
            rationale="an upstream HIP option line distinct from the one "
            "0100_cmake_options anchors to, so both packages apply "
            "independently to pristine upstream",
            text=_OPTIONS,
            guard=r"^option\(GGML_HIP_AUTOTUNE\b",
        ),
        Edit(
            id="campaign-tune-record-validation",
            # GGML_WEBGPU_DEBUG immediately follows GGML_WEBGPU, which
            # 0100_cmake_options anchors its own validation block to -- these
            # are two distinct pristine lines, so neither package's anchor
            # depends on the other having applied first.
            anchor=r"^option\(GGML_WEBGPU_DEBUG\b.*$",
            rationale="a distinct backend-option-list line from the one "
            "0100_cmake_options anchors its validation to",
            text=_VALIDATION,
            guard=r"bigcherry: HIP campaign tune/record build validation",
        ),
    ),
)

HIP_BACKEND_PATCH = FilePatch(
    path="ggml/src/ggml-hip/CMakeLists.txt",
    description="HIP backend campaign-only compile definitions and tune/record sources",
    edits=(
        Edit(
            id="campaign-tune-record-definitions",
            # ggml_cuda_fattn_vec_instances(...) is a separate, non-overlapping
            # source-glob block from the mmf*.cu one 0100_cmake_options
            # anchors to -- both packages apply independently to pristine
            # upstream and both execute before ggml_add_backend_library(), so
            # both may safely append to GGML_SOURCES_ROCM.
            anchor=r"^ggml_cuda_fattn_vec_instances\(\$\{CMAKE_CURRENT_SOURCE_DIR\}/\.\./ggml-cuda SRCS\)\n"
            r"list\(APPEND GGML_SOURCES_ROCM \$\{SRCS\}\)$",
            rationale="the end of the fattn-vec-instances source-glob block, "
            "distinct from the mmf*.cu block 0100_cmake_options anchors to",
            text=_HIP_DEFINITIONS,
            guard=r"bigcherry: HIP campaign tune/record build",
        ),
    ),
)

PATCHES = [OPTIONS_PATCH, HIP_BACKEND_PATCH]
