"""HI02 - build options for HIP measured dispatch.

Two files are touched. The options and their validation go in
``ggml/CMakeLists.txt`` because that is where the backend switches live and,
more importantly, because the validation has to fire even when the HIP backend
directory is never processed -- ``GGML_HIP_AUTOTUNE=ON`` with ``GGML_HIP=OFF``
must fail at configure time, not produce a quietly inert build.

The HIP backend's own ``CMakeLists.txt`` then turns the options into compile
definitions and pulls in SQLite for the record/tune builds.

Two rules from the standards are enforced here rather than at runtime, because
both describe builds that should not exist:

* section 4.3 / 6.1 -- ``GGML_CUDA_FORCE_MMQ`` and ``GGML_CUDA_FORCE_CUBLAS``
  hide legal families from measured dispatch. A build combining either with
  dispatch would tune against an incomplete candidate set and never know.
* section 6.2 -- the ``workload-max`` variant set is defined by an inventory
  file. Without one there is nothing to derive the candidate set from.
"""

from bigcherry.patcher import Edit, FilePatch

_OPTIONS = """
option(GGML_HIP_AUTOTUNE                    "ggml: build the HIP dispatch autotuner"          OFF)
option(GGML_HIP_DISPATCH_REPLAY             "ggml: build HIP replay dispatch (no tuner)"      OFF)
set   (GGML_HIP_AUTOTUNE_VARIANT_SET "inventory" CACHE STRING
                                            "ggml: HIP autotune candidate set")
set_property(CACHE GGML_HIP_AUTOTUNE_VARIANT_SET PROPERTY STRINGS
             "inventory;workload-max;full-max;replay-full;replay-slim")
set   (GGML_HIP_AUTOTUNE_SIGNATURE_FILE "" CACHE STRING
                                            "ggml: inventory JSON driving workload-max")
option(GGML_HIP_AUTOTUNE_SQLITE             "ggml: link SQLite for record/tune modes"          ON)
option(GGML_HIP_AUTOTUNE_RECORD             "ggml: build signature record mode"                OFF)

# Accept uppercase spellings from BUILD_PROFILES.md as aliases (B10).
set(__BC_UPPERCASE_SETS "NATIVE" "WORKLOAD_MAX" "FULL_MAX" "REPLAY_FULL" "REPLAY_SLIM")
if (GGML_HIP_AUTOTUNE_VARIANT_SET IN_LIST __BC_UPPERCASE_SETS)
    string(TOLOWER "${GGML_HIP_AUTOTUNE_VARIANT_SET}" _variant_lower)
    set(GGML_HIP_AUTOTUNE_VARIANT_SET "${_variant_lower}")
endif()
"""

_VALIDATION = """
# ---- bigcherry: HIP measured dispatch validation -----------------------------
if (GGML_HIP_AUTOTUNE OR GGML_HIP_DISPATCH_REPLAY)
    if (NOT GGML_HIP)
        message(FATAL_ERROR
            "GGML_HIP_AUTOTUNE/GGML_HIP_DISPATCH_REPLAY require GGML_HIP=ON.")
    endif()

    # Standards 4.3: these flags remove legal families from consideration, so a
    # tuned or replayed build would be measuring a truncated candidate set.
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

    # Standards 9.1: production replay builds contain no tuner and no SQLite.
    if (GGML_HIP_DISPATCH_REPLAY AND GGML_HIP_AUTOTUNE)
        message(FATAL_ERROR
            "GGML_HIP_DISPATCH_REPLAY and GGML_HIP_AUTOTUNE are mutually "
            "exclusive: a replay build must not carry the tuning engine.")
    endif()
endif()
# ---- end bigcherry -----------------------------------------------------------
"""

_HIP_DEFINITIONS = """
# ---- bigcherry: HIP measured dispatch ---------------------------------------
if (GGML_HIP_AUTOTUNE OR GGML_HIP_DISPATCH_REPLAY)
    add_compile_definitions(GGML_HIP_DISPATCH)
    add_compile_definitions(GGML_HIP_AUTOTUNE_VARIANT_SET="${GGML_HIP_AUTOTUNE_VARIANT_SET}")

    if (GGML_HIP_AUTOTUNE)
        add_compile_definitions(GGML_HIP_AUTOTUNE)
    endif()
    if (GGML_HIP_DISPATCH_REPLAY)
        add_compile_definitions(GGML_HIP_DISPATCH_REPLAY)
    endif()

    # The generated registry, manifest hash header and MMVQ instances are
    # written into the tree by `bigcherry generate` before configure.
    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-cuda/hip-autotune-registry.inc")
        message(FATAL_ERROR
            "HIP dispatch is enabled but hip-autotune-registry.inc is missing. "
            "Run `python -m bigcherry generate` before configuring.")
    endif()

    file(GLOB   SRCS "../ggml-cuda/template-instances/mmvq-autotune-instance-*.cu")
    list(APPEND GGML_SOURCES_ROCM ${SRCS})

    # Upstream globs only *.cu from ggml-cuda/. Standards 12.1 names the host
    # parts of this layer .cpp -- signature, record, db, replay, metrics -- and
    # they are host C++ with no device code, so they should stay .cpp rather
    # than being renamed to compile as HIP. They therefore need their own glob.
    file(GLOB   SRCS "../ggml-cuda/hip-autotune-*.cpp")
    list(APPEND GGML_SOURCES_ROCM ${SRCS})

    # Recording is a *separate capability* from tuning, and the distinction
    # matters: the inventory build records signatures with no tuner and no
    # benchmarking, and its output is what drives workload-max generation.
    # Folding record into GGML_HIP_AUTOTUNE would make the documented inventory
    # profile (DISPATCH_REPLAY=ON, AUTOTUNE=OFF, MODE=record) silently record
    # nothing -- and the whole pipeline would then start from an empty
    # inventory without any error.
    if (GGML_HIP_AUTOTUNE OR GGML_HIP_AUTOTUNE_RECORD)
        add_compile_definitions(GGML_HIP_AUTOTUNE_RECORD)
    endif()

    # No SQLite anywhere. Record mode writes JSON Lines and
    # `python -m bigcherry.inventory` builds the database offline with the
    # stdlib sqlite3 module, against the same sql/dispatch-db.sql schema.
    #
    # That is not a shortcut: it removes a build dependency neither machine
    # has, makes a killed tuning run keep everything already flushed, and makes
    # standards 9.1 -- production links no SQLite -- true by construction
    # rather than by remembering to gate it.
endif()
# ---- end bigcherry -----------------------------------------------------------
"""


OPTIONS_PATCH = FilePatch(
    path="ggml/CMakeLists.txt",
    description="HIP autotune build options and their validation",
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
            guard=r"^option\(GGML_HIP_AUTOTUNE\b",
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
    description="HIP backend compile definitions, generated sources, SQLite",
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
        # No link edit. The dispatch layer has no external dependencies -- the
        # only one it ever had was SQLite, and record mode writes JSON Lines
        # instead. Leaving an edit that inserts nothing would be a permanent
        # false positive in every patch report.
    ),
)

PATCHES = [OPTIONS_PATCH, HIP_BACKEND_PATCH]
