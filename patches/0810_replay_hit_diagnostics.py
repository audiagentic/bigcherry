"""Optional, compile-time replay hit diagnostics.

Production replay builds do not compile the hit recorder, so the normal lookup
path has no diagnostics branch or synchronization cost. Diagnostic replay
builds enable GGML_HIP_REPLAY_DIAGNOSTICS and write an aggregated JSONL hit log
when GGML_HIP_DISPATCH_HIT_LOG is set.
"""

# Patch modules are loaded with ``tools`` on PYTHONPATH by the CLI.
# pyright: reportMissingImports=false
from bigcherry.patcher import Edit, FilePatch  # noqa: I001


OPTIONS = FilePatch(
    path="ggml/CMakeLists.txt",
    description="add the opt-in replay diagnostics build option",
    edits=(
        Edit(
            id="replay-diagnostics-option",
            anchor=r"^option\(GGML_HIP_DISPATCH_REPLAY.*$",
            rationale="HIP replay option declarations",
            text='\noption(GGML_HIP_REPLAY_DIAGNOSTICS "ggml: compile replay hit diagnostics" OFF)',
            guard=r"^option\(GGML_HIP_REPLAY_DIAGNOSTICS",
        ),
    ),
)

DEFINITIONS = FilePatch(
    path="ggml/src/ggml-hip/CMakeLists.txt",
    description="compile replay hit diagnostics only when requested",
    edits=(
        Edit(
            id="replay-diagnostics-definition",
            anchor=(
                r"^    if \(GGML_HIP_DISPATCH_REPLAY\)\n"
                r"        add_compile_definitions\(GGML_HIP_DISPATCH_REPLAY\)\n"
                r"    endif\(\)$"
            ),
            rationale="HIP replay compile definitions",
            text=(
                "\n    if (GGML_HIP_REPLAY_DIAGNOSTICS)\n"
                "        if (NOT GGML_HIP_DISPATCH_REPLAY)\n"
                '            message(FATAL_ERROR "GGML_HIP_REPLAY_DIAGNOSTICS requires GGML_HIP_DISPATCH_REPLAY=ON.")\n'
                "        endif()\n"
                "        add_compile_definitions(GGML_HIP_REPLAY_DIAGNOSTICS)\n"
                "    endif()"
            ),
            guard=r"GGML_HIP_REPLAY_DIAGNOSTICS\)",
        ),
    ),
)

PATCHES = [OPTIONS, DEFINITIONS]
