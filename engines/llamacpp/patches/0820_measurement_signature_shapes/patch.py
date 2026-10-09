"""Persist canonical signature shapes in tuning measurements."""

GROUP = "core"
STATE = "validated"

from bigcherry.patcher import Edit, FilePatch  # pyright: ignore[reportMissingImports]

TUNER = FilePatch(
    path="ggml/src/ggml-cuda/hip-autotune-tuner.cu",
    description="include canonical signature metadata in measurement results",
    edits=(
        Edit(
            id="result-canonical-json",
            anchor=r"^    ggml_hip_digest hardware_digest  = \{\};$",
            rationale="retain the shape used to produce each tuning result",
            text="    ggml_hip_digest hardware_digest  = {};\n    std::string canonical_json;",
            guard=r"std::string canonical_json;",
        ),
        Edit(
            id="set-result-canonical-json",
            anchor=r"^    result\.hardware_digest  = ggml_hip_hardware_digest\(hw\);$",
            rationale="serialize the canonical signature once on the cold path",
            text="    result.hardware_digest  = ggml_hip_hardware_digest(hw);\n    result.canonical_json   = ggml_hip_signature_json(sig, true);",
            guard=r"result\.canonical_json",
        ),
        Edit(
            id="emit-result-canonical-json",
            anchor=(
                r'^                "\\\"signature\\\":\\\"%s\\\",\\\"hardware\\\":\\\"%s\\\",\\\"winner\\\":\\\"%s\\\",\\\"$'
            ),
            rationale="make tuning JSONL self-describing for SQLite import",
            text=(
                '                "\\"signature\\":\\"%s\\",\\"hardware\\":\\"%s\\","\n'
                '                "\\"canonical\\":%s,\\"winner\\":\\"%s\\","'
            ),
            guard=r'"\\\"canonical\\\":%s',
        ),
        Edit(
            id="format-result-canonical-json",
            anchor=r"^                ggml_hip_digest_hex\(r\.hardware_digest\)\.c_str\(\),$",
            rationale="supply serialized canonical JSON to the result formatter",
            text=(
                "                ggml_hip_digest_hex(r.hardware_digest).c_str(),\n"
                "                r.canonical_json.c_str(),"
            ),
            guard=r"r\.canonical_json\.c_str",
        ),
    ),
)

PATCHES = [TUNER]
