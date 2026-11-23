#!/usr/bin/env python3
"""Extract a range-capable lab DiT evaluator from the pinned native implementation.

No production source is changed; source drift fails at generation.
"""
from pathlib import Path
import sys

source, dest = map(Path, sys.argv[1:])
text = source.read_text()
start = text.index("static void dit_eval(")
left = text.index("{", start)
depth = 0
end = None
for i in range(left, len(text)):
    if text[i] == "{":
        depth += 1
    elif text[i] == "}":
        depth -= 1
        if depth == 0:
            end = i + 1
            break
if end is None:
    raise RuntimeError("native DiT evaluator has no closing brace")
function = text[start:end]
for expected in ("for (int64_t l = 0; l < L; ++l)","    RAD_ISSUE_N(c, m.op_lv,","    const RadOperand tv =","    RAD_ISSUE_N(c, m.op_fnv,","    m.pout_a.step("):
    if function.count(expected) != 1:
        raise RuntimeError("native DiT evaluator structural drift: " + expected)
function = function.replace("static void dit_eval(", "static void pass_dit_eval_range(", 1)
function = function.replace("bool iso) {", "bool iso, int64_t begin, int64_t end) {", 1)
if "int64_t begin, int64_t end" not in function:
    raise RuntimeError("range signature cannot be added")
function = function.replace("    RAD_ISSUE_N(c, m.op_lv,", "    if (begin == 0) {\n    RAD_ISSUE_N(c, m.op_lv,", 1)
function = function.replace("    const RadOperand tv =", "    }\n    const RadOperand tv =", 1)
function = function.replace("for (int64_t l = 0; l < L; ++l)", "for (int64_t l = begin; l < end; ++l)", 1)
function = function.replace("    RAD_ISSUE_N(c, m.op_fnv,", "    if (end == L) {\n    RAD_ISSUE_N(c, m.op_fnv,", 1)
where = function.rfind("\n}")
function = function[:where] + "\n    }" + function[where:]
dest.write_text("// Generated from native dit_eval: opt-in lab only.\nnamespace ltx2 {\n" + function + "\n} // namespace ltx2\n")
