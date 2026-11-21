"""Diagnostic LTX C01: native IQ8R fused-bank/column weight declaration only.

Targets the frozen Radiance 2.2 ABI44 candidate, never production v1.17.
TP1 callers pass no C01Slice and keep their exact original code path.
No real TP3 launch is enabled: the upstream world_size>1 refusal remains.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

from bigcherry.patcher import Edit, FilePatch

_SRC = Path(__file__).resolve().parent / "src"
_SHA256 = {
    "bc_ltx_tp3_c01.h": "c13114e5726441235e0a4a92f8f61c57c1b3cb351dade5290bbfb15530c7484a",
    "preflight.inc": "edfbd9753a8d9fc489cc43526c595a238fef6856329e55bababa35de3ae46257",
    "bias.inc": "ca8701172fb02baf0211f4bbdc55d25bbb219a4c2835f19e0676953bc6f86e3e",
    "linear.inc": "5e6af267c582872681306a492769879fc90609cf2b21284f903555239a06eacb",
}


def _source(filename: str) -> str:
    data = (_SRC / filename).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != _SHA256[filename]:
        raise RuntimeError(f"LT3 C01 {filename} changed without digest update")
    return data.decode("utf8")


def _edit(label: str, before: str, after: str, marker: str) -> Edit:
    return Edit(
        id=label,
        anchor=re.escape(before),
        mode="replace",
        text=after,
        guard=re.escape(marker),
        expect_matches=1,
        max_span_lines=max(4, before.count("\n") + 2),
        rationale="LT3 native IQ8R builder-only staging: fail closed if frozen ABI44 source changes",
    )


PATCHES = (
    FilePatch(
        path="arch/ltx2/bc_ltx_tp3_c01.h",
        create=True,
        language="none",
        description="Native C01 LTX2 TP3 IQ8R shard admission geometry",
        edits=(Edit(
            id="ltx-c01-shard-header",
            anchor=r"\A",
            mode="insert_after",
            text=_source("bc_ltx_tp3_c01.h"),
            guard=r"BC_LTX3_C01_HEADER",
            expect_matches=1,
            max_span_lines=1,
            rationale="Create a new LTX-only C01 header; never replace an upstream file",
        ),),
    ),
    FilePatch(
        path="arch/ltx2/ltx2.cpp",
        language="none",
        description="Opt-in LTX Lin::declare sharding, all TP1 call sites unchanged",
        edits=(
            _edit(
                "ltx-c01-deferred-bias-state",
                "    int64_t    N = 0, K = 0;\n",
                "    int64_t    N = 0, K = 0;\n"
                "    bool deferred_bias = false; /* BC_LTX3_C01_DEFERRED_STATE */\n",
                "BC_LTX3_C01_DEFERRED_STATE",
            ),
            _edit(
                "ltx-c01-include",
                "#include <arch/rad_fp8.h>\n",
                "#include <arch/rad_fp8.h>\n"
                "#include \"bc_ltx_tp3_c01.h\" /* BC_LTX3_C01_INCLUDE */\n",
                "BC_LTX3_C01_INCLUDE",
            ),
            _edit(
                "ltx-c01-optional-argument",
                "int group, const Act& in, rad_buf y, int64_t max_rows, bool small = false) {",
                "int group, const Act& in, rad_buf y, int64_t max_rows, bool small = false,\n"
                "                const C01Slice* slice = nullptr) { /* BC_LTX3_C01_ARGUMENT */",
                "BC_LTX3_C01_ARGUMENT",
            ),
            _edit(
                "ltx-c01-preflight",
                '        K = n_in;\n        const char* wn = nm.f("%s.weight", base);\n',
                "        K = n_in;\n"
                + _source("preflight.inc")
                + '        const char* wn = nm.f("%s.weight", base);\n',
                "BC_LTX3_C01_PREFLIGHT",
            ),
            _edit(
                "ltx-c01-bias",
                "            bias = decl_w(b, bn, RAD_BF16, { N }, access, RAD_SHARD_NONE, grp_layer(group));\n"
                "            if (!bias) return RAD_E_INVAL;\n",
                _source("bias.inc"),
                "BC_LTX3_C01_BIAS",
            ),
            _edit(
                "ltx-c01-weight",
                "        RAD_ARCH_TRY(l.declare(b, nm, g, base, N, K, RAD_SHARD_NONE, grp_layer(group), wsrc, in.a, y,\n"
                "                               max_rows, parts));\n",
                _source("linear.inc"),
                "BC_LTX3_C01_LINEAR",
            ),
        ),
    ),
)
