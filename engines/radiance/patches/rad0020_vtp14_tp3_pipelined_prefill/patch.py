"""rad0020 (VTP14): the fp8 Qwen plugin's pipelined prefill as configuration (--pipelined-prefill, --pipelined-prefill-min-rows).

qwen35_fp8 already pipelines a wide prefill step (step_pipelined): the step's rows run as two pieces, and each block's
all-reduce for one piece runs on the second lane beside the other piece's work, bit for bit the serial step. Upstream
fixes both choices in the plugin: it pipelines at two ranks only, from 1024 rows (kPipeMinRows). Every all-reduce in
the pipelined step is the collective the serial step declares, at `overlap` 1, and selection picks its kernel by
world, size and peer access as for the serial one; nothing in the step depends on the rank count.

This patch makes both choices configuration, carried in the build context:

- `--pipelined-prefill auto|on|off`. `auto` (the default) is upstream's two ranks, and also three ranks under
  --tp-ownership (Geom::tp_owned), so the TP3 ownership prefills like the two-rank split it is compared with. `on`
  pipelines at any world of two or more ranks; `off` never pipelines.
- `--pipelined-prefill-min-rows N`: the narrowest step that pipelines; 0 (the default) is the plugin's own 1024.

A trunk whose norms write rotated int8 codes (a_h.q8_fed) still prefills serially whatever the mode, as upstream.
Without either option the plugin behaves as upstream plus the three-rank ownership case.
Source-pinned to radiance v1.17.0 (19d4e8fe), applied after rad0019.
"""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch


def _edit(id: str, before: str, after: str, *, guard: str, max_span: int | None = None) -> Edit:
    return Edit(
        id=id,
        anchor=re.escape(before),
        mode="replace",
        text=after,
        guard=re.escape(guard),
        expect_matches=1,
        max_span_lines=max_span or max(2, before.count("\n") + 1),
        rationale="VTP14: pipelined prefill as configuration",
    )


ABI = FilePatch(path="abi/rad_builder.h", language="none", description="VTP14 pipelined prefill build context", edits=(
    _edit(
        "pipe-ctx-fields",
        "    const char* tp_ownership;\n"
        "} RadBuildCtx;",
        "    const char* tp_ownership;\n"
        "\n"
        "    /* ---- BC_VTP14_PIPE_CTX (BigCherry rad0020) */\n"
        "    /* --pipelined-prefill: BC_PIPE_AUTO (0, the plugin's own rule), BC_PIPE_ON or BC_PIPE_OFF;\n"
        "     * --pipelined-prefill-min-rows: the narrowest pipelined step, 0 for the plugin's own.\n"
        "     * Trailing and zero-defaulted. */\n"
        "    int32_t pipelined_prefill;\n"
        "    int64_t pipelined_prefill_min_rows;\n"
        "} RadBuildCtx;\n"
        "enum { BC_PIPE_AUTO = 0, BC_PIPE_ON = 1, BC_PIPE_OFF = 2 };",
        guard="BC_VTP14_PIPE_CTX",
    ),
))

RAD_CORE = FilePatch(path="core/rad_core.h", language="none", description="VTP14 pipelined prefill config", edits=(
    _edit(
        "pipe-config",
        "    std::string tp_ownership;\n",
        "    std::string tp_ownership;\n"
        "    /* BC_VTP14_PIPE_CONFIG: --pipelined-prefill auto|on|off (0|1|2) and its narrowest step, 0 the\n"
        "     * plugin's own. */\n"
        "    int     pipelined_prefill = 0;\n"
        "    int64_t pipelined_prefill_min_rows = 0;\n",
        guard="BC_VTP14_PIPE_CONFIG",
    ),
))

CONFIG = FilePatch(path="core/config.cpp", language="none", description="VTP14 --pipelined-prefill", edits=(
    _edit(
        "pipe-help",
        "    { \"--tp-ownership\",       \"SPEC\",",
        "    { \"--pipelined-prefill\",  \"MODE\",   \"BigCherry VTP14: pipeline a wide one-sequence prefill step\\n\"\n"
        "                                        \"(all-reduces on the second lane): auto (default: the\\n\"\n"
        "                                        \"plugin's own rule), on (any world of 2 or more), off\" }, /* BC_VTP14_PIPE_HELP */\n"
        "    { \"--pipelined-prefill-min-rows\", \"N\", \"BigCherry VTP14: the narrowest pipelined prefill step,\\n\"\n"
        "                                        \"at least 128 rows; 0 (default) is the plugin's own\" },\n"
        "    { \"--tp-ownership\",       \"SPEC\",",
        guard="BC_VTP14_PIPE_HELP",
        max_span=1,
    ),
    _edit(
        "pipe-parse",
        "        else if (S(\"--tp-ownership\")) { /* BC_VTP14_N2_PARSE */\n",
        "        else if (S(\"--pipelined-prefill\")) { /* BC_VTP14_PIPE_PARSE */\n"
        "            if (!(v = need(i))) return RAD_E_INVAL;\n"
        "            if      (!std::strcmp(v, \"auto\")) c->pipelined_prefill = 0;\n"
        "            else if (!std::strcmp(v, \"on\"))   c->pipelined_prefill = 1;\n"
        "            else if (!std::strcmp(v, \"off\"))  c->pipelined_prefill = 2;\n"
        "            else { RAD_ERR(\"--pipelined-prefill takes auto, on or off, not '%s'\", v); return RAD_E_INVAL; }\n"
        "        }\n"
        "        else if (S(\"--pipelined-prefill-min-rows\")) {\n"
        "            if (!(v = need(i))) return RAD_E_INVAL;\n"
        "            int64_t n = 0;\n"
        "            if (!parse_int(a.c_str(), v, 0, INT64_MAX, &n)) return RAD_E_INVAL;\n"
        "            /* Both pieces of a pipelined step are whole 64-row bands (pipe_rows), so at least 128. */\n"
        "            if (n != 0 && n < 128) {\n"
        "                RAD_ERR(\"--pipelined-prefill-min-rows: %lld is below 128 (0 is the plugin's own)\", (long long)n);\n"
        "                return RAD_E_INVAL;\n"
        "            }\n"
        "            c->pipelined_prefill_min_rows = n;\n"
        "        }\n"
        "        else if (S(\"--tp-ownership\")) { /* BC_VTP14_N2_PARSE */\n",
        guard="BC_VTP14_PIPE_PARSE",
        max_span=2,
    ),
    _edit(
        "pipe-explain",
        "    { \"--tp-ownership\",       [](const Config& c) { /* BC_VTP14_N2_EXPLAIN */\n",
        "    { \"--pipelined-prefill\",  [](const Config& c) { /* BC_VTP14_PIPE_EXPLAIN */\n"
        "          return std::string(c.pipelined_prefill == 1 ? \"on\" : c.pipelined_prefill == 2 ? \"off\" : \"auto\"); } },\n"
        "    { \"--pipelined-prefill-min-rows\", [](const Config& c) {\n"
        "          return c.pipelined_prefill_min_rows ? std::to_string(c.pipelined_prefill_min_rows)\n"
        "                                              : std::string(\"the plugin's own\"); } },\n"
        "    { \"--tp-ownership\",       [](const Config& c) { /* BC_VTP14_N2_EXPLAIN */\n",
        guard="BC_VTP14_PIPE_EXPLAIN",
        max_span=2,
    ),
))

CONFIG_TEST = FilePatch(path="tests/config_test.cpp", language="none", description="VTP14 option inventory", edits=(
    _edit(
        "pipe-config-test",
        "        { \"--tp-ownership\", { \"",
        "        { \"--pipelined-prefill\", { \"on\" } }, { \"--pipelined-prefill-min-rows\", { \"2048\" } }, /* BC_VTP14_PIPE_CONFIG_TEST */\n"
        "        { \"--tp-ownership\", { \"",
        guard="BC_VTP14_PIPE_CONFIG_TEST",
        max_span=1,
    ),
))

ENGINE = FilePatch(path="core/engine_bringup.cpp", language="none", description="VTP14 pipelined prefill build context", edits=(
    _edit(
        "pipe-build-ctx",
        "    bc.tp_ownership = cfg_.tp_ownership.empty() ? nullptr : cfg_.tp_ownership.c_str();\n",
        "    bc.tp_ownership = cfg_.tp_ownership.empty() ? nullptr : cfg_.tp_ownership.c_str();\n"
        "    /* BC_VTP14_PIPE_BUILD_CTX */\n"
        "    bc.pipelined_prefill = cfg_.pipelined_prefill;\n"
        "    bc.pipelined_prefill_min_rows = cfg_.pipelined_prefill_min_rows;\n",
        guard="BC_VTP14_PIPE_BUILD_CTX",
    ),
))

QWEN = FilePatch(path="arch/qwen35_fp8/qwen35_fp8.cpp", language="none", description="VTP14 pipelined prefill mode", edits=(
    _edit(
        "pipe-model-field",
        "    bool      pipe_on = false;\n",
        "    bool      pipe_on = false;\n"
        "    int64_t   pipe_min_rows = 0;   /* BC_VTP14_PIPE_FIELD: the narrowest pipelined step (rad0020) */\n",
        guard="BC_VTP14_PIPE_FIELD",
    ),
    _edit(
        "pipe-mode",
        "    if (m.a_h.q8_fed) {\n"
        "        if (g.world == 2 && g.max_tok >= kPipeMinRows && ctx->rank == 0)\n",
        "    /* BC_VTP14_PIPE_MODE (rad0020): --pipelined-prefill. auto is the two-rank rule above, and also\n"
        "     * three ranks under --tp-ownership; on is any world of two or more; off never. The narrowest\n"
        "     * step is --pipelined-prefill-min-rows, or kPipeMinRows. */\n"
        "    m.pipe_min_rows = ctx->pipelined_prefill_min_rows > 0 ? ctx->pipelined_prefill_min_rows : kPipeMinRows;\n"
        "    const bool pipe_want = ctx->pipelined_prefill == BC_PIPE_OFF ? false\n"
        "                         : ctx->pipelined_prefill == BC_PIPE_ON ? g.world >= 2\n"
        "                         : (g.world == 2 || (g.world == 3 && g.tp_owned));\n"
        "    if (m.a_h.q8_fed) {\n"
        "        if (pipe_want && g.max_tok >= m.pipe_min_rows && ctx->rank == 0)\n",
        guard="BC_VTP14_PIPE_MODE",
    ),
    _edit(
        "pipe-on",
        "        m.pipe_on = g.world == 2 && g.max_tok >= kPipeMinRows;\n",
        "        m.pipe_on = pipe_want && g.max_tok >= m.pipe_min_rows; /* BC_VTP14_PIPE_ON */\n",
        guard="BC_VTP14_PIPE_ON",
    ),
    _edit(
        "pipe-ready-rows",
        "    if (!m.pipe_on || !m.b_pipe_cu || T < kPipeMinRows || T > m.g.max_tok ||\n",
        "    if (!m.pipe_on || !m.b_pipe_cu || T < m.pipe_min_rows || T > m.g.max_tok || /* BC_VTP14_PIPE_READY */\n",
        guard="BC_VTP14_PIPE_READY",
    ),
    _edit(
        "pipe-note-rows",
        "                    (long long)kPipeMinRows);\n",
        "                    (long long)m.pipe_min_rows); /* BC_VTP14_PIPE_NOTE_ROWS */\n",
        guard="BC_VTP14_PIPE_NOTE_ROWS",
    ),
))

PATCHES = (ABI, RAD_CORE, CONFIG, CONFIG_TEST, ENGINE, QWEN)
