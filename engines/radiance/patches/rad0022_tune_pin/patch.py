"""rad0022 (VTP31): `--tune-pin SPEC`, radiance's kernel tune pin as an engine option.

Radiance's selector takes a kernel's tunable axes from the tune cache or the kernel's defaults, and
`RADIANCE_TUNE=op:key=value[,key=value][;op:...]` pins them (core/plugin/select.cpp; spec section 15). The pin is
environment only, so a run's manifest could not carry it. `--tune-pin SPEC` takes the same text and sets
`RADIANCE_TUNE` before the plugins load, so the selector, its checks and its `--debug-graph` labels are radiance's
own: an axis or value a kernel does not offer is still a hard error naming what it has.

The first use is the three-rank exact all-reduce's launch geometry (VTP31): its default grid scales with the message
at 1024 threads a block, tuned on two- and four-rank worlds; `all_reduce:nblocks=4,nthreads=256` serves a mixed
three-card world faster at every size with identical output. Nothing about a model, world or card is built in; the
value is the layout's or the run's.

- core: `--tune-pin SPEC` (config, help, explain, config_test value). The option checks the form only.
- engine_bringup: before the plugins load, the spec becomes `RADIANCE_TUNE`. If `RADIANCE_TUNE` is already set to a
  different value the engine refuses to start (two pins, one silently lost, is worse than none).

Without --tune-pin nothing changes. Source-pinned to radiance v1.17.0 (19d4e8fe).
"""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

# The config_test inventory value. Test data only.
EXAMPLE = "all_reduce:nblocks=4,nthreads=256"


def _edit(id: str, before: str, after: str, *, guard: str, max_span: int | None = None) -> Edit:
    return Edit(
        id=id,
        anchor=re.escape(before),
        mode="replace",
        text=after,
        guard=re.escape(guard),
        expect_matches=1,
        max_span_lines=max_span or max(6, before.count("\n") + 1),
        rationale="VTP31: the kernel tune pin as an engine option",
    )


RAD_CORE = FilePatch(path="core/rad_core.h", language="none", description="VTP31 tune pin config", edits=(
    _edit(
        "tp-config",
        "    int   p2p = RAD_P2P_AUTO;         /* --p2p auto|on|off */\n",
        "    /* BC_VTP31_TUNE_PIN_CONFIG: --tune-pin, RADIANCE_TUNE's text; empty leaves the environment as it is. */\n"
        "    std::string tune_pin;\n"
        "    int   p2p = RAD_P2P_AUTO;         /* --p2p auto|on|off */\n",
        guard="BC_VTP31_TUNE_PIN_CONFIG",
        max_span=2,
    ),
))

CONFIG = FilePatch(path="core/config.cpp", language="none", description="VTP31 --tune-pin", edits=(
    _edit(
        "tp-help",
        "    { \"--p2p\",                \"MODE\",   \"how the collectives reach the other cards: auto (default)\\n\"",
        "    { \"--tune-pin\",           \"SPEC\",   \"BigCherry VTP31: pin kernels' tunable axes, as\\n\"\n"
        "                                        \"RADIANCE_TUNE: op:key=value[,key=value][;op:...]\\n\"\n"
        "                                        \"(an axis a kernel does not offer is an error)\" }, /* BC_VTP31_TUNE_PIN_HELP */\n"
        "    { \"--p2p\",                \"MODE\",   \"how the collectives reach the other cards: auto (default)\\n\"",
        guard="BC_VTP31_TUNE_PIN_HELP",
        max_span=1,
    ),
    _edit(
        "tp-parse",
        "        else if (S(\"--p2p\")) {\n"
        "            if (!(v = need(i))) return RAD_E_INVAL;",
        "        else if (S(\"--tune-pin\")) { /* BC_VTP31_TUNE_PIN_PARSE */\n"
        "            if (!(v = need(i))) return RAD_E_INVAL;\n"
        "            /* The form only; the selector checks every axis and value against the kernel it pins. */\n"
        "            if (!*v || std::strspn(v, \"abcdefghijklmnopqrstuvwxyz0123456789_=,;:\") != std::strlen(v) ||\n"
        "                !std::strchr(v, ':') || !std::strchr(v, '=')) {\n"
        "                RAD_ERR(\"--tune-pin: '%s' is not a tune pin (op:key=value[,key=value][;op:...])\", v);\n"
        "                return RAD_E_INVAL;\n"
        "            }\n"
        "            c->tune_pin = v;\n"
        "        }\n"
        "        else if (S(\"--p2p\")) {\n"
        "            if (!(v = need(i))) return RAD_E_INVAL;",
        guard="BC_VTP31_TUNE_PIN_PARSE",
    ),
    _edit(
        "tp-explain",
        "    { \"--p2p\",                [](const Config& c) {",
        "    { \"--tune-pin\",           [](const Config& c) { /* BC_VTP31_TUNE_PIN_EXPLAIN */\n"
        "          return c.tune_pin.empty() ? std::string(\"none (tune cache, kernel defaults)\") : c.tune_pin; } },\n"
        "    { \"--p2p\",                [](const Config& c) {",
        guard="BC_VTP31_TUNE_PIN_EXPLAIN",
        max_span=1,
    ),
))

CONFIG_TEST = FilePatch(path="tests/config_test.cpp", language="none", description="VTP31 option inventory", edits=(
    _edit(
        "tp-config-test",
        "        { \"--max-source-pixels\", { \"1000000\" } },\n",
        "        { \"--max-source-pixels\", { \"1000000\" } },\n"
        "        { \"--tune-pin\", { \"" + EXAMPLE + "\" } }, /* BC_VTP31_TUNE_PIN_CONFIG_TEST */\n",
        guard="BC_VTP31_TUNE_PIN_CONFIG_TEST",
        max_span=2,
    ),
))

ENGINE = FilePatch(path="core/engine_bringup.cpp", language="none", description="VTP31 tune pin to the selector", edits=(
    _edit(
        "tp-include",
        "#include \"rad_internal.h\"\n",
        "#include \"rad_internal.h\"\n"
        "#include <cstdlib> /* BC_VTP31_TUNE_PIN_INCLUDE: setenv */\n",
        guard="BC_VTP31_TUNE_PIN_INCLUDE",
        max_span=2,
    ),
    _edit(
        "tp-apply",
        "int Engine::load_plugins() {\n",
        "int Engine::load_plugins() {\n"
        "    /* BC_VTP31_TUNE_PIN_APPLY: --tune-pin is RADIANCE_TUNE, set before the selector first reads it (at\n"
        "     * declare, which follows). A different RADIANCE_TUNE already set is refused, not overridden. */\n"
        "    if (!cfg_.tune_pin.empty()) {\n"
        "        const char* env = std::getenv(\"RADIANCE_TUNE\");\n"
        "        if (env && cfg_.tune_pin != env) {\n"
        "            RAD_ERR(\"--tune-pin '%s' and RADIANCE_TUNE='%s' disagree; give one\", cfg_.tune_pin.c_str(), env);\n"
        "            return RAD_E_INVAL;\n"
        "        }\n"
        "        if (!env && ::setenv(\"RADIANCE_TUNE\", cfg_.tune_pin.c_str(), 1) != 0) {\n"
        "            RAD_ERR(\"--tune-pin: could not set RADIANCE_TUNE\");\n"
        "            return RAD_E_INVAL;\n"
        "        }\n"
        "        RAD_INFO(\"kernel tune pin (--tune-pin): %s\", cfg_.tune_pin.c_str());\n"
        "    }\n",
        guard="BC_VTP31_TUNE_PIN_APPLY",
        max_span=2,
    ),
))

PATCHES = (RAD_CORE, CONFIG, CONFIG_TEST, ENGINE)
