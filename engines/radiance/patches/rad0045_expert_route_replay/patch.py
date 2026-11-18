"""VTP44: `--expert-route-replay FILE`, replay a recorded expert route (rad0044's file) in a measured run.

KLD against a reference cannot rank KV caches on a routed model because any change in the arithmetic re-draws the expert
choices (FLASHNEXT-ROUTING.md); replay makes the measured run take the reference run's choices, so only the cache differs.

- core: the option (config, help, explain, config_test value).
"""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

RATIONALE = "VTP44: the expert route replay as an engine option, beside --expert-stats"


def _after(id: str, line: str, text: str, *, guard: str) -> Edit:
    return Edit(
        id=id, anchor="^" + re.escape(line) + "$", mode="insert_after", text="\n" + text,
        guard=re.escape(guard), expect_matches=1, max_span_lines=1, rationale=RATIONALE,
    )


RAD_CORE = FilePatch(path="core/rad_core.h", language="none", description="VTP44 route replay config", edits=(
    _after(
        "rp-config", "    std::string expert_stats;",
        "    /* BC_VTP44_ROUTE_REPLAY_CONFIG: --expert-route-replay FILE: take each token's experts from this recorded route\n"
        "     * (--expert-route-record's file) instead of the router's own choice. Empty is off. */\n"
        "    std::string expert_route_replay;",
        guard="BC_VTP44_ROUTE_REPLAY_CONFIG",
    ),
))

CONFIG = FilePatch(path="core/config.cpp", language="none", description="VTP44 --expert-route-replay", edits=(
    Edit(
        id="rp-help",
        anchor=re.escape('    { "--expert-profile",     "FILE",   "rank the expert plane'),
        mode="insert_before",
        text='    { "--expert-route-replay", "FILE",  "BigCherry VTP44: take each token\'s experts from a route recorded by\\n"\n'
             '    "                                    --expert-route-record instead of the router\'s own choice (one\\n"\n'
             '    "                                    sequence, no speculation, same model and prompts)" },'
             ' /* BC_VTP44_ROUTE_REPLAY_HELP */\n',
        guard=re.escape("BC_VTP44_ROUTE_REPLAY_HELP"), expect_matches=1, max_span_lines=1, rationale=RATIONALE,
    ),
    _after(
        "rp-parse",
        '        else if (S("--expert-stats"))          { if (!(v = need(i))) return RAD_E_INVAL; c->expert_stats = v; }',
        '        else if (S("--expert-route-replay"))   { if (!(v = need(i))) return RAD_E_INVAL; c->expert_route_replay = v; }'
        ' /* BC_VTP44_ROUTE_REPLAY_PARSE */',
        guard="BC_VTP44_ROUTE_REPLAY_PARSE",
    ),
    _after(
        "rp-explain",
        '    { "--expert-stats",       [](const Config& c) { return or_none(c.expert_stats, "off"); } },',
        '    { "--expert-route-replay", [](const Config& c) { return or_none(c.expert_route_replay, "off"); } },'
        ' /* BC_VTP44_ROUTE_REPLAY_EXPLAIN */',
        guard="BC_VTP44_ROUTE_REPLAY_EXPLAIN",
    ),
))

CONFIG_TEST = FilePatch(path="tests/config_test.cpp", language="none", description="VTP44 option inventory", edits=(
    _after(
        "rp-config-test", '        { "--expert-stats", { "/s/experts.json" } },',
        '        { "--expert-route-replay", { "/s/route.bin" } }, /* BC_VTP44_ROUTE_REPLAY_CONFIG_TEST */',
        guard="BC_VTP44_ROUTE_REPLAY_CONFIG_TEST",
    ),
))

PATCHES = (RAD_CORE, CONFIG, CONFIG_TEST)
