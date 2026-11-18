"""VTP44: `--expert-route-replay FILE`, replay a recorded expert route (rad0044's file) in a measured run.

KLD against a reference cannot rank KV caches on a routed model because any change in the arithmetic re-draws the expert
choices (FLASHNEXT-ROUTING.md); replay makes the measured run take the reference run's choices, so only the cache differs.

- core: the option (config, help, explain, config_test value).
- libr4d (the sources the libr3 plugin is built from): a new op `router_topk_replay` (logits, forced ids -> expert ids,
  expert weights). It is router_topk's row function with the selection loop skipped: lane r takes the recorded expert r of
  the row, and the weights come from this run's own logits over that set, by the same expression. An id outside
  [0, n_expert) is taken as expert 0 (the host validates the record, so only a generic launch reaches it).
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

def _before(id: str, line: str, text: str, *, guard: str, max_span: int = 1) -> Edit:
    return Edit(
        id=id, anchor="^" + re.escape(line), mode="insert_before", text=text,
        guard=re.escape(guard), expect_matches=1, max_span_lines=max_span, rationale=RATIONALE,
    )


def _replace(id: str, old: str, new: str, *, guard: str, max_span: int) -> Edit:
    return Edit(
        id=id, anchor=re.escape(old), mode="replace", text=new,
        guard=re.escape(guard), expect_matches=1, max_span_lines=max_span, rationale=RATIONALE,
    )


R4D_MOE = FilePatch(path="libr4d/r4d_moe.hip", language="none", description="VTP44 router_topk_replay kernel", edits=(
    _after(
        "rp-sink", "    Protect  prot;",
        "    const int32_t* forced = nullptr;   /* BC_VTP44_ROUTE_REPLAY_SINK: router_topk_replay's recorded ids, [row, top_k]; null selects */",
        guard="BC_VTP44_ROUTE_REPLAY_SINK",
    ),
    _replace(
        "rp-select", "    uint32_t nsel  = 0;\n    for (uint32_t r = 0; r < top_k; ++r) {",
        "    uint32_t nsel  = 0;\n"
        "    for (uint32_t r = 0; r < (snk.forced ? 0u : top_k); ++r) {   /* BC_VTP44_ROUTE_REPLAY_SELECT: replay skips the selection */",
        guard="BC_VTP44_ROUTE_REPLAY_SELECT", max_span=2,
    ),
    _before(
        "rp-keys", "    const float    p  = __expf(bf16_to_f32(topk_key_raw(mykey)) - mx) * inv;",
        "    if (snk.forced) {   /* BC_VTP44_ROUTE_REPLAY_KEYS: lane r takes the recorded expert r; the weights below are this run's own */\n"
        "        if (lane < top_k) {\n"
        "            const uint32_t fi = (uint32_t)snk.forced[(size_t)m * top_k + lane];\n"
        "            const uint32_t id = fi < n_expert ? fi : 0u;\n"
        "            const uint32_t kk = topk_key(row[id], id);\n"
        "            mykey = kk ? kk : (~id & 0xFFFFu);   /* a NaN logit keeps the expert; its weight is NaN, as the logit says */\n"
        "        }\n"
        "        nsel = top_k;\n"
        "    }\n",
        guard="BC_VTP44_ROUTE_REPLAY_KEYS",
    ),
    _before(
        "rp-kernel", "/* The pair in one launch. Operands are the two ops' own, selection first:",
        r"""/* BC_VTP44_ROUTE_REPLAY_KERNEL: router_topk_replay (rad0045). Operands: logits, forced (i32 [M, top_k]) -> expert_ids,
 * expert_w. router_topk's row function with the recorded experts in place of the selection. */
template <uint32_t NW, uint32_t MAXOWN>
__global__ __launch_bounds__(NW * kWave) void router_topk_replay_kernel(
    const uint16_t* __restrict__ lg, const int32_t* __restrict__ forced, int32_t* __restrict__ ids,
    uint16_t* __restrict__ w, uint32_t M, uint32_t n_expert, uint32_t top_k, uint32_t lg_ld, int norm) {
    const uint32_t wave = threadIdx.x / kWave, lane = threadIdx.x % kWave;
    const uint32_t m = blockIdx.x * NW + wave;
    if (m >= M) return;
    router_topk_row<MAXOWN>(lg, ids, w, m, n_expert, top_k, lg_ld, norm, lane,
                            SlotSink{nullptr, 0, 0, Protect{}, forced});
}

extern "C" int r4d_rad_router_topk_replay(const RadArgs* a, RadStream s) {
    const RadTensor* lg = r4d_opt(a, 0);
    const RadTensor* fo = r4d_opt(a, 1);
    const RadTensor* ids = r4d_opt(a, 2);
    const RadTensor* w = r4d_opt(a, 3);
    if (!lg || !fo || !ids || !w || !lg->data || !fo->data || !ids->data || !w->data) return R4D_NO(RAD_E_INVAL);
    const long long n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long top_k    = r4d_geti_or(a, "top_k", 0);
    const long long norm     = r4d_geti_or(a, "norm", 0);
    if (n_expert <= 0 || n_expert > (long long)kMaxExpert || top_k <= 0 || top_k > n_expert) return R4D_NO(RAD_E_SHAPE);
    if (top_k > (long long)kWave) return R4D_NO(RAD_E_SHAPE);
    if (lg->dtype != RAD_BF16 || w->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (ids->dtype != RAD_I32 || fo->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    long long M = 0;
    const long long lg_ld = r4d_rows_pitch(lg, n_expert, &M);
    if (lg_ld < 0 || M <= 0) return R4D_NO(RAD_E_STRIDE);
    if (!r4d_contig(fo) || !r4d_contig(ids) || !r4d_contig(w)) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(fo) < M * top_k || r4d_numel(ids) < M * top_k || r4d_numel(w) < M * top_k) return R4D_NO(RAD_E_SHAPE);
    constexpr uint32_t NW = 4;
    const dim3 grid((unsigned)((M + NW - 1) / NW)), blk(NW * kWave);
    if (n_expert <= (long long)(8 * kWave))
        hipLaunchKernelGGL((router_topk_replay_kernel<NW, 8>), grid, blk, 0, (hipStream_t)s,
                           (const uint16_t*)lg->data, (const int32_t*)fo->data, (int32_t*)ids->data, (uint16_t*)w->data,
                           (uint32_t)M, (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)lg_ld, (int)norm);
    else
        hipLaunchKernelGGL((router_topk_replay_kernel<NW, 16>), grid, blk, 0, (hipStream_t)s,
                           (const uint16_t*)lg->data, (const int32_t*)fo->data, (int32_t*)ids->data, (uint16_t*)w->data,
                           (uint32_t)M, (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)lg_ld, (int)norm);
    R4D_MOE_CHK();
    return RAD_OK;
}

""",
        guard="BC_VTP44_ROUTE_REPLAY_KERNEL",
    ),
))

R4D_PLUGIN_H = FilePatch(path="libr4d/r4d_plugin.h", language="none", description="VTP44 router_topk_replay prototypes", edits=(
    _after(
        "rp-proto", "int     r4d_rad_router_topk        (const RadArgs*, RadStream);",
        "int     r4d_rad_router_topk_replay (const RadArgs*, RadStream);   /* BC_VTP44_ROUTE_REPLAY_PROTO */",
        guard="BC_VTP44_ROUTE_REPLAY_PROTO",
    ),
    _after(
        "rp-shape-proto", "int r4d_shape_router_topk    (const RadParam*, int, int operand, RadOpdDesc*);",
        "int r4d_shape_router_topk_replay(const RadParam*, int, int operand, RadOpdDesc*);   /* BC_VTP44_ROUTE_REPLAY_SHAPE_PROTO */",
        guard="BC_VTP44_ROUTE_REPLAY_SHAPE_PROTO",
    ),
))

R4D_SHAPES = FilePatch(path="libr4d/r4d_shapes.cpp", language="none", description="VTP44 router_topk_replay shape", edits=(
    _before(
        "rp-shape", 'extern "C" int r4d_shape_router_topk(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {',
        'extern "C" int r4d_shape_router_topk_replay(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {   /* BC_VTP44_ROUTE_REPLAY_SHAPE */\n'
        '    const int64_t M  = pi(p, n_p, "M");\n'
        '    const int64_t ne = pi(p, n_p, "n_expert");\n'
        '    const int64_t k  = pi(p, n_p, "top_k");\n'
        '    if (ne <= 0 || ne > 256 || k <= 0 || k > ne) return RAD_E_UNSUPPORTED;\n'
        '    const uint32_t D = dt_act(p, n_p, "dtype");\n'
        '    const RadOpdDesc l[] = {\n'
        '        opd(D, M, ne),\n'
        '        opd((uint32_t)RAD_I32, M, k),\n'
        '        opd((uint32_t)RAD_I32, M, k),\n'
        '        opd(D, M, k),\n'
        '    };\n'
        '    R4D_PICK(l);\n'
        '}\n\n',
        guard="BC_VTP44_ROUTE_REPLAY_SHAPE",
    ),
))

R4D_ROWS = FilePatch(path="libr4d/r4d_rows.cpp", language="none", description="VTP44 router_topk_replay row", edits=(
    _before(
        "rp-operands", "static const RadParamSpec pMoeScatter[] = {",
        "/* BC_VTP44_ROUTE_REPLAY_OPERANDS: router_topk's operands with the recorded experts after the logits (rad0045). */\n"
        "static const RadOperandSpec oRouterTopkReplay[] = {\n"
        '    { "logits",     RAD_OPD_IN,  0 },\n'
        '    { "forced",     RAD_OPD_IN,  0 },\n'
        '    { "expert_ids", RAD_OPD_OUT, 0 },\n'
        '    { "expert_w",   RAD_OPD_OUT, 0 },\n'
        "};\n",
        guard="BC_VTP44_ROUTE_REPLAY_OPERANDS",
    ),
    _before(
        "rp-schema", '    SCHEMA("router_topk_scatter", pRouterTopkScatter, oRouterTopkScatter,',
        '    SCHEMA("router_topk_replay", pRouterTopk, oRouterTopkReplay,   /* BC_VTP44_ROUTE_REPLAY_SCHEMA */\n'
        '           "`router_topk` with the experts given: lane r of a row takes `forced[row, r]` instead of the r-th largest "\n'
        '           "logit, and the weights are this run\'s own softmax over ALL n_expert logits at those experts (renormalised "\n'
        '           "if `norm`). For replaying a recorded route; the host validates the ids."),\n',
        guard="BC_VTP44_ROUTE_REPLAY_SCHEMA",
    ),
    _before(
        "rp-row", '{ "router_topk_scatter_bf16", "router_topk_scatter", "moe",',
        '/* BC_VTP44_ROUTE_REPLAY_ROW */\n'
        '{ "router_topk_replay_bf16", "router_topk_replay", "moe",\n'
        '  "router_topk with the experts given (a recorded route): the weights are this run\'s own",\n'
        '  "n_expert <= 512, top_k <= min(n_expert, 32); bf16 logits and weights, i32 forced and expert ids",\n'
        '  "bf16 logits + i32 forced ids in, bf16 weights + i32 expert ids out",\n'
        '  RAD_DOMAIN_DEVICE, R4D_PRIO_ONLY,\n'
        '  ROW(cRouterTopK), NOTUNE,\n'
        '  nullptr, nullptr, r4d_rad_router_topk_replay, nullptr, nullptr, nullptr,\n'
        '  r4d_shape_router_topk_replay },\n\n',
        guard="BC_VTP44_ROUTE_REPLAY_ROW", max_span=1,
    ),
))

PATCHES = (RAD_CORE, CONFIG, CONFIG_TEST, R4D_MOE, R4D_PLUGIN_H, R4D_SHAPES, R4D_ROWS)
