"""VTP44: `--expert-route-replay FILE`, replay a recorded expert route (rad0044's file) in a measured run.

KLD against a reference cannot rank KV caches on a routed model because any change in the arithmetic re-draws the expert
choices (FLASHNEXT-ROUTING.md); replay makes the measured run take the reference run's choices, so only the cache differs.

- core: the option (config, help, explain, config_test value).
- libr4d (the sources the libr3 plugin is built from): a new op `router_topk_replay` (logits, forced ids -> expert ids,
  expert weights). It is router_topk's row function with the selection loop skipped: lane r takes the recorded expert r of
  the row, and the weights come from this run's own logits over that set, by the same expression. An id outside
  [0, n_expert) is taken as expert 0 (the host validates the record, so only a generic launch reaches it).
- runtime: `rad_route_replay_load` (abi, wrapper, test fake). The engine reads and checks the record against the model at
  start (header strictly parsed, routed layers, n_expert, rows, documents and their token digests, id range, file size,
  at most 1 GiB) and every rank's context keeps it as a device table. Each pass must be one sequence's prefill without
  drafting whose positions continue or restart at 0 for the next document; its rows are looked up through the document
  starts, so chunk size cannot matter. After every pass each routed layer must have been asked for exactly the pass's rows
  (an architecture without the hook fails the pass instead of selecting its own experts). Anything else, a run that does
  not consume the whole record, or documents whose token digests differ from the recorded ones, fails the run. Recorded
  passes (the step tape) are off while replaying.
- block (arch/common/rad_block_moe_q.h): declares `router_topk_replay` and a persistent step-tall forced-ids buffer when a
  plugin offers the op; on a replayed pass it loads the pass's rows, skips the fused routers and issues the replay op. With
  no such op the core fails the step.
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
        '    if (ne <= 0 || ne > 512 || k <= 0 || k > ne || k > 32) return RAD_E_UNSUPPORTED;\n'
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

R_RUNTIME_H = FilePatch(path="abi/rad_runtime.h", language="none", description="VTP44 route replay ABI", edits=(
    _after(
        "rp-abi", "int32_t* rad_route_counts(RadCtx* c, int layer, int64_t n_expert);",
        "/* BC_VTP44_ROUTE_REPLAY_ABI (rad0045): the experts to take for rows [r0, r0 + rows) of this pass's layer when a recorded\n"
        " * route is being replayed: they are copied, stream-ordered, into `dst` (rows of `top_k` int32 at row r0). Returns 1 when\n"
        " * replaying (a null `dst` then fails the step: the block has no way to use them), 0 when not replaying or the layer is\n"
        " * not a routed trunk layer, <0 on error (the step is failed too). */\n"
        "int rad_route_replay_load(RadCtx* c, int layer, rad_buf dst, int64_t r0, int64_t rows, int64_t top_k);",
        guard="BC_VTP44_ROUTE_REPLAY_ABI",
    ),
))

ISSUE_CPP = FilePatch(path="core/runtime/issue.cpp", language="none", description="VTP44 route replay wrapper", edits=(
    Edit(
        id="rp-wrapper",
        anchor=re.escape("    return c ? static_cast<rad::Ctx*>(c)->route_counts(layer, n_expert) : nullptr;\n}\n"),
        mode="insert_after",
        text="\n/* BC_VTP44_ROUTE_REPLAY_WRAPPER */\n"
             "int rad_route_replay_load(RadCtx* c, int layer, rad_buf dst, int64_t r0, int64_t rows, int64_t top_k) {\n"
             "    return c ? static_cast<rad::Ctx*>(c)->route_replay_load(layer, dst, r0, rows, top_k) : RAD_E_INVAL;\n"
             "}\n",
        guard=re.escape("BC_VTP44_ROUTE_REPLAY_WRAPPER"), expect_matches=1, max_span_lines=3, rationale=RATIONALE,
    ),
))

ARCH_FAKES = FilePatch(path="tests/arch_fakes.h", language="none", description="VTP44 route replay test fake", edits=(
    Edit(
        id="rp-fake",
        anchor=re.escape("    (void)c; (void)layer; (void)n_expert;\n    return nullptr;\n}\n"),
        mode="insert_after",
        text="/* BC_VTP44_ROUTE_REPLAY_FAKE: no recorded route here, so nothing is replayed. */\n"
             "extern \"C\" int rad_route_replay_load(RadCtx*, int, rad_buf, int64_t, int64_t, int64_t) { return 0; }\n",
        guard=re.escape("BC_VTP44_ROUTE_REPLAY_FAKE"), expect_matches=1, max_span_lines=4, rationale=RATIONALE,
    ),
))

CTX_H = FilePatch(path="core/runtime/ctx.h", language="none", description="VTP44 route replay in the context", edits=(
    _before(
        "rp-table", "struct CtxDesc {",
        "/* BC_VTP44_ROUTE_REPLAY_TABLE: a recorded expert route (--expert-route-record's file), read once and shared by every rank. */\n"
        "struct RouteReplay {\n"
        "    int32_t              top_k = 0;\n"
        "    int64_t              rows = 0;                        /* rows per layer, over all recorded documents */\n"
        "    std::vector<int>     layers;                          /* the routed trunk layers, ascending */\n"
        "    std::vector<int64_t> doc_starts;                      /* the first row of each recorded document */\n"
        "    std::vector<std::vector<int32_t>> ids;                /* [layers.size()][rows * top_k] */\n"
        "    std::vector<uint64_t> doc_hashes;                     /* each document's FNV-1a 64 over its token ids */\n"
        "};\n\n",
        guard="BC_VTP44_ROUTE_REPLAY_TABLE",
    ),
    _after(
        "rp-desc", "    bool            record_passes = true;",
        "    const RouteReplay* route_replay = nullptr;   /* BC_VTP44_ROUTE_REPLAY_DESC: replay this route (see rad_route_replay_load); null is off */",
        guard="BC_VTP44_ROUTE_REPLAY_DESC",
    ),
    _after(
        "rp-api", "    int64_t           route_n_expert() const { return route_n_expert_; }",
        "    /* BC_VTP44_ROUTE_REPLAY_API: replaying a recorded route (rad0045). */\n"
        "    int  route_replay_load(int layer, rad_buf dst, int64_t r0, int64_t rows, int64_t top_k);\n"
        "    const std::string& route_replay_problem() const { return route_rp_why_; }\n"
        "    int64_t route_replay_rows_used() const { return route_rp_used_; }\n"
        "    int64_t route_replay_rows_total() const { return route_rp_ ? route_rp_->rows : 0; }\n"
        "    const std::vector<uint64_t>& route_replay_doc_hashes() const { return route_rp_hash_; }\n"
        "    const std::vector<uint64_t>& route_replay_record_hashes() const { return route_rp_->doc_hashes; }",
        guard="BC_VTP44_ROUTE_REPLAY_API",
    ),
    _after(
        "rp-members", "    int32_t*               route_dev_      = nullptr;   /* card */",
        "    /* BC_VTP44_ROUTE_REPLAY_MEMBERS */\n"
        "    const RouteReplay*     route_rp_       = nullptr;\n"
        "    int32_t*               route_rp_dev_   = nullptr;   /* card: [routed layer][route_rp_->rows][top_k] */\n"
        "    std::vector<int>       route_rp_at_;                /* layer -> index in route_rp_->layers, or -1 */\n"
        "    int                    route_rp_doc_   = -1;\n"
        "    int64_t                route_rp_base_  = 0, route_rp_next_pos_ = 0, route_rp_used_ = 0;\n"
        "    std::string            route_rp_why_;\n"
        "    std::vector<int64_t>   route_rp_cov_;               /* rows loaded this pass, per routed layer */\n"
        "    std::vector<uint64_t>  route_rp_hash_;              /* per document run so far: FNV-1a 64 of its token ids */\n"
        "    int  route_replay_begin();\n"
        "    int  route_replay_end();",
        guard="BC_VTP44_ROUTE_REPLAY_MEMBERS",
    ),
))

CTX_CPP = FilePatch(path="core/runtime/ctx.cpp", language="none", description="VTP44 route replay in the context", edits=(
    _after(
        "rp-init", "    record_passes_ = d.record_passes;",
        "    route_rp_ = d.route_replay; /* BC_VTP44_ROUTE_REPLAY_INIT */",
        guard="BC_VTP44_ROUTE_REPLAY_INIT",
    ),
    Edit(
        id="rp-tape-off",
        anchor=re.escape("&& route_rec_cap_ == 0;"),
        mode="replace",
        text="&& route_rec_cap_ == 0 && !route_rp_; /* BC_VTP44_ROUTE_REPLAY_TAPE: a played pass would not re-run the load */",
        guard=re.escape("BC_VTP44_ROUTE_REPLAY_TAPE"), expect_matches=1, max_span_lines=1, rationale=RATIONALE,
    ),
    _after(
        "rp-free",
        "    if (route_rec_dev_) { rad_dev_free(route_rec_dev_, RAD_MEM_DEVICE); route_rec_dev_ = nullptr; } /* BC_VTP44_ROUTE_RECORD_FREE */",
        "    if (route_rp_dev_) { rad_dev_free(route_rp_dev_, RAD_MEM_DEVICE); route_rp_dev_ = nullptr; } /* BC_VTP44_ROUTE_REPLAY_FREE */",
        guard="BC_VTP44_ROUTE_REPLAY_FREE",
    ),
    _after(
        "rp-begin", "    step_fail_.clear();",
        "    if (route_rp_) { const int rp = route_replay_begin(); if (rp < 0 && step_status_ >= 0) step_status_ = rp; } /* BC_VTP44_ROUTE_REPLAY_BEGIN */",
        guard="BC_VTP44_ROUTE_REPLAY_BEGIN",
    ),
    Edit(
        id="rp-pass-end",
        anchor=re.escape("        const int rf = route_flush();\n        if (rf < 0 && step_status_ >= 0) step_status_ = rf;\n    }\n"),
        mode="insert_after",
        text="    if (route_rp_) { /* BC_VTP44_ROUTE_REPLAY_PASS_END: every routed layer took the recorded experts for every row */\n"
             "        const int re = route_replay_end();\n"
             "        if (re < 0 && step_status_ >= 0) step_status_ = re;\n"
             "    }\n",
        guard=re.escape("BC_VTP44_ROUTE_REPLAY_PASS_END"), expect_matches=1, max_span_lines=4, rationale=RATIONALE,
    ),
    Edit(
        id="rp-impl",
        anchor="^" + re.escape("int Ctx::route_flush() {") + "$",
        mode="insert_before",
        text=r"""/* BC_VTP44_ROUTE_REPLAY_IMPL: replaying a recorded expert route (rad0045). The pass's rows are looked up through the
 * document starts: a restart at position 0 is the next recorded document, so the pass size cannot matter. Any
 * deviation fails the run, never a silent fallback to the router's own choice: a layer that was not asked for its rows
 * (an architecture without the hook) fails the pass, and the documents' token digests must match at the end. */
int Ctx::route_replay_begin() {
    if (!route_rp_ || !batch_) return RAD_OK;
    const RouteReplay& R = *route_rp_;
    auto bad = [&](const char* why) {
        if (route_rp_why_.empty()) { route_rp_why_ = why; RAD_ERR("expert route replay: %s", why); }
        return (int)RAD_E_INVAL;
    };
    if (!route_rp_why_.empty()) return RAD_E_INVAL;
    if (batch_->phase != RAD_PHASE_PREFILL || batch_->n_seq != 1 || batch_->n_spec != 0 || batch_->draft_pass != 0 ||
        !batch_->positions_h)
        return bad("only one-sequence prefill passes without drafting can be replayed");
    const int64_t n = batch_->n_tok;
    const int32_t* pos = batch_->positions_h;
    if (n <= 0) { std::fill(route_rp_cov_.begin(), route_rp_cov_.end(), (int64_t)0); return RAD_OK; }
    for (int64_t i = 1; i < n; ++i)
        if ((int64_t)pos[i] != (int64_t)pos[0] + i) return bad("token positions within a pass are not contiguous");
    if (!batch_->token_ids) return bad("the pass has no token ids to digest");
    if (!route_rp_dev_) {
        const int64_t n_trunk = program_->meta.n_layers;
        route_rp_at_.assign((size_t)n_trunk, -1);
        for (const WeightInfo& w : program_->weights) {
            const int32_t l = w.decl.group.layer;
            if (w.decl.group.expert >= 0 && l >= 0 && l < n_trunk) route_rp_at_[(size_t)l] = 0;
        }
        std::vector<int> layers;
        for (int64_t l = 0; l < n_trunk; ++l) {
            if (route_rp_at_[(size_t)l] < 0) continue;
            route_rp_at_[(size_t)l] = (int)layers.size();
            layers.push_back((int)l);
        }
        if (layers != R.layers) return bad("the record's routed layers are not this model's");
        const size_t per = (size_t)R.rows * (size_t)R.top_k;
        route_rp_dev_ = static_cast<int32_t*>(rad_dev_alloc(layers.size() * per * sizeof(int32_t), RAD_MEM_DEVICE));
        if (!route_rp_dev_) return bad("the table allocation failed");
        for (size_t i = 0; i < layers.size(); ++i)
            RAD_TRY(rad_memcpy_async(route_rp_dev_ + i * per, R.ids[i].data(), per * sizeof(int32_t), stream_));
        RAD_TRY(rad_stream_sync(stream_));
        route_rp_cov_.assign(layers.size(), 0);
    }
    if (pos[0] == 0) { ++route_rp_doc_; route_rp_hash_.push_back(14695981039346656037ull); }
    else if ((int64_t)pos[0] != route_rp_next_pos_) return bad("token positions do not continue the previous pass");
    if (route_rp_doc_ < 0 || route_rp_doc_ >= (int)R.doc_starts.size()) return bad("the run has more documents than the record");
    const int64_t d0 = R.doc_starts[(size_t)route_rp_doc_];
    const int64_t d1 = (size_t)route_rp_doc_ + 1 < R.doc_starts.size() ? R.doc_starts[(size_t)route_rp_doc_ + 1] : R.rows;
    if (d0 + pos[0] + n > d1) return bad("a document is longer than the record's");
    {   /* the document's identity: the same tokens as the recorded run's */
        std::vector<int32_t> tk((size_t)n);
        RAD_TRY(rad_memcpy_async(tk.data(), batch_->token_ids, (size_t)n * sizeof(int32_t), stream_));
        RAD_TRY(rad_stream_sync(stream_));
        uint64_t& h = route_rp_hash_.back();
        for (int64_t i = 0; i < n; ++i) { h ^= (uint64_t)(uint32_t)tk[(size_t)i]; h *= 1099511628211ull; }
    }
    route_rp_base_ = d0 + pos[0];
    route_rp_next_pos_ = (int64_t)pos[0] + n;
    route_rp_used_ += n;
    std::fill(route_rp_cov_.begin(), route_rp_cov_.end(), (int64_t)0);
    return RAD_OK;
}

int Ctx::route_replay_load(int layer, rad_buf dst, int64_t r0, int64_t rows, int64_t top_k) {
    if (!route_rp_) return 0;
    auto fail = [&](const char* why, int rc) {
        if (route_rp_why_.empty()) { route_rp_why_ = why; RAD_ERR("expert route replay: %s", why); }
        if (step_status_ >= 0) step_status_ = rc;
        return rc;
    };
    if (!route_rp_why_.empty()) return fail("", RAD_E_INVAL);
    if (route_rp_cov_.empty()) return fail("a layer asked for recorded experts before its pass was admitted", RAD_E_INVAL);
    if (layer < 0 || (size_t)layer >= route_rp_at_.size() || route_rp_at_[(size_t)layer] < 0) return 0;   /* an MTP head's layer */
    if (!dst) return fail("no kernel plugin offers router_topk_replay for this router", RAD_E_UNSUPPORTED);
    if (top_k != route_rp_->top_k) return fail("the model's top_k is not the record's", RAD_E_INVAL);
    int32_t* d = static_cast<int32_t*>(buf_ptr(dst));
    if (!d || !batch_ || rows <= 0 || r0 < 0 || r0 + rows > batch_->n_tok) return fail("a pass asked for rows it does not have", RAD_E_INVAL);
    const int64_t at = route_rp_at_[(size_t)layer];
    const int rc = rad_memcpy_async(d + r0 * top_k, route_rp_dev_ + ((at * route_rp_->rows) + route_rp_base_ + r0) * top_k,
                                    (size_t)(rows * top_k) * sizeof(int32_t), stream_);
    if (rc < 0) return fail("the copy of the recorded experts failed", rc);
    route_rp_cov_[(size_t)at] += rows;
    return 1;
}

/* After the pass: every routed layer must have been asked for exactly the pass's rows. An architecture that selects its
 * own experts (no hook) asks for none, and so fails here instead of reporting a replay that did not happen. */
int Ctx::route_replay_end() {
    if (!route_rp_ || route_rp_cov_.empty() || !route_rp_why_.empty() || !batch_ || batch_->n_tok <= 0) return RAD_OK;
    for (size_t i = 0; i < route_rp_cov_.size(); ++i)
        if (route_rp_cov_[i] != batch_->n_tok) {
            route_rp_why_ = "a routed layer did not take the recorded experts for exactly the pass's rows";
            RAD_ERR("expert route replay: %s (routed layer %d: %lld of %lld rows)", route_rp_why_.c_str(),
                    route_rp_->layers[i], (long long)route_rp_cov_[i], (long long)batch_->n_tok);
            return RAD_E_INVAL;
        }
    return RAD_OK;
}

""",
        guard=re.escape("BC_VTP44_ROUTE_REPLAY_IMPL"), expect_matches=1, max_span_lines=1, rationale=RATIONALE,
    ),
))

BRINGUP = FilePatch(path="core/engine_bringup.cpp", language="none", description="VTP44 route replay bring-up", edits=(
    _before(
        "rp-read", "int Engine::start() {",
        r"""/* BC_VTP44_ROUTE_REPLAY_READ: a record written by --expert-route-record (rad0044), read whole and checked against this
 * model: an incomplete file, another model, other routed layers, an id outside the model, a header that is not exactly
 * what the writer makes, or a file whose size is not its header's refuse the start. */
static int bc_route_replay_read(const std::string& path, int meta_layers, int meta_n_expert, const char* meta_name,
                                RouteReplay* out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { RAD_ERR("--expert-route-replay: cannot read %s", path.c_str()); return RAD_E_IO; }
    auto refuse = [&](const char* why) {
        RAD_ERR("--expert-route-replay: %s: %s", path.c_str(), why);
        std::fclose(f);
        return (int)RAD_E_INVAL;
    };
    std::string magic, h;
    int ch;
    while ((ch = std::fgetc(f)) != EOF && ch != '\n' && magic.size() < 16) magic += (char)ch;
    if (magic != "RADROUTE1") return refuse("not an expert route record");
    while ((ch = std::fgetc(f)) != EOF && ch != '\n') {
        h += (char)ch;
        if (h.size() > (1u << 20)) return refuse("the header is too long");
    }
    if (ch != '\n') return refuse("the header is missing");
    auto field = [&](const char* key) -> const char* {
        const std::string k = std::string("\"") + key + "\":";
        const size_t p = h.find(k);
        return p == std::string::npos ? nullptr : h.c_str() + p + k.size();
    };
    auto closes = [](char c) { return c == ',' || c == '}' || c == ']'; };
    auto num = [&](const char* key, long long* v) {
        const char* p = field(key);
        char* e = nullptr;
        if (!p) return false;
        errno = 0;
        *v = std::strtoll(p, &e, 10);
        return e != p && errno == 0 && closes(*e);
    };
    auto list = [&](const char* key, std::vector<long long>* v) {
        const char* p = field(key);
        if (!p || *p != '[') return false;
        for (++p; *p && *p != ']';) {
            char* e = nullptr;
            errno = 0;
            v->push_back(std::strtoll(p, &e, 10));
            if (e == p || errno != 0 || !closes(*e)) return false;
            p = e;
            if (*p == ',') ++p;
        }
        return *p == ']';
    };
    auto hashes = [&](const char* key, std::vector<uint64_t>* v) {   /* ["0123456789abcdef", ...] */
        const char* p = field(key);
        if (!p || *p != '[') return false;
        for (++p; *p && *p != ']';) {
            if (*p != '"') return false;
            char* e = nullptr;
            errno = 0;
            v->push_back((uint64_t)std::strtoull(p + 1, &e, 16));
            if (e != p + 17 || errno != 0 || *e != '"') return false;
            p = e + 1;
            if (*p == ',') ++p;
        }
        return *p == ']';
    };
    long long layers = 0, top_k = 0, n_expert = 0, rows = 0;
    std::vector<long long> routed, docs;
    const char* cp = field("complete");
    if (!cp || std::strncmp(cp, "true", 4) != 0 || !closes(cp[4])) return refuse("the record is incomplete");
    if (!num("layers", &layers) || !num("top_k", &top_k) || !num("n_expert", &n_expert) || !num("rows", &rows) ||
        !list("routed", &routed) || !list("doc_starts", &docs))
        return refuse("the header is not a record header");
    if (!hashes("doc_hashes", &out->doc_hashes) || out->doc_hashes.size() != docs.size())
        return refuse("the record has no document digests for its documents (record again with this build)");
    std::string want;
    for (const char* q = meta_name; *q; ++q) want += (*q == '"' || *q == '\\' || (unsigned char)*q < 0x20) ? '?' : *q;
    const char* mp = field("model");
    if (!mp || *mp != '"' || h.compare(mp - h.c_str(), want.size() + 2, "\"" + want + "\"") != 0)
        return refuse("the record is of another model");
    if (layers != meta_layers || n_expert != meta_n_expert) return refuse("the record's layer or expert count is not this model's");
    if (top_k <= 0 || top_k > 32 || rows <= 0 || routed.empty() || docs.empty() || docs[0] != 0)
        return refuse("the record's shape is invalid");
    for (size_t i = 0; i < routed.size(); ++i)
        if (routed[i] < 0 || routed[i] >= layers || (i && routed[i] <= routed[i - 1])) return refuse("the record's routed layers are invalid");
    for (size_t i = 0; i < docs.size(); ++i)
        if (docs[i] < 0 || docs[i] >= rows || (i && docs[i] <= docs[i - 1])) return refuse("the record's document starts are invalid");
    /* The table lives on every rank's card beyond the memory plan, so it is bounded; the size is checked in 128 bits
     * and against the file before anything is allocated. */
    const unsigned __int128 bytes = (unsigned __int128)rows * (unsigned __int128)top_k * routed.size() * sizeof(int32_t);
    if (bytes > ((unsigned __int128)1 << 30)) return refuse("larger than the 1 GiB a replay holds");
    const long here = std::ftell(f);
    if (here < 0 || std::fseek(f, 0, SEEK_END) != 0) return refuse("cannot size the file");
    const long total = std::ftell(f);
    if (total < here || (unsigned __int128)(total - here) != bytes) return refuse("the file's size is not its header's");
    if (std::fseek(f, here, SEEK_SET) != 0) return refuse("cannot read the file");
    out->top_k = (int32_t)top_k;
    out->rows = rows;
    for (const long long l : routed) out->layers.push_back((int)l);
    out->doc_starts.assign(docs.begin(), docs.end());
    try {
        out->ids.assign(routed.size(), {});
        for (auto& v : out->ids) {
            v.resize((size_t)(rows * top_k));
            if (std::fread(v.data(), sizeof(int32_t), v.size(), f) != v.size()) return refuse("the record is shorter than its header says");
            for (const int32_t id : v)
                if (id < 0 || id >= n_expert) return refuse("the record holds an expert id outside the model");
        }
    } catch (const std::exception&) {
        return refuse("not enough host memory for the record");
    }
    std::fclose(f);
    return RAD_OK;
}

""",
        guard="BC_VTP44_ROUTE_REPLAY_READ",
    ),
    _after(
        "rp-bringup", "        d.profile_ops = cfg_.profile_ops;",
        "        if (!cfg_.expert_route_replay.empty()) { /* BC_VTP44_ROUTE_REPLAY_BRINGUP */\n"
        "            if (cfg_.max_seqs != 1) {\n"
        "                RAD_ERR(\"--expert-route-replay needs --max-num-seqs 1: rows are replayed in token order\");\n"
        "                return RAD_E_INVAL;\n"
        "            }\n"
        "            if (!cfg_.expert_route_record.empty()) {\n"
        "                RAD_ERR(\"--expert-route-replay and --expert-route-record are exclusive\");\n"
        "                return RAD_E_INVAL;\n"
        "            }\n"
        "            static std::mutex route_mu;   /* one table, read for each engine and file, shared by that engine's ranks */\n"
        "            static RouteReplay route_table;\n"
        "            static const void* route_owner = nullptr;\n"
        "            static std::string route_path;\n"
        "            {\n"
        "                std::lock_guard<std::mutex> route_lk(route_mu);\n"
        "                if (route_owner != this || route_path != cfg_.expert_route_replay) {\n"
        "                    RouteReplay fresh;\n"
        "                    RAD_TRY(bc_route_replay_read(cfg_.expert_route_replay, (int)meta_.n_layers, (int)meta_.n_expert,\n"
        "                                                 meta_.name ? meta_.name : \"\", &fresh));\n"
        "                    route_table = std::move(fresh);\n"
        "                    route_owner = this;\n"
        "                    route_path = cfg_.expert_route_replay;\n"
        "                }\n"
        "            }\n"
        "            d.route_replay = &route_table;\n"
        "        }",
        guard="BC_VTP44_ROUTE_REPLAY_BRINGUP",
    ),
))

ENGINE = FilePatch(path="core/engine.cpp", language="none", description="VTP44 route replay end check", edits=(
    Edit(
        id="rp-end",
        anchor="^" + re.escape("    xstats_close();") + "$",
        mode="insert_after",
        text=r"""
    if (!cfg_.expert_route_replay.empty()) { /* BC_VTP44_ROUTE_REPLAY_END: a replay that did not take the whole record is not a measurement */
        for (auto& r : ranks_) {
            const auto& rc = r->ctx;
            if (!rc.route_replay_problem().empty()) {
                RAD_ERR("--expert-route-replay: rank %d: %s", r->index, rc.route_replay_problem().c_str());
                if (s >= 0) s = RAD_E_INVAL;
            } else if (s >= 0 && rc.route_replay_rows_used() != rc.route_replay_rows_total()) {
                RAD_ERR("--expert-route-replay: rank %d replayed %lld rows, the record has %lld", r->index,
                        (long long)rc.route_replay_rows_used(), (long long)rc.route_replay_rows_total());
                s = RAD_E_INVAL;
            } else if (s >= 0 && rc.route_replay_doc_hashes() != rc.route_replay_record_hashes()) {
                RAD_ERR("--expert-route-replay: rank %d: the run's documents are not the recorded ones (their token digests differ)", r->index);
                s = RAD_E_INVAL;
            }
        }
    }""",
        guard=re.escape("BC_VTP44_ROUTE_REPLAY_END"), expect_matches=1, max_span_lines=1, rationale=RATIONALE,
    ),
))

BLOCK = FilePatch(path="arch/common/rad_block_moe_q.h", language="none", description="VTP44 route replay in the MoE block", edits=(
    _after(
        "rp-members", "    rad_op op_router = 0, op_topk = 0, op_scatter = 0;",
        "    /* BC_VTP44_ROUTE_REPLAY_MEMBERS (rad0045): router_topk with the experts given, and its step-tall buffer of them. */\n"
        "    rad_op  op_topk_replay = 0;\n"
        "    rad_buf b_forced = 0;",
        guard="BC_VTP44_ROUTE_REPLAY_MEMBERS",
    ),
    _before(
        "rp-declare", "    /* Protected experts are sorted last (`protect`); layers without them declare no protect",
        "    /* BC_VTP44_ROUTE_REPLAY_DECLARE (rad0045): when a kernel plugin offers router_topk_replay, declare it with a persistent\n"
        "     * buffer the core fills with a replayed pass's recorded experts (written outside the op stream, so not transient). */\n"
        "    if (!c.sigmoid_router && rad_op_declared(b, \"router_topk_replay\")) {\n"
        "        b_forced = decl_b(b, nm.f(\"blk.%d.route_forced\", l), RAD_I32, {g.max_tok, c.top_k}, RAD_BUF_PERSIST);\n"
        "        if (!b_forced) return RAD_E_INVAL;\n"
        "        op_topk_replay = rw(b, RAD_OP(b, \"router_topk_replay\",\n"
        "                                   RAD_PARAMS(RAD_RANGE(\"M\", 1, c.rows), RAD_INT(\"n_expert\", c.n_expert_all),\n"
        "                                              RAD_INT(\"top_k\", c.top_k), RAD_INT(\"norm\", c.norm_topk),\n"
        "                                              RAD_STR(\"dtype\", g.dtype)),\n"
        "                                   RAD_NOWEIGHTS),\n"
        "                            {w.logits, b_forced}, {w.ids, w.ew});\n"
        "        if (op_topk_replay && !rad_op_resolved(b, op_topk_replay)) op_topk_replay = 0;\n"
        "    }\n",
        guard="BC_VTP44_ROUTE_REPLAY_DECLARE", max_span=1,
    ),
    Edit(
        id="rp-fused",
        anchor=re.escape("    const bool rfused = op_router_fused && rows <= router_fused_rows;\n"
                         "    const bool sfused = op_sigmoid_scatter && rows <= sigmoid_scatter_rows;\n"),
        mode="replace",
        text="    /* BC_VTP44_ROUTE_REPLAY_ISSUE (rad0045): a replayed pass takes this pass's recorded experts and the separate router,\n"
             "     * router_topk_replay and scatter: the fused forms select their own. A block with no replay op (or a sigmoid router)\n"
             "     * passes no buffer, which fails the step in the core while replaying. */\n"
             "    const int replay = rad_route_replay_load(c_, layer, (op_topk_replay && !c.sigmoid_router) ? b_forced : 0, r0, rows, k);\n"
             "    const bool rfused = replay == 0 && op_router_fused && rows <= router_fused_rows;\n"
             "    const bool sfused = replay == 0 && op_sigmoid_scatter && rows <= sigmoid_scatter_rows;\n",
        guard=re.escape("BC_VTP44_ROUTE_REPLAY_ISSUE"), expect_matches=1, max_span_lines=3, rationale=RATIONALE,
    ),
    Edit(
        id="rp-topk",
        anchor=re.escape("    } else if (op_topk_scatter && rows <= topk_scatter_rows) {"),
        mode="replace",
        text="    } else if (op_topk_scatter && rows <= topk_scatter_rows && replay == 0) {",
        guard=re.escape("replay == 0) {"), expect_matches=1, max_span_lines=1, rationale=RATIONALE,
    ),
    Edit(
        id="rp-issue",
        anchor=re.escape("        if (!c.sigmoid_router)\n"
                         "        RAD_ISSUE_N(c_, op_topk, rows, brow_slice(w.logits, r0, rows, ne_all),\n"
                         "                    brow_slice(w.ids, r0, rows, k), brow_slice(w.ew, r0, rows, k));\n"),
        mode="replace",
        text="        if (!c.sigmoid_router && replay > 0 && op_topk_replay)   /* BC_VTP44_ROUTE_REPLAY_TOPK */\n"
             "            RAD_ISSUE_N(c_, op_topk_replay, rows, brow_slice(w.logits, r0, rows, ne_all),\n"
             "                        brow_slice(b_forced, r0, rows, k), brow_slice(w.ids, r0, rows, k),\n"
             "                        brow_slice(w.ew, r0, rows, k));\n"
             "        else if (!c.sigmoid_router)\n"
             "        RAD_ISSUE_N(c_, op_topk, rows, brow_slice(w.logits, r0, rows, ne_all),\n"
             "                    brow_slice(w.ids, r0, rows, k), brow_slice(w.ew, r0, rows, k));\n",
        guard=re.escape("BC_VTP44_ROUTE_REPLAY_TOPK"), expect_matches=1, max_span_lines=4, rationale=RATIONALE,
    ),
))

PATCHES = (RAD_CORE, CONFIG, CONFIG_TEST, R4D_MOE, R4D_PLUGIN_H, R4D_SHAPES, R4D_ROWS,
           R_RUNTIME_H, ISSUE_CPP, ARCH_FAKES, CTX_H, CTX_CPP, BRINGUP, ENGINE, BLOCK)
