"""VTP44: `--expert-route-record FILE`, the experts each token's router chose, per routed layer.

KLD against a reference cannot rank KV caches on a routed model: any change in the arithmetic moves about 1.1% of
the expert choices (FLASHNEXT-ROUTING.md), and the KLD that follows is the same for every arm. To score a cache
alone the measured run must take the reference run's choices. This is the first half: the reference run records them.
The second half (replay) is a later patch.

- core: the option (config, help, explain, config_test value).
- ctx: on the recording rank each routed trunk layer's reported ids are copied, stream-ordered, into a per-pass staging
  buffer; at the end of the pass they are appended to a host table, in token order. Passes are recorded only when they are
  one sequence's prefill without drafting and positions continue (or restart at 0 for the next document); anything
  else marks the record incomplete with the reason. Recorded passes (the step tape) are off, as for the calibration taps.
- engine: refuses more than one sequence; removes a stale file at start; at shutdown writes the file (magic line, one
  JSON header line, int32 ids for each routed layer in turn) through a per-process temporary file renamed into place. A
  failed run, an incomplete record or a write failure fails the run.
- routed layers are those with expert weights (a dense layer of a mixed model never reports); their indices are in the
  header. A pass that reports no routing once recording has begun, a position that does not continue, or more than 2 GiB
  of ids marks the record incomplete.
"""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

RATIONALE = "VTP44: the expert route record as an engine option, beside --expert-stats"


def _after(id: str, line: str, text: str, *, guard: str) -> Edit:
    return Edit(
        id=id, anchor="^" + re.escape(line) + "$", mode="insert_after", text="\n" + text,
        guard=re.escape(guard), expect_matches=1, max_span_lines=1, rationale=RATIONALE,
    )


RAD_CORE = FilePatch(path="core/rad_core.h", language="none", description="VTP44 route record config", edits=(
    _after(
        "rr-config", "    std::string expert_stats;",
        "    /* BC_VTP44_ROUTE_RECORD_CONFIG: --expert-route-record FILE: write the experts each token chose, per routed\n"
        "     * layer, at shutdown. Empty is off. */\n"
        "    std::string expert_route_record;",
        guard="BC_VTP44_ROUTE_RECORD_CONFIG",
    ),
))

CONFIG = FilePatch(path="core/config.cpp", language="none", description="VTP44 --expert-route-record", edits=(
    Edit(
        id="rr-help",
        anchor=re.escape('    { "--expert-profile",     "FILE",   "rank the expert plane'),
        mode="insert_before",
        text='    { "--expert-route-record", "FILE",  "BigCherry VTP44: write the experts each token\'s router chose, per\\n"\n'
             '    "                                    routed layer, to FILE at shutdown (a reference run for replay)" },'
             ' /* BC_VTP44_ROUTE_RECORD_HELP */\n',
        guard=re.escape("BC_VTP44_ROUTE_RECORD_HELP"), expect_matches=1, max_span_lines=1, rationale=RATIONALE,
    ),
    _after(
        "rr-parse",
        '        else if (S("--expert-stats"))          { if (!(v = need(i))) return RAD_E_INVAL; c->expert_stats = v; }',
        '        else if (S("--expert-route-record"))   { if (!(v = need(i))) return RAD_E_INVAL; c->expert_route_record = v; }'
        ' /* BC_VTP44_ROUTE_RECORD_PARSE */',
        guard="BC_VTP44_ROUTE_RECORD_PARSE",
    ),
    _after(
        "rr-explain",
        '    { "--expert-stats",       [](const Config& c) { return or_none(c.expert_stats, "off"); } },',
        '    { "--expert-route-record", [](const Config& c) { return or_none(c.expert_route_record, "off"); } },'
        ' /* BC_VTP44_ROUTE_RECORD_EXPLAIN */',
        guard="BC_VTP44_ROUTE_RECORD_EXPLAIN",
    ),
))

CONFIG_TEST = FilePatch(path="tests/config_test.cpp", language="none", description="VTP44 option inventory", edits=(
    _after(
        "rr-config-test", '        { "--expert-stats", { "/s/experts.json" } },',
        '        { "--expert-route-record", { "/s/route.bin" } }, /* BC_VTP44_ROUTE_RECORD_CONFIG_TEST */',
        guard="BC_VTP44_ROUTE_RECORD_CONFIG_TEST",
    ),
))

CTX_H = FilePatch(path="core/runtime/ctx.h", language="none", description="VTP44 route record in the context", edits=(
    _after(
        "rr-desc", "    bool            record_passes = true;",
        "    /* BC_VTP44_ROUTE_RECORD_DESC: the largest pass in rows when this rank records the experts each token chose\n"
        "     * (rad0044); 0 is off. */\n"
        "    int64_t         route_record_rows = 0;",
        guard="BC_VTP44_ROUTE_RECORD_DESC",
    ),
    _after(
        "rr-api", "    int64_t           route_n_expert() const { return route_n_expert_; }",
        "    /* BC_VTP44_ROUTE_RECORD_API: the expert route record (rad0044). rows()[i] is the ids, top_k to a row, of every\n"
        "     * row of every recorded pass in arrival order for routed layer layers()[i]; doc_starts() the row where a sequence began (position 0). */\n"
        "    bool              route_record_on() const { return route_rec_cap_ > 0; }\n"
        "    int               route_record_top_k() const { return (int)route_rec_k_; }\n"
        "    const std::vector<std::vector<int32_t>>& route_record_rows() const { return route_rec_rows_; }\n"
        "    const std::vector<int64_t>& route_record_doc_starts() const { return route_rec_docs_; }\n"
        "    const std::vector<int>& route_record_layers() const { return route_rec_layers_; }\n"
        "    const std::vector<uint64_t>& route_record_doc_hashes() const { return route_rec_hash_; }\n"
        "    const std::string& route_record_problem() const { return route_rec_why_; }",
        guard="BC_VTP44_ROUTE_RECORD_API",
    ),
    _after(
        "rr-members", "    int32_t*               route_dev_      = nullptr;   /* card */",
        "    /* BC_VTP44_ROUTE_RECORD_MEMBERS */\n"
        "    int64_t                route_rec_cap_  = 0, route_rec_k_ = 0;\n"
        "    int32_t*               route_rec_dev_  = nullptr;   /* card: [layer][route_rec_cap_][top_k] staging */\n"
        "    std::vector<std::vector<int32_t>> route_rec_rows_;\n"
        "    std::vector<int64_t>   route_rec_docs_;\n"
        "    std::vector<char>      route_rec_got_;\n"
        "    std::vector<int>       route_rec_layers_, route_rec_at_;   /* routed trunk layers; layer -> index or -1 */\n"
        "    std::vector<uint64_t>  route_rec_hash_;   /* per document: FNV-1a 64 of its token ids */\n"
        "    int64_t                route_rec_next_pos_ = 0;\n"
        "    std::string            route_rec_why_;\n"
        "    void route_record_note(const char* why);\n"
        "    int  route_record_copy(int layer, const RadRouting* r);\n"
        "    int  route_record_drain();",
        guard="BC_VTP44_ROUTE_RECORD_MEMBERS",
    ),
))

CTX_CPP = FilePatch(path="core/runtime/ctx.cpp", language="none", description="VTP44 route record in the context", edits=(
    _after(
        "rr-init", "    record_passes_ = d.record_passes;",
        "    route_rec_cap_ = d.route_record_rows; /* BC_VTP44_ROUTE_RECORD_INIT */",
        guard="BC_VTP44_ROUTE_RECORD_INIT",
    ),
    Edit(
        id="rr-tape-off",
        anchor=re.escape("!profiling_ && !oracle_ && !taps;"),
        mode="replace",
        text="!profiling_ && !oracle_ && !taps && route_rec_cap_ == 0; /* BC_VTP44_ROUTE_RECORD_TAPE: a played pass would not re-run the record */",
        guard=re.escape("BC_VTP44_ROUTE_RECORD_TAPE"), expect_matches=1, max_span_lines=1, rationale=RATIONALE,
    ),
    Edit(
        id="rr-free",
        anchor="^(?:" + re.escape("    route_dev_ = nullptr;") + "|"
        + re.escape("    if (route_dev_)  { rad_dev_free(route_dev_, RAD_MEM_DEVICE);        route_dev_  = nullptr; }") + ")$",
        mode="insert_after",
        text="\n    if (route_rec_dev_) { rad_dev_free(route_rec_dev_, RAD_MEM_DEVICE); route_rec_dev_ = nullptr; } /* BC_VTP44_ROUTE_RECORD_FREE */",
        guard=re.escape("BC_VTP44_ROUTE_RECORD_FREE"), expect_matches=1, max_span_lines=1, rationale=RATIONALE,
    ),
    _after(
        "rr-report", "    s.reported = step;",
        "    if (route_rec_cap_ > 0) RAD_TRY(route_record_copy(layer, r)); /* BC_VTP44_ROUTE_RECORD_REPORT */",
        guard="BC_VTP44_ROUTE_RECORD_REPORT",
    ),
    Edit(
        id="rr-drain",
        anchor=re.escape("        const int rf = route_flush();\n        if (rf < 0 && step_status_ >= 0) step_status_ = rf;\n    }\n"),
        mode="insert_after",
        text="    if (route_rec_cap_ > 0) { /* BC_VTP44_ROUTE_RECORD_DRAIN */\n"
             "        const int rd = route_record_drain();\n"
             "        if (rd < 0 && step_status_ >= 0) step_status_ = rd;\n"
             "    }\n",
        guard=re.escape("BC_VTP44_ROUTE_RECORD_DRAIN"), expect_matches=1, max_span_lines=4, rationale=RATIONALE,
    ),
    Edit(
        id="rr-impl",
        anchor="^" + re.escape("int Ctx::route_flush() {") + "$",
        mode="insert_before",
        text=r"""/* BC_VTP44_ROUTE_RECORD_IMPL: the expert route record (rad0044). The first problem is kept and the record from then on is
 * not written to; the engine reports it, marks the file incomplete and fails the run. */
void Ctx::route_record_note(const char* why) {
    if (!route_rec_why_.empty()) return;
    route_rec_why_ = why;
    RAD_ERR("expert route record: %s; the record is incomplete", why);
}

/* A routed trunk layer's ids, copied on the compute stream into the pass's staging before the next layer's router
 * overwrites the shared ids buffer. Only a one-sequence prefill pass without drafting has rows that are tokens in order.
 * The routed layers are those with expert weights: a dense layer of a mixed model never reports. */
int Ctx::route_record_copy(int layer, const RadRouting* r) {
    const int64_t n_trunk = program_->meta.n_layers;
    if (layer < 0 || layer >= n_trunk) return RAD_OK;   /* an MTP head's layer: not part of the sequence's routing */
    if (!route_rec_why_.empty()) return RAD_OK;
    if (!batch_ || !r->expert_ids || r->top_k <= 0) { route_record_note("a layer reported no expert ids"); return RAD_OK; }
    if (batch_->phase != RAD_PHASE_PREFILL || batch_->n_seq != 1 || batch_->n_spec != 0 || batch_->draft_pass != 0 ||
        !batch_->positions_h) {
        route_record_note("only one-sequence prefill passes without drafting can be recorded");
        return RAD_OK;
    }
    const int64_t n = batch_->n_tok;
    if (n <= 0 || n > route_rec_cap_) { route_record_note("a pass is larger than --max-num-batched-tokens"); return RAD_OK; }
    if (!route_rec_dev_) {
        route_rec_at_.assign((size_t)n_trunk, -1);
        for (const WeightInfo& w : program_->weights) {
            const int32_t l = w.decl.group.layer;
            if (w.decl.group.expert >= 0 && l >= 0 && l < n_trunk) route_rec_at_[(size_t)l] = 0;
        }
        for (int64_t l = 0; l < n_trunk; ++l) {
            if (route_rec_at_[(size_t)l] < 0) continue;
            route_rec_at_[(size_t)l] = (int)route_rec_layers_.size();
            route_rec_layers_.push_back((int)l);
        }
        const int64_t n_r = (int64_t)route_rec_layers_.size();
        if (n_r == 0) { route_record_note("the model has no routed layers"); return RAD_OK; }
        route_rec_k_ = r->top_k;
        route_rec_dev_ = static_cast<int32_t*>(
            rad_dev_alloc(n_r * route_rec_cap_ * route_rec_k_ * (int64_t)sizeof(int32_t), RAD_MEM_DEVICE));
        if (!route_rec_dev_) { route_record_note("the staging allocation failed"); return RAD_OK; }
        route_rec_rows_.assign((size_t)n_r, {});
        route_rec_got_.assign((size_t)n_r, 0);
    } else if (r->top_k != route_rec_k_) {
        route_record_note("top_k changed between layers");
        return RAD_OK;
    }
    const int at = route_rec_at_[(size_t)layer];
    if (at < 0) { route_record_note("a layer without routed experts reported routing"); return RAD_OK; }
    if (route_rec_got_[(size_t)at]) { route_record_note("a layer reported twice in one pass"); return RAD_OK; }
    route_rec_got_[(size_t)at] = 1;
    return rad_memcpy_async(route_rec_dev_ + (int64_t)at * route_rec_cap_ * route_rec_k_, r->expert_ids,
                            n * route_rec_k_ * (int64_t)sizeof(int32_t), stream_);
}

/* The pass's staging to the host tables, once the pass has finished. A record run is a measurement, so the stream wait
 * is no cost that matters. A pass that reports nothing once recording has begun is a hole in the record. */
int Ctx::route_record_drain() {
    if (!route_rec_dev_ || !batch_) return RAD_OK;
    const size_t n_r = route_rec_got_.size();
    size_t got = 0;
    for (const char g : route_rec_got_) got += g != 0;
    if (got == 0) { route_record_note("a pass reported no routing"); return RAD_OK; }
    const int64_t n = batch_->n_tok;
    const int32_t* pos = batch_->positions_h;
    if (route_rec_why_.empty() && got != n_r) route_record_note("a routed layer did not report in a pass");
    if (route_rec_why_.empty()) {
        const int64_t p0 = pos[0];
        for (int64_t i = 1; i < n && route_rec_why_.empty(); ++i)
            if ((int64_t)pos[i] != p0 + i) route_record_note("token positions within a pass are not contiguous");
        if (route_rec_why_.empty()) {
            if (p0 == 0) {
                route_rec_docs_.push_back((int64_t)(route_rec_rows_[0].size() / (size_t)route_rec_k_));
                route_rec_hash_.push_back(14695981039346656037ull);
            } else if (p0 != route_rec_next_pos_) {
                route_record_note("token positions do not continue the previous pass");
            }
        }
        if (route_rec_why_.empty() && (!batch_->token_ids || route_rec_hash_.empty())) route_record_note("the pass has no token ids to digest");
        if (route_rec_why_.empty()) {   /* the document's identity: replay must see the same tokens */
            std::vector<int32_t> tk((size_t)n);
            RAD_TRY(rad_memcpy_async(tk.data(), batch_->token_ids, (size_t)n * sizeof(int32_t), stream_));
            RAD_TRY(rad_stream_sync(stream_));
            uint64_t& h = route_rec_hash_.back();
            for (int64_t i = 0; i < n; ++i) { h ^= (uint64_t)(uint32_t)tk[(size_t)i]; h *= 1099511628211ull; }
        }
        route_rec_next_pos_ = p0 + n;
    }
    const size_t per = (size_t)n * (size_t)route_rec_k_;
    constexpr size_t kLimit = (size_t)2 << 30;   /* bytes of ids: a long server run must not exhaust the host */
    if (route_rec_why_.empty() && (route_rec_rows_[0].size() + per) * n_r * sizeof(int32_t) > kLimit)
        route_record_note("the record reached its 2 GiB limit");
    if (route_rec_why_.empty()) {
        std::vector<size_t> at(n_r, 0);
        for (size_t l = 0; l < n_r; ++l) { at[l] = route_rec_rows_[l].size(); route_rec_rows_[l].resize(at[l] + per); }
        for (size_t l = 0; l < n_r; ++l)
            RAD_TRY(rad_memcpy_async(route_rec_rows_[l].data() + at[l],
                                     route_rec_dev_ + (int64_t)l * route_rec_cap_ * route_rec_k_,
                                     per * sizeof(int32_t), stream_));
        RAD_TRY(rad_stream_sync(stream_));
    }
    std::fill(route_rec_got_.begin(), route_rec_got_.end(), (char)0);
    return RAD_OK;
}

""",

        guard=re.escape("BC_VTP44_ROUTE_RECORD_IMPL"), expect_matches=1, max_span_lines=1, rationale=RATIONALE,
    ),
))

BRINGUP = FilePatch(path="core/engine_bringup.cpp", language="none", description="VTP44 route record bring-up", edits=(
    _after(
        "rr-bringup", "        d.profile_ops = cfg_.profile_ops;",
        "        if (!cfg_.expert_route_record.empty()) { /* BC_VTP44_ROUTE_RECORD_BRINGUP */\n"
        "            if (cfg_.max_seqs != 1) {\n"
        "                RAD_ERR(\"--expert-route-record needs --max-num-seqs 1: rows are recorded in token order\");\n"
        "                return RAD_E_INVAL;\n"
        "            }\n"
        "            if (r->index == 0) {   /* every rank routes alike: one records; a stale file must not stand for this run */\n"
        "                d.route_record_rows = cfg_.max_tok;\n"
        "                (void)std::remove(cfg_.expert_route_record.c_str());\n"
        "            }\n"
        "        }",
        guard="BC_VTP44_ROUTE_RECORD_BRINGUP",
    ),
))

ENGINE = FilePatch(path="core/engine.cpp", language="none", description="VTP44 route record file", edits=(
    Edit(
        id="rr-write",
        anchor="^" + re.escape("    xstats_close();") + "$",
        mode="insert_after",
        text=r"""
    if (!cfg_.expert_route_record.empty() && !ranks_.empty() && ranks_[0]->ctx.route_record_on()) { /* BC_VTP44_ROUTE_RECORD_WRITE */
        auto& rc = ranks_[0]->ctx;
        /* Not after a failed run or a step-loop error, where a stream may never drain. */
        const bool sane = s >= 0 && loop_status_.load(std::memory_order_acquire) >= 0 && rad_dev_set(ranks_[0]->device) >= 0 &&
                          rad_stream_sync(rc.stream()) >= 0;
        const auto& rows = rc.route_record_rows();
        const auto& routed = rc.route_record_layers();
        const int rk = rc.route_record_top_k();
        const int64_t nr = rows.empty() || rk <= 0 ? 0 : (int64_t)(rows[0].size() / (size_t)rk);
        const bool complete = sane && rc.route_record_problem().empty() && nr > 0 && routed.size() == rows.size();
        const std::string why = complete ? "" : s < 0 ? "the run failed" : !sane ? "the run did not finish cleanly" :
                                nr <= 0 && rc.route_record_problem().empty() ? "no pass was recorded" : rc.route_record_problem();
        auto js = [](const std::string& in) {   /* the header is one JSON line: keep text from breaking it */
            std::string o;
            for (const char c : in) o += (c == '"' || c == '\\' || (unsigned char)c < 0x20) ? '?' : c;
            return o;
        };
        const std::string tmp = cfg_.expert_route_record + "." + std::to_string((long long)getpid()) + ".tmp";
        FILE* rf = std::fopen(tmp.c_str(), "wb");
        bool ok = rf != nullptr;
        if (!rf) {
            RAD_ERR("--expert-route-record: cannot write %s", tmp.c_str());
        } else {
            std::string docs, lay, hashes;
            for (const int64_t d : rc.route_record_doc_starts()) docs += (docs.empty() ? "" : ",") + std::to_string((long long)d);
            for (const int l : routed) lay += (lay.empty() ? "" : ",") + std::to_string(l);
            for (const uint64_t hv : rc.route_record_doc_hashes()) {
                char hb[24];
                std::snprintf(hb, sizeof hb, "\"%016llx\"", (unsigned long long)hv);
                hashes += (hashes.empty() ? "" : ",") + std::string(hb);
            }
            ok = std::fprintf(rf, "RADROUTE1\n{\"format\":\"radiance-expert-route/1\",\"model\":\"%s\",\"arch\":\"%s\",\"layers\":%d,"
                                  "\"routed\":[%s],\"top_k\":%d,\"n_expert\":%d,\"rows\":%lld,\"tp\":%d,\"max_tok\":%lld,"
                                  "\"complete\":%s,\"why\":\"%s\",\"doc_starts\":[%s],\"doc_hashes\":[%s]}\n",
                              js(meta_.name ? meta_.name : "").c_str(), js(meta_.arch_id ? meta_.arch_id : "").c_str(),
                              (int)meta_.n_layers, lay.c_str(), rk, (int)meta_.n_expert, (long long)nr, cfg_.tp,
                              (long long)cfg_.max_tok, complete ? "true" : "false", js(why).c_str(), docs.c_str(), hashes.c_str()) > 0;
            if (complete)
                for (const auto& lr : rows) ok = ok && std::fwrite(lr.data(), sizeof(int32_t), lr.size(), rf) == lr.size();
            ok = std::fclose(rf) == 0 && ok;
            ok = ok && std::rename(tmp.c_str(), cfg_.expert_route_record.c_str()) == 0;
            if (!ok) {
                RAD_ERR("--expert-route-record: cannot finish %s", cfg_.expert_route_record.c_str());
                (void)std::remove(tmp.c_str());
            } else {
                RAD_INFO("expert route record: %lld rows x %zu routed layers x top-%d%s written to %s", (long long)nr, rows.size(),
                         rk, complete ? "" : " (INCOMPLETE)", cfg_.expert_route_record.c_str());
            }
        }
        /* A record that was asked for and is missing or incomplete is a failed run, not a log line. */
        if ((!ok || !complete) && s >= 0) s = RAD_E_IO;
    }""",

        guard=re.escape("BC_VTP44_ROUTE_RECORD_WRITE"), expect_matches=1, max_span_lines=1, rationale=RATIONALE,
    ),
))

PATCHES = (RAD_CORE, CONFIG, CONFIG_TEST, CTX_H, CTX_CPP, BRINGUP, ENGINE)
