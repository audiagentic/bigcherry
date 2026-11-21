"""RUM09: opt-in native per-rank HIP event phase trace.

The managed serving owner admits this only for an authenticated native-v3
serving manifest; the path is an ephemeral owner-selected run-dir child.
The independent HIP event ring brackets replayed tape stretches and host-KV
waits WITHOUT turning off recorded-pass execution. Native per-op profiling is
unchanged. Without the switch there are no new events, clocks or output writes.
"""
from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch


def rep(id: str, before: str, after: str, guard: str) -> Edit:
    return Edit(
        id=id, anchor=re.escape(before), mode="replace", text=after,
        guard=re.escape(guard), expect_matches=1,
        max_span_lines=max(6, before.count("\n") + 2),
        rationale="RUM09: native sealed-owner per-rank HIP phase timing, absent by default",
    )


CTX_H = FilePatch(path="core/runtime/ctx.h", language="none",
    description="Expose optional KV phase intervals and rank-local trace file", edits=(
    rep("phase-header", "#include <atomic>\n",
        '#include <atomic>\n#include <cstdio> // BC_RUM09_PHASE_HEADER: trace FILE owned by Ctx\n',
        "BC_RUM09_PHASE_HEADER"),
    rep("phase-api", "    int run_step(const RadBatch* b);\n",
        """    int run_step(const RadBatch* b);
    // BC_RUM09_PHASE_API: no event or call into HIP when the run-directory
    // diagnostic is absent. Negative tags mark KV serial copy and DMA wait.
    size_t phase_kv_begin(int tag);
    void phase_kv_end(size_t token);
""", "BC_RUM09_PHASE_API"),
    rep("phase-slot-range",
        "    struct ProfSlot { RadEvent a = nullptr, b = nullptr; int32_t op = 0; uint8_t lane = 0; };\n",
        "    struct ProfSlot { RadEvent a = nullptr, b = nullptr; int32_t op = 0; uint8_t lane = 0; "
        "int32_t packet_from = -1, packet_to = -1; }; // BC_RUM09_SEGMENT_RANGE\n",
        "BC_RUM09_SEGMENT_RANGE"),
    rep("phase-state", "    bool                 profiling_ = false;\n",
        """    bool                 profiling_ = false;
    // BC_RUM09_PHASE_STATE: initialized only for the native serving diagnostic.
    bool                 phase_tracing_ = false; // independent of profile_ops and tape
    std::FILE*           phase_file_ = nullptr;
    int64_t              phase_wall_start_ns_ = 0;
""", "BC_RUM09_PHASE_STATE"),
))

CTX_CPP = FilePatch(path="core/runtime/ctx.cpp", language="none",
    description="Per-pass rank HIP event aggregation with no default-path events", edits=(
    rep("phase-prototype", '#include "xp_state.h"\n',
        '#include "xp_state.h"\n'
        '// BC_RUM09_PHASE_PROTO: authenticated owner-only path from the serving component.\n'
        'extern "C" const char* bc_serving_step_phase_dir();\n',
        "BC_RUM09_PHASE_PROTO"),
    rep("phase-init", "    profiling_ = d.profile_ops;\n",
        """    profiling_ = d.profile_ops;
    // BC_RUM09_PHASE_INIT: the off-by-default owner flag never disables tapes.
    phase_tracing_ = bc_serving_step_phase_dir() != nullptr;
""", "BC_RUM09_PHASE_INIT"),
    rep("phase-bind", """    if (!profiling_) return RAD_OK;

    RAD_WARN("per-op timing is on; it adds synchronisations that change the mover's slot supply, "
             "so this run is not comparable with an unprofiled one (spec §16)");
""",
        """    // BC_RUM09_PHASE_BIND: HIP pairs are allocated only for a selected trace.
    // Native per-op profiling still disables recorded passes; phase tracing does not.
    if (!profiling_ && !phase_tracing_) return RAD_OK;
    if (profiling_) {
        RAD_WARN("per-op timing is on; it adds synchronisations that change the mover's slot supply, "
                 "so this run is not comparable with an unprofiled one (spec §16)");
    } else {
        RAD_WARN("RUM09: HIP phase trace is on; end-pass synchronisation perturbs timing, "
                 "but recorded passes remain enabled");
    }
""", "BC_RUM09_PHASE_BIND"),
    rep("phase-close", "    for (auto& p : prof_ring_) {\n",
        """    // BC_RUM09_PHASE_CLOSE: a rank owns its own file, never shared across ranks.
    if (phase_file_) { std::fclose(phase_file_); phase_file_ = nullptr; }
    for (auto& p : prof_ring_) {
""", "BC_RUM09_PHASE_CLOSE"),
    rep("phase-tape-segment",
        """        case kTapePlay:
            rc = tape_play(t.dev, st.a, st.b);
            if (rc < 0) step_fail_ = fmt("a recorded pass could not be submitted: %s",
                                         rad_dev_last_error());
            break;
""",
        """        case kTapePlay: {
            // BC_RUM09_PHASE_TAPE: HIP events wrap each replayed device stretch,
            // never tape capture; no profiler flag, no change to the tape's key.
            const size_t token = phase_kv_begin(-3);
            if (token != SIZE_MAX) {
                prof_ring_[token].packet_from = st.a;
                prof_ring_[token].packet_to = st.b;
            }
            rc = tape_play(t.dev, st.a, st.b);
            phase_kv_end(token);
            if (rc < 0) step_fail_ = fmt("a recorded pass could not be submitted: %s",
                                         rad_dev_last_error());
            break;
        }
""", "BC_RUM09_PHASE_TAPE"),
    rep("phase-tape-op-map",
        """    const int bit = b->draft_pass + 8;
    if (rank_ == 0 && bit >= 0 && bit < 64 && !(tape_told_ & (1ull << bit))) {
""",
        """    // BC_RUM09_TAPE_OP_MAP: rank-local op -> packet map is sampled at
    // capture, before the immutable replay. These are op names, NOT
    // speculative duration allocations across overlapping launch queues.
    if (phase_tracing_ && b->draft_pass < 0 && b->n_draft_out > 0) {
        for (const HazOp& h : haz_ops_) {
            if (h.op == 0 || (size_t)h.op >= plans_.size() || !plans_[h.op].info) continue;
            const OpInfo& op = *plans_[h.op].info;
            const char* name = op.op.c_str();
            RAD_INFO("RUM09 tape op rank=%d step=%d pass=%d context=%d packets=[%d,%d) op=%s collective=%d",
                rank_, b->step, b->draft_pass, b->max_ctx_len, h.rec0, h.rec1,
                name, plans_[h.op].collective ? 1 : 0);
        }
    }
    const int bit = b->draft_pass + 8;
    if (rank_ == 0 && bit >= 0 && bit < 64 && !(tape_told_ & (1ull << bit))) {
""", "BC_RUM09_TAPE_OP_MAP"),
    rep("phase-run-pass-lambda", "    scratch_cur_[0] = scratch_cur_[1] = 0;\n",
        """    scratch_cur_[0] = scratch_cur_[1] = 0;
    // BC_RUM09_PHASE_PASS: a live or newly recorded pass gets one envelope;
    // tape replay has finer per-stretch event brackets above.
    const auto traced_arch = [&] {
        const size_t token = phase_kv_begin(-3);
        arch_step_(this, b);
        phase_kv_end(token);
    };
""", "BC_RUM09_PHASE_PASS"),
    rep("phase-run-unkeyed",
        "    if (!tape_on_ || staging_) { bump(pc_unkeyed_); arch_step_(this, b); return RAD_OK; }\n",
        "    if (!tape_on_ || staging_) { bump(pc_unkeyed_); traced_arch(); return RAD_OK; }\n",
        "traced_arch(); return RAD_OK"),
    rep("phase-run-key-zero",
        "    if (key == 0) { bump(pc_unkeyed_); arch_step_(this, b); return RAD_OK; }\n",
        "    if (key == 0) { /* BC_RUM09_PHASE_KEY_ZERO */ bump(pc_unkeyed_); traced_arch(); return RAD_OK; }\n",
        "BC_RUM09_PHASE_KEY_ZERO"),
    rep("phase-record-lambda", "    const uint64_t key = pass_key(b);\n",
        """    const uint64_t key = pass_key(b);
    // BC_RUM09_PHASE_RECORD: marks are outside tape_begin/tape_end. Inside
    // capture, phase_kv_begin refuses to emit markers, so the tape stays exact.
    const auto traced_record = [&](PassTape* out) {
        const size_t token = phase_kv_begin(-3);
        const int result = record_pass(b, key, out);
        phase_kv_end(token);
        return result;
    };
""", "BC_RUM09_PHASE_RECORD"),
    rep("phase-run-refused",
        """        bump(pc_refused_);
        arch_step_(this, b);
        return RAD_OK;""",
        """        bump(pc_refused_);
        traced_arch();
        return RAD_OK;""", "bump(pc_refused_);\n        traced_arch();"),
    rep("phase-run-first",
        """        bump(pc_first_);
        arch_step_(this, b);
        return RAD_OK;""",
        """        bump(pc_first_);
        traced_arch();
        return RAD_OK;""", "bump(pc_first_);\n        traced_arch();"),
    rep("phase-run-audit",
        "        if (!record_pass(b, key, &fresh)) return RAD_OK;\n",
        "        if (!traced_record(&fresh)) return RAD_OK;\n", "traced_record(&fresh)"),
    rep("phase-run-new",
        "    if (!record_pass(b, key, &nt)) return RAD_OK;\n",
        "    if (!traced_record(&nt)) return RAD_OK;\n", "traced_record(&nt)"),
    rep("phase-run-collect", "    if (profiling_) profile_collect();\n    flush_issued();\n",
        "    if (profiling_ || phase_tracing_) profile_collect();\n    flush_issued();\n",
        "profiling_ || phase_tracing_) profile_collect()"),
    rep("phase-begin", "    prof_n_      = 0;\n    kvstream_enter(this, b);\n",
        """    prof_n_      = 0;
    // BC_RUM09_PHASE_BEGIN: epoch range is used only to join the native
    // benchmark's cold/warm windows; HIP event durations stay device-local.
    phase_wall_start_ns_ = 0;
    if (phase_tracing_) {
        phase_wall_start_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
    kvstream_enter(this, b);
""", "BC_RUM09_PHASE_BEGIN"),
    rep("phase-collect", '''void Ctx::profile_collect() {
    /* The profiling sync, once at the end of the step so ordering within the step is
     * undisturbed. */
    if (prof_n_ > 0) {
        /* Lane 1 is already joined into the compute stream, so this covers every launch. */
        ++n_host_syncs_;
        if (rad_stream_sync(stream_) < 0) prof_n_ = 0;
    }
    /* Idle time before each op, per lane. A lane's first op in the step is not charged: its gap
     * is mostly this collection's own sync. */
    const ProfSlot* prev[2] = { nullptr, nullptr };
    for (size_t i = 0; i < prof_n_; ++i) {
        ProfSlot& s = prof_ring_[i];
        float ms = 0.0f;
        if (rad_event_elapsed_ms(s.a, s.b, &ms) < 0) continue;
        prof_ms_[(size_t)s.op]    += (double)ms;
        prof_calls_[(size_t)s.op] += 1;
        const ProfSlot*& p = prev[s.lane & 1];
        float gap = 0.0f;
        if (p && rad_event_elapsed_ms(p->b, s.a, &gap) >= 0 && gap > 0.0f)
            prof_wait_ms_[(size_t)s.op] += (double)gap;
        p = &s;
    }
    if (prof_overflowed_) {
        RAD_WARN("per-op timing: the step issued more than %zu ops; the tail was not timed",
                 prof_ring_.size());
        prof_overflowed_ = false;
    }
    prof_n_ = 0;
}
''',
'''// BC_RUM09_PHASE_COLLECT: bracket serial KV copies and compute-stream DMA
// waits with the same device event API the ordinary profiler already uses.
// These cannot measure the concurrent DMA queue's own engine occupancy.
size_t Ctx::phase_kv_begin(int tag) {
    if (!phase_wall_start_ns_ || rec_) return SIZE_MAX;
    if (prof_n_ >= prof_ring_.size()) {
        prof_overflowed_ = true;
        return SIZE_MAX;
    }
    const size_t slot = prof_n_++;
    ProfSlot& p = prof_ring_[slot];
    p.op = tag;
    p.lane = (uint8_t)lane_;
    p.packet_from = -1;
    p.packet_to = -1;
    if (rad_event_record(p.a, cur_stream()) < 0) {
        prof_n_ = slot;
        prof_overflowed_ = true;
        return SIZE_MAX;
    }
    return slot;
}

void Ctx::phase_kv_end(size_t token) {
    if (token == SIZE_MAX) return;
    if (rad_event_record(prof_ring_[token].b, cur_stream()) < 0)
        prof_overflowed_ = true;
}

void Ctx::profile_collect() {
    const bool tracing = phase_wall_start_ns_ > 0;
    bool valid = !prof_overflowed_;
    if (prof_n_ > 0) {
        ++n_host_syncs_;
        if (rad_stream_sync(stream_) < 0) { valid = false; prof_n_ = 0; }
    }
    const ProfSlot* prev[2] = { nullptr, nullptr };
    double compute_ms = 0, collective_ms = 0, host_kv_serial_ms = 0;
    double host_kv_wait_ms = 0, idle_wait_ms = 0;
    int n_compute = 0, n_collective = 0, n_kv_serial = 0, n_kv_wait = 0;
    // BC_RUM09_SEGMENT_COLLECT: packet boundaries map replay spans back to
    // the immutable tape, without injecting events into tape capture.
    std::string packet_spans = "[";
    for (size_t i = 0; i < prof_n_; ++i) {
        ProfSlot& s = prof_ring_[i];
        float ms = 0.0f;
        if (rad_event_elapsed_ms(s.a, s.b, &ms) < 0) { valid = false; continue; }
        if (s.op == -1) { host_kv_serial_ms += ms; ++n_kv_serial; }
        else if (s.op == -2) { host_kv_wait_ms += ms; ++n_kv_wait; }
        else if (s.op == -3) {
            compute_ms += ms; ++n_compute;
            if (packet_spans.size() > 1) packet_spans += ',';
            packet_spans += fmt(R"JSON({"from":%d,"to":%d,"ms":%.6f})JSON",
                                s.packet_from, s.packet_to, (double)ms);
        }
        else if (s.op >= 0 && (size_t)s.op < plans_.size()) {
            prof_ms_[(size_t)s.op] += (double)ms;
            prof_calls_[(size_t)s.op] += 1;
            if (plans_[(size_t)s.op].collective) {
                collective_ms += ms; ++n_collective;
            } else { compute_ms += ms; ++n_compute; }
        } else valid = false;
        const ProfSlot*& p = prev[s.lane & 1];
        float gap = 0.0f;
        if (p && rad_event_elapsed_ms(p->b, s.a, &gap) >= 0 && gap > 0.0f) {
            idle_wait_ms += (double)gap;
            if (s.op >= 0 && (size_t)s.op < prof_wait_ms_.size())
                prof_wait_ms_[(size_t)s.op] += (double)gap;
        }
        p = &s;
    }
    packet_spans += ']';
    if (prof_overflowed_) {
        RAD_WARN("per-op timing: the step issued more than %zu ops or phase marks were lost",
                 prof_ring_.size());
        valid = false;
        prof_overflowed_ = false;
    }
    if (tracing && batch_) {
        if (!phase_file_) {
            const std::string file = std::string(bc_serving_step_phase_dir()) +
                "/engine-rank-" + std::to_string(rank_) + ".jsonl";
            phase_file_ = std::fopen(file.c_str(), "wx"); // never overwrite an earlier run
        }
        if (!phase_file_) {
            step_status_ = RAD_E_IO;
            step_fail_ = "RUM09 phase trace: output file cannot be created";
        } else {
            const auto end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            // Values from distinct lanes can overlap. Compute and collective
            // are summed DEVICE busy intervals, not an asserted critical path.
            // 'null' means an event category was not observed, not zero time.
            const char* absent = "null";
            auto number = [&](double value, int count) {
                return count ? std::to_string(value) : std::string(absent);
            };
            if (std::fprintf(phase_file_,
                R"JSON({"schema_version":1,"rank":%d,"step":%d,"draft_pass":%d,"phase":%d,"n_tok":%lld,"ctx":%d,"wall_start_ns":%lld,"wall_end_ns":%lld,"valid":%s,"compute_ms":%s,"collective_ms":%s,"host_kv_serial_ms":%s,"host_kv_wait_ms":%s,"host_kv_dma_ms":null,"idle_wait_ms":%s,"n_compute":%d,"n_collective":%d,"n_kv_serial":%d,"n_kv_wait":%d,"tape_segments":%s}
)JSON",
                rank_, batch_->step, batch_->draft_pass, (int)batch_->phase,
                (long long)batch_->n_tok, (int)batch_->max_ctx_len,
                (long long)phase_wall_start_ns_, (long long)end_ns,
                valid ? "true" : "false",
                number(compute_ms, n_compute).c_str(),
                number(collective_ms, n_collective).c_str(),
                number(host_kv_serial_ms, n_kv_serial).c_str(),
                number(host_kv_wait_ms, n_kv_wait).c_str(),
                number(idle_wait_ms, prof_n_ > 1 ? 1 : 0).c_str(),
                n_compute, n_collective, n_kv_serial, n_kv_wait, packet_spans.c_str()) < 0 ||
                std::fflush(phase_file_) != 0) {
                step_status_ = RAD_E_IO;
                step_fail_ = "RUM09 phase trace: output write failed";
            }
        }
    }
    prof_n_ = 0;
    phase_wall_start_ns_ = 0;
}
''', "BC_RUM09_PHASE_COLLECT"),
))

STAGE_CPP = FilePatch(path="core/kvstream/stage.cpp", language="none",
    description="Instrument true streamed host-KV serial copy and DMA wait", edits=(
    rep("phase-serial-copy",
'''int serial_stage(Ctx* c, const StageSrc& s) {
    RadStream st = c->stream();
    for (size_t i = 0; i < s.len.size(); ++i) {
''',
'''int serial_stage(Ctx* c, const StageSrc& s) {
    // BC_RUM09_KV_SERIAL: same compute stream, so HIP events can bracket the copy.
    const size_t trace = c->phase_kv_begin(-1);
    RadStream st = c->stream();
    for (size_t i = 0; i < s.len.size(); ++i) {
''', "BC_RUM09_KV_SERIAL"),
    rep("phase-serial-error",
'''        if (rc < 0) return rc;
    }
    return 0;
}

void count_stage''',
'''        if (rc < 0) { c->phase_kv_end(trace); return rc; }
    }
    c->phase_kv_end(trace);
    return 0;
}

void count_stage''', "c->phase_kv_end(trace);"),
    rep("phase-pipe-wait",
'''        if (g_pipe.inflight == layer && g_pipe.group == group) {
            if (rad_event_wait(c->stream(), g_pipe.done) < 0) {
''',
'''        if (g_pipe.inflight == layer && g_pipe.group == group) {
            // BC_RUM09_KV_WAIT: compute stream blocked on separate DMA queue.
            const size_t trace = c->phase_kv_begin(-2);
            const int wait_rc = rad_event_wait(c->stream(), g_pipe.done);
            c->phase_kv_end(trace);
            if (wait_rc < 0) {
''', "BC_RUM09_KV_WAIT"),
    rep("phase-first-wait",
'''                    const bool ok = rad_event_wait(c->stream(), g_pipe.done) >= 0;
                    g_pipe.inflight = -1;
''',
'''                    // BC_RUM09_KV_FIRST: first layer's transfer is not pipelined.
                    const size_t trace = c->phase_kv_begin(-2);
                    const bool ok = rad_event_wait(c->stream(), g_pipe.done) >= 0;
                    c->phase_kv_end(trace);
                    g_pipe.inflight = -1;
''', "BC_RUM09_KV_FIRST"),
))

PATCHES = (CTX_H, CTX_CPP, STAGE_CPP)
