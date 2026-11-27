// Host-only workload fixture; reuse the exact upstream scheduler and fakes.
#include "sched_test.cpp"

namespace {
struct BurstResult {
    int steps = 0, decode_steps = 0, decode_rows = 0;
    int peak_decode = 0, outputs = 0;
    int64_t prefill_rows = 0;
};

BurstResult burst(int streams, int prompt, int output, int interleave, int target) {
    Fixture f;
    f.build(8192, 64, 2048, 8192);
    f.cfg.max_seqs = streams;
    f.cfg.checkpoint_interval = 4096;
    f.cfg.prefix_cache = false;
    f.cfg.n_spec = 3;
    f.cfg.prefill_interleave = interleave;
    f.cfg.prefill_interleave_share = 1.0;
    f.cfg.mixed_step_target_ms = target;
    CHECK_OK(f.start());
    std::vector<uint64_t> ids;
    for (int i = 0; i < streams; ++i) {
        auto r = f.request(prompt, output);
        // Distinct inputs: no prefix sharing can mask work.
        for (auto& tok : r->prompt) tok += i * 8192;
        ids.push_back(f.add(std::move(r)));
        CHECK(ids.back() != 0);
    }
    BurstResult out;
    constexpr int32_t draft[] = {21, 22, 23};
    for (int step = 0; step < 4096; ++step) {
        if (!f.sch.step()) break;
        const auto& plan = f.sch.plan();
        ++out.steps;
        int dec = 0, pf = 0;
        for (const auto& e : plan.e) {
            if (e.is_prefill) pf += e.n_tokens;
            else ++dec;
        }
        out.prefill_rows += pf;
        if (dec) {
            ++out.decode_steps;
            out.decode_rows += dec;
            out.peak_decode = std::max(out.peak_decode, dec);
        }
        // Synthetic cost curve in the observed ballpark; not a hardware prediction.
        const double seconds = 0.025 + pf * 0.00045;
        f.sch.note_step_cost(pf != 0, seconds);
        f.sch.note_mixed_step_cost(static_cast<int64_t>(seconds * 1e9));
        CHECK_OK(f.commit(7, 0, [](const StepEntry& e) {
            return std::min(2, e.n_spec);
        }));
        for (auto id : ids) {
            const auto* r = f.sch.find(id);
            if (r && r->state == ReqState::Running && !r->output.empty())
                CHECK_OK(f.sch.set_draft(id, draft, 3));
        }
    }
    for (auto id : ids) {
        const auto* r = f.sch.find(id);
        CHECK(r != nullptr);
        if (!r) continue;
        CHECK_EQ((int)r->state, (int)ReqState::Finished);
        CHECK_EQ((int)r->output.size(), output);
        out.outputs += r->output.size();
    }
    CHECK_EQ(f.sch.metrics().preemptions, 0);
    CHECK_EQ(out.prefill_rows, (int64_t)streams * prompt);
    printf("BC_BURST streams=%d prompt=%d output=%d interleave=%d target=%d steps=%d decode_steps=%d decode_rows=%d peak=%d outputs=%d\n",
           streams, prompt, output, interleave, target, out.steps,
           out.decode_steps, out.decode_rows, out.peak_decode, out.outputs);
    return out;
}
}

TEST(bc_burst_prompt_boundary_contract) {
    for (int prompt : {127, 128, 129, 2047, 2048, 2049, 4095, 4096, 4097, 6144})
        for (int n : {1, 4, 8}) {
            burst(n, prompt, 128, -1, 0);
            burst(n, prompt, 128, 0, 250);
        }
}

TEST(bc_burst_detects_decode_overlap_loss) {
    const auto a = burst(8, 4096, 128, -1, 0);
    const auto b = burst(8, 4096, 128, 0, 250);
    CHECK_EQ(a.outputs, b.outputs);
    CHECK(a.decode_steps > b.decode_steps);
    CHECK(b.peak_decode >= a.peak_decode);
}
