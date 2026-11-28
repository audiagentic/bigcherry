"""Small test-only extension of the native Radiance burst simulator.

No second scheduler. Exact anchors fail closed on upstream changes. The first
screen is synchronous bursts only: no age policy, preemption or host-clock
dependent arrival. Native simulated time is used for emission metrics.
"""
from pathlib import Path


def extend_native_sim(text: str) -> str:
    start = "TEST(sim_burst_of_long_prompts) {"
    end = "/* ================================================================== --draft-suffix (SuffixDecoding as a step selector) */"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError("native simulator seam changed")
    lo, hi = text.index(start), text.index(end)
    if hi <= lo:
        raise ValueError("invalid simulator seam")
    body = text[lo:hi]

    def replace(old, new):
        nonlocal body
        if body.count(old) != 1:
            raise ValueError("native simulator anchor changed: " + old[:80])
        body = body.replace(old, new, 1)

    replace("    Fixture f;", """    const bool bc = std::getenv("BC_SIM_SCREEN") != nullptr;
    const int mixed_ms = (int)sim_env("SIM_MIXED_MS", 0);
    const int accept = (int)sim_env("SIM_ACCEPT", 2);
    if (bc) {
        REQUIRE(N >= 1 && N <= 32 && L >= 1 && OUT >= 1 && MT >= 64);
        REQUIRE(mixed_ms >= 0 && mixed_ms <= 60000 && accept >= 0 && accept <= 3);
        REQUIRE(std::isfinite(dec0) && std::isfinite(decr) && std::isfinite(pf0) && std::isfinite(pfb));
        REQUIRE(dec0 > 0 && decr >= 0 && pf0 >= 0 && pfb > 0);
        REQUIRE(sim_env("SIM_DECODERS", 0) == 0 && sim_env("SIM_SEQS", 32) >= N);
    }
    Fixture f;""")
    replace("    f.cfg.prefix_exact = true;", """    f.cfg.prefix_exact = true;
    if (bc) {
        f.cfg.prefix_cache = false;
        f.cfg.mixed_step_target_ms = mixed_ms;
    }""")
    replace("    f.sch.set_stage_rows(1025);", """    f.sch.set_stage_rows(bc ? (int)sim_env("SIM_STAGE_ROWS", 0) : 1025);""")
    replace("    double t = 0, t_pf = 0, dec_s = 0;", """    double t = 0, t_pf = 0, dec_s = 0;
    std::vector<double> last(ids.size(), -1), gaps;
    std::vector<int64_t> emitted(ids.size(), 0);
    int64_t bc_dec_steps = 0, bc_dec_rows = 0;""")
    replace("        ++steps;", """        if (bc) {
            std::printf("BC_STEP %lld %lld %lld %lld", (long long)steps, (long long)P,
                        (long long)D, (long long)std::llround(dt*1e9));
            for (const StepEntry& e : p.e)
                std::printf(" %llu:%d:%d:%d:%zu", (unsigned long long)e.seq,
                            (int)e.is_prefill, (int)e.n_tokens, (int)e.ctx_len, e.req->output.size());
            std::printf("\\n");
            bc_dec_steps += nd > 0;
            bc_dec_rows += nd;
            f.sch.note_mixed_step_cost((int64_t)std::llround(dt*1e9));
            // Like live step_finished, charge the interleave cost after commit.
        }
        ++steps;""")
    replace("""        CHECK_OK(f.commit(7, 0, [&](const StepEntry& e) { return random_accept(rng, e); }));
        f.sch.note_step_cost(P > 0, dt);""","""        CHECK_OK(f.commit(7, 0, [&](const StepEntry& e) {
            return bc ? std::min(accept, (int)e.n_spec) : random_accept(rng, e);
        }));
        f.sch.note_step_cost(P > 0, dt);""")
    replace("            const bool burst = b.first >= ids[0];", """            const bool burst = b.first >= ids[0];
            if (bc && burst && r->output.size() > b.second) {
                const size_t k = idx(b.first);
                emitted[k] += r->output.size() - b.second;
                if (last[k] >= 0) gaps.push_back(t - last[k]);
                last[k] = t;
                std::printf("BC_EMIT %llu %zu %lld\\n", (unsigned long long)b.first,
                            r->output.size(), (long long)std::llround(t*1e9));
            }""")
    replace("    if (tt.empty()) return;", """    if (bc) {
        REQUIRE_EQ(tt.size(), ids.size());
        REQUIRE_EQ(pf_tok, N*L);
        REQUIRE_EQ(f.sch.metrics().preemptions, 0);
        for (size_t k = 0; k < ids.size(); ++k) {
            CHECK(done[k] >= 0);
            CHECK_EQ(emitted[k], OUT);
        }
        REQUIRE(!tt.empty() && t > 0);
        std::sort(gaps.begin(), gaps.end());
        double p95 = gaps.empty() ? 0.0 : gaps[(gaps.size()*95-1)/100];
        double maxgap = gaps.empty() ? 0.0 : gaps.back();
        std::printf("BC_RESULT %.9f %.9f %.9f %.9f %lld %lld %lld %lld\\n",
                    t, tt.back(), p95, maxgap, (long long)steps,
                    (long long)bc_dec_steps, (long long)bc_dec_rows, (long long)mixed);
    }
    if (tt.empty()) return;""")
    return text[:lo] + body + text[hi:]


def stage_sim(tree: Path) -> None:
    path = tree / "tests/sched_test.cpp"
    path.write_text(extend_native_sim(path.read_text()))
