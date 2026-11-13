"""RHP01 independent CPU coverage oracle for the managed Radiance evidence package.

The oracle supplies issue/phase/multiplicity counts independently of the tracker;
it never fabricates device completion or treats the host-bank ring as a GPU counter.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
HEADER = ROOT / "engines/radiance/patches/rad0010_routing_evidence"

CPP = r"""
int rad_event_query(int value) { return value == 1 ? 1 : 0; }
#include "bc_moe_remote_evidence.h"
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

using Tracker = BcMoeRemoteEvidence;
struct Row {
    int layer, bank, step, kind, passes;
};
struct Expected {
    uint64_t issued = 0, produced = 0, passes = 0;
};

void complete(Tracker& t, const std::vector<Row>& rows) {
    for (const Row& row : rows) {
        // The existing GPU event is still pending until this explicit oracle
        // transition; no unlanded row is credited.
        assert(!t.landed(row.layer, row.bank, row.step, false));
        assert(t.landed(row.layer, row.bank, row.step, true));
        assert(t.credit(row.layer, row.bank, row.step, row.passes));
    }
}

void replay(int chunk) {
    constexpr int context = 32768;
    const int steps = (context + chunk - 1) / chunk;
    Tracker t;
    t.reset(3, 32);
    std::array<Expected, Tracker::Phases> expected{};
    std::vector<Row> previous;
    for (int step = 0; step < steps; ++step) {
        std::vector<Row> current;
        const int bank = step % t.banks;
        const int kind = step % 3 == 0 ? Tracker::Prefill :
                         step % 3 == 1 ? Tracker::Decode : Tracker::Mixed;
        for (int layer = 0; layer < 3; ++layer) {
            const int phase = layer == 2 ? Tracker::Draft : kind;
            const int passes = layer == 2 ? 3 : 1; // three same-step MTP passes
            current.push_back({layer, bank, step, phase, passes});
            expected[phase].issued++;
            expected[phase].produced += passes;
            expected[phase].passes += passes;
            for (int pass = 0; pass < passes; ++pass)
                t.issue(layer, bank, step, phase);
        }
        t.flush(bank, 0, 2, step);
        complete(t, previous); // host one step ahead of the completion event
        previous = std::move(current);
    }
    complete(t, previous);
    uint64_t issued = 0, credited = 0;
    for (int kind = 0; kind < Tracker::Phases; ++kind) {
        const auto& actual = t.count[kind];
        const Expected& oracle = expected[kind];
        assert(actual.issued == oracle.issued);
        assert(actual.produced == oracle.produced);
        assert(actual.produced_passes == oracle.passes);
        assert(actual.credited == oracle.issued);
        assert(actual.credited_passes == oracle.passes);
        assert(actual.overwritten == 0);
        assert(actual.lost_passes == 0);
        issued += oracle.issued;
        credited += actual.credited;
    }
    assert(issued == uint64_t(steps) * 3);
    assert(t.outstanding == 0 && credited == issued);
    assert(Tracker::phase(64, 64, 0) == Tracker::Decode);
    assert(Tracker::phase(64, 32, 0) == Tracker::Mixed);
    assert(Tracker::phase(64, 64, 1) == Tracker::Draft);
    t.print(0, 0);
}

int main() {
    for (int chunk : {2048, 4096, 8192}) replay(chunk);

    // Deliberate ring wrap without any completed consumer credit. Old source
    // steps must be counted as overwritten; recycled events cannot revive them.
    Tracker wrapped;
    wrapped.reset(1, 4);
    for (int step = 0; step < 10; ++step) {
        wrapped.issue(0, step % 4, step, Tracker::Prefill);
        wrapped.flush(step % 4, 0, 0, step);
    }
    assert(wrapped.count[Tracker::Prefill].issued == 10);
    assert(wrapped.count[Tracker::Prefill].overwritten == 6);
    assert(wrapped.count[Tracker::Prefill].lost_passes == 6);
    assert(wrapped.outstanding == 4);
    assert(!wrapped.landed(0, 0, 0, true));
    for (int step = 6; step < 10; ++step) {
        assert(wrapped.landed(0, step % 4, step, true));
        assert(wrapped.credit(0, step % 4, step, 1));
    }
    assert(wrapped.outstanding == 0);
    assert(wrapped.count[Tracker::Prefill].credited == 4);
    wrapped.print(0, 0);

    Tracker late;
    late.reset(1, 4);
    late.issue(0, 0, 15, Tracker::Mixed);
    late.flush(0, 0, 0, 15);
    assert(!late.landed(0, 0, 15, false));
    assert(late.count[Tracker::Mixed].completed == 0);
    assert(late.outstanding == 1);
    late.issue(0, 1, 16, Tracker::Decode);
    late.flush(1, 0, 0, 16);
    late.print(0, 0); // pending and in-flight at shutdown, no false credit
    return 0;
}
"""


class TestRoutingEvidenceSyntheticOracle(unittest.TestCase):
    def test_chunks_mtp_ring_wrap_and_late_shutdown(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "oracle.cpp"
            binary = Path(temp) / "oracle"
            source.write_text(CPP)
            compile_result = subprocess.run(
                ["g++", "-std=c++20", "-O0", "-I", str(HEADER),
                 "-o", str(binary), str(source)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run = subprocess.run(
                [str(binary)], capture_output=True, text=True,
                env={**os.environ, "RAD_ROUTING_EVIDENCE": "1"}, check=False,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            phase_lines = [
                dict(re.findall(r"(\w+)=([^\s]+)", line))
                for line in run.stderr.splitlines()
                if line.startswith("RHP01 rank=") and " phase=" in line
            ]
            self.assertEqual(len(phase_lines), 5 * 4, run.stderr)
            for run_index in range(3):
                for record in phase_lines[run_index * 4:run_index * 4 + 4]:
                    self.assertEqual(record["issued"], record["credited"])
                    self.assertEqual(record["overwritten"], "0")
                    self.assertEqual(record["pending"], "0")
                    self.assertEqual(record["produced_passes"], record["credited_passes"])
            ring = phase_lines[3 * 4]
            self.assertEqual(ring["issued"], "10")
            self.assertEqual(ring["credited"], "4")
            self.assertEqual(ring["overwritten"], "6")
            self.assertEqual(ring["pending"], "0")
            late = next(row for row in phase_lines[4 * 4:5 * 4] if row["phase"] == "mixed")
            self.assertEqual(late["issued"], "1")
            self.assertEqual(late["credited"], "0")
            self.assertEqual(late["pending"], "1")
            self.assertEqual(late["in_flight"], "1")
            self.assertEqual(late["oldest_uncredited_steps"], "1")
            self.assertEqual(late["oldest_uncredited_source_step"], "15")



if __name__ == "__main__":
    unittest.main()


# A second, independently maintained generation ledger. The expected fates do
# not read tracker slots or derive counters from its printed totals.
CPP_RANKED = r"""
int rad_event_query(int event) { return event == 1 ? 1 : 0; }
#include "bc_moe_remote_evidence.h"
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

using T = BcMoeRemoteEvidence;
struct Hist {
    int rank, layer, step, bank, phase, passes;
    uint64_t generation;
    bool flushed = false, completed = false, credited = false, lost = false;
};
struct Ledger {
    int rank, banks;
    uint64_t next_generation = 0;
    std::vector<Hist> history;
    std::map<std::pair<int,int>, size_t> current;
    std::map<int,int> last_credit;

    Hist* at(int layer, int bank) {
        auto it = current.find({layer,bank});
        return it == current.end() ? nullptr : &history[it->second];
    }
    void issue(int layer, int step, int phase) {
        const int bank = step % banks;
        Hist* old = at(layer, bank);
        if (old && old->step == step && !old->lost && !old->credited) {
            assert(old->phase == phase);
            ++old->passes;
            old->flushed = old->completed = false;
            return;
        }
        if (old && !old->lost && !old->credited) old->lost = true;
        history.push_back({rank,layer,step,bank,phase,1,++next_generation});
        current[{layer,bank}] = history.size() - 1;
    }
    void flush(int bank, int lo, int hi, int step) {
        for (int layer = lo; layer <= hi; ++layer) {
            Hist* row = at(layer,bank);
            if (!row) continue;
            if (row->step != step) {
                if (!row->credited) row->lost = true;
            } else if (!row->lost && !row->credited) row->flushed = true;
        }
    }
    bool landed(int layer, int step, bool ready) {
        if (!ready) return false;
        Hist* row = at(layer,step % banks);
        if (!row || row->step != step || !row->flushed || row->lost || row->credited)
            return false;
        row->completed = true;
        for (const Hist& prior : history)
            if (prior.layer == layer && prior.step < step &&
                !prior.credited && !prior.lost) return false;
        return true;
    }
    bool credit(int layer, int step, int passes) {
        Hist* row = at(layer,step % banks);
        if (!row || row->step != step || !row->completed || row->lost ||
            row->credited || row->passes != passes) return false;
        auto it = last_credit.find(layer);
        if (it != last_credit.end() && it->second >= step) return false;
        last_credit[layer] = step;
        row->credited = true;
        return true;
    }
    void verify(const T& t) const {
        std::array<T::Counts, T::Phases> expected{};
        uint64_t pending = 0;
        std::set<std::tuple<int,int,int,uint64_t>> distinct;
        for (const Hist& row : history) {
            assert(distinct.insert({row.rank,row.layer,row.step,row.generation}).second);
            T::Counts& c = expected[row.phase];
            ++c.issued;
            c.produced += row.passes;
            c.produced_passes += row.passes;
            c.completed += row.completed;
            c.credited += row.credited;
            c.overwritten += row.lost;
            c.credited_passes += row.credited ? row.passes : 0;
            c.lost_passes += row.lost ? row.passes : 0;
            pending += !row.credited && !row.lost;
        }
        assert(t.outstanding == pending);
        for (int phase = 0; phase < T::Phases; ++phase) {
            const auto& got = t.count[phase];
            const auto& want = expected[phase];
            assert(got.issued == want.issued);
            assert(got.produced == want.produced);
            assert(got.completed == want.completed);
            assert(got.credited == want.credited);
            assert(got.overwritten == want.overwritten);
            assert(got.produced_passes == want.produced_passes);
            assert(got.credited_passes == want.credited_passes);
            assert(got.lost_passes == want.lost_passes);
            uint64_t phase_pending = 0, pass_pending = 0;
            for (const Hist& row : history)
                if (row.phase == phase && !row.credited && !row.lost) {
                    ++phase_pending;
                    pass_pending += row.passes;
                }
            assert(want.issued == want.credited + want.overwritten + phase_pending);
            assert(want.produced_passes ==
                   want.credited_passes + want.lost_passes + pass_pending);
        }
    }
};
struct Harness {
    T t;
    Ledger ledger;
    Harness(int rank, int layers, int banks) : ledger{rank,banks} {
        t.reset(layers,banks);
    }
    void verify() const { ledger.verify(t); }
    void issue(int layer, int step, int phase, int passes) {
        for (int pass = 0; pass < passes; ++pass) {
            ledger.issue(layer,step,phase);
            t.issue(layer,step % t.banks,step,phase);
        }
        verify();
    }
    void flush(int lo, int hi, int step) {
        ledger.flush(step % t.banks,lo,hi,step);
        t.flush(step % t.banks,lo,hi,step);
        verify();
    }
    bool landed(int layer, int step, bool ready) {
        const bool expected = ledger.landed(layer,step,ready);
        const bool actual = t.landed(layer,step % t.banks,step,ready);
        assert(actual == expected);
        verify();
        return actual;
    }
    bool credit(int layer, int step, int passes) {
        const bool expected = ledger.credit(layer,step,passes);
        const bool actual = t.credit(layer,step % t.banks,step,passes);
        assert(actual == expected);
        verify();
        return actual;
    }
};

int main() {
    // Independent rank-local ledgers: 45 issuing steps outrun a real 32-bank
    // ring. A stale ready event from the old bank generation never credits.
    Harness head(0,1,32);
    for (int step = 0; step < 45; ++step) {
        head.issue(0,step,T::Prefill,1);
        head.flush(0,0,step);
    }
    assert(head.t.count[T::Prefill].overwritten == 13);
    assert(head.t.outstanding == 32 && head.t.high_water == 32);
    assert(!head.landed(0,0,true)); // old generation despite a stale ready event
    assert(!head.credit(0,0,1));
    // The recycled bank has been re-recorded, but its own event is not
    // complete yet; neither stale readiness nor an early credit may pass.
    assert(!head.landed(0,32,false));
    assert(!head.credit(0,32,1));
    for (int step = 13; step < 45; ++step) {
        assert(head.landed(0,step,true));
        assert(head.credit(0,step,1));
    }
    head.verify();
    assert(head.t.count[T::Prefill].credited == 32);
    head.t.print(0,0);

    // Rank 1 gets its own source steps. A later ready event cannot pass an
    // earlier, delayed event; the completed count can lead credited count.
    Harness helper(1,3,4);
    helper.issue(1,7,T::Decode,1);
    helper.flush(1,1,7);
    helper.issue(1,8,T::Mixed,1);
    helper.flush(1,1,8);
    assert(!helper.landed(1,8,true));
    assert(helper.t.count[T::Mixed].completed == 1);
    assert(!helper.landed(1,7,false));
    assert(helper.landed(1,7,true) && helper.credit(1,7,1));
    assert(helper.landed(1,8,true) && helper.credit(1,8,1));
    assert(!helper.credit(1,8,1)); // duplicate, no second heat weight
    assert(helper.t.duplicate_rejected > 0);
    helper.verify();

    // DMA [lo,hi] overwrites a gap layer which did not report in the new
    // generation. Its two passes count as lost, not pending or credited.
    helper.issue(1,12,T::Prefill,2);
    helper.flush(1,1,12);
    helper.issue(0,16,T::Decode,1);
    helper.issue(2,16,T::Decode,4);
    helper.flush(0,2,16);
    assert(!helper.landed(1,12,true));
    assert(helper.t.last_loss_layer == 1);
    assert(helper.t.count[T::Prefill].lost_passes == 2);
    assert(helper.landed(0,16,true) && helper.credit(0,16,1));
    assert(helper.landed(2,16,true) && helper.credit(2,16,4));

    // Same-step draft MTP multiplicity is 1/2/4 raw producer calls but
    // exactly one credit generation and one weighted HeatEngine observation.
    Harness mtp(2,2,4);
    for (int step = 0; step < 3; ++step) {
        const int passes = 1 << step;
        mtp.issue(0,step,T::Draft,passes);
        mtp.issue(1,step,T::Mixed,1);
        mtp.flush(0,1,step);
        assert(!mtp.landed(0,step,false)); // unfinished DMA
        assert(mtp.landed(0,step,true));
        assert(mtp.landed(1,step,true));
        assert(mtp.credit(0,step,passes));
        assert(mtp.credit(1,step,1));
    }
    assert(mtp.t.count[T::Draft].issued == 3);
    assert(mtp.t.count[T::Draft].produced == 7);
    assert(mtp.t.count[T::Draft].credited_passes == 7);
    mtp.verify();

    // Finite simulated teardown: a late bank is pending (not overwritten);
    // completion does not credit it until a bounded follow-up reader pass.
    Harness tail(3,1,4);
    tail.issue(0,21,T::Decode,4);
    tail.flush(0,0,21);
    assert(!tail.landed(0,21,false));
    assert(tail.t.outstanding == 1);
    tail.t.print(3,0);
    assert(tail.landed(0,21,true));
    assert(tail.credit(0,21,4));
    assert(tail.t.outstanding == 0);
    tail.verify();

    // Counters and same-slot report multiplicity saturate, not wrap.
    T saturated;
    saturated.reset(1,4);
    saturated.count[T::Draft].issued = std::numeric_limits<uint64_t>::max();
    saturated.issue(0,0,0,T::Draft);
    assert(saturated.count[T::Draft].issued == std::numeric_limits<uint64_t>::max());
    auto* slot = saturated.slot(0,0);
    assert(slot);
    slot->passes = std::numeric_limits<int>::max();
    saturated.issue(0,0,0,T::Draft);
    assert(slot->passes == std::numeric_limits<int>::max());
    assert(saturated.overflow >= 2);
    saturated.reset(1,4);
    assert(saturated.outstanding == 0 && saturated.overflow == 0);
    assert(saturated.count[T::Draft].issued == 0);
    assert(!saturated.landed(0,0,0,true)); // old event on restarted context
    return 0;
}
"""


def test_ranked_generation_conservation_adversarial():
    with tempfile.TemporaryDirectory() as td:
        source = Path(td) / "ranked.cpp"
        binary = Path(td) / "ranked"
        source.write_text(CPP_RANKED)
        build = subprocess.run(
            ["g++", "-std=c++20", "-O0", "-I", str(HEADER),
             "-o", str(binary), str(source)],
            capture_output=True, text=True, check=False,
        )
        assert build.returncode == 0, build.stderr
        run = subprocess.run(
            [str(binary)], capture_output=True, text=True,
            env={**os.environ, "RAD_ROUTING_EVIDENCE": "1"}, check=False,
            timeout=10,
        )
        assert run.returncode == 0, run.stderr
        lines = [
            dict(re.findall(r"(\w+)=([^\s]+)", line))
            for line in run.stderr.splitlines()
            if line.startswith("RHP01 rank=") and " phase=" in line
        ]
        assert len(lines) == 8, run.stderr
        head = next(row for row in lines if row["rank"] == "0" and row["phase"] == "prefill")
        assert (head["issued"], head["credited"], head["overwritten"], head["pending"]) == (
            "45", "32", "13", "0"
        )
        late = next(row for row in lines if row["rank"] == "3" and row["phase"] == "decode")
        assert (late["issued"], late["credited"], late["pending"], late["in_flight"]) == (
            "1", "0", "1", "1"
        )
        assert late["produced_passes"] == "4"
