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
