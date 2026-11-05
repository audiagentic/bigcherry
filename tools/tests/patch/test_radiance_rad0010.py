"""Card-free RHP01 instrumentation tests: pinned Radiance patching and C++ lifecycle mutations."""
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
from bigcherry.patcher import apply_all  # noqa: E402

PACKAGE = REPO / "engines/radiance/patches/rad0010_routing_evidence"
PIN = REPO / "engines/radiance/tests/fixtures/769b4c6"

# Exact git blobs of the v1.16.0 release (769b4c64), not a synthetic copy.
BEFORE = {
    "core/runtime/ctx.cpp": "fa8f9d3",
    "core/engine.cpp": "1178c60",
}
AFTER = {
    "core/runtime/ctx.cpp": "32ec456",
    "core/engine.cpp": "e579ce6",
    "core/runtime/bc_moe_remote_evidence.h": "47f1e86",
}

def blob(b: bytes) -> str:
    return hashlib.sha1(f"blob {len(b)}\0".encode() + b).hexdigest()[:7]

def patches():
    spec = importlib.util.spec_from_file_location("rad0010", PACKAGE / "patch.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.PATCHES

def test_package_pinned_apply_idempotent_and_thin():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for rel, expected in BEFORE.items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PIN / rel, target)
            observed = blob(target.read_bytes())
            assert not expected or observed == expected, (rel, observed, expected)
        ps = patches()
        assert {p.path for p in ps} == set(AFTER)
        first = apply_all(list(ps), root)
        assert all(r.ok for r in first), first
        assert all(e.status == "applied" for r in first for e in r.results)
        outputs = {rel: blob((root / rel).read_bytes()) for rel in AFTER}
        assert all(not expected or outputs[rel] == expected for rel, expected in AFTER.items()), outputs
        print("RHP01_PIN_BLOBS", {rel: blob((PIN / rel).read_bytes()) for rel in BEFORE}, outputs)
        originals = {rel: (root / rel).read_bytes() for rel in AFTER}
        second = apply_all(list(ps), root)
        assert all(r.ok for r in second), second
        assert all(e.status == "already-applied" for r in second for e in r.results)
        assert all((root / rel).read_bytes() == original for rel, original in originals.items())
        # The code-line-only thin-call rule; never put policy in an upstream body.
        for rel in BEFORE:
            import difflib
            before = (PIN / rel).read_text().splitlines()
            after = (root / rel).read_text().splitlines()
            added = [l[1:].strip() for l in difflib.unified_diff(before, after, n=0)
                     if l.startswith("+") and not l.startswith("+++")]
            assert added, rel
            assert all(re.search(r"bc_moe_remote|bc_remote|BcMoeRemote", l)
                       for l in added if l and not l.startswith(("/*", "*"))), (rel, added)
        # Fail-closed: exact source code line drift must not leave a half-patched file.
        for rel, before_string, after_string in [
            ("core/runtime/ctx.cpp", "const bool again = s.step[bank] == step;",
             "const bool again = s.step[bank] != step;"),
            ("core/engine.cpp", "XsTally* const xs = xstats_ ? xstats_->tally(r.index) : nullptr;",
             "XsTally* const xs = xstats_ ? xstats_->tally(r.index) : 0;"),
        ]:
            source = (PIN / rel).read_text()
            assert before_string in source
            with tempfile.TemporaryDirectory() as other:
                temp = Path(other)
                for f in BEFORE:
                    dest = temp / f
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes((PIN / f).read_bytes())
                (temp / rel).write_text(source.replace(before_string, after_string))
                result = apply_all(list(ps), temp)
                assert not all(r.ok for r in result), (rel, result)

CPP = r"""
int rad_event_query(int event) { return event == 1 ? 1 : 0; }
#include "bc_moe_remote_evidence.h"
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>
struct StaticCtx {
    int credited = 0;
    int n_route_layers() const { return 1; }
    int next_histogram(int, int after, int before) const {
        return after < 2 && 2 < before ? 0 : -1;
    }
    int last_histogram_step(int, int) const { return 2; }
    const int32_t* last_histogram(int, int) const { static const int32_t h[2] = {3, 4}; return h; }
    void mark_credited(int, int) { ++credited; }
};
int main() {
    using T = BcMoeRemoteEvidence;
    T t;
    t.reset(3, 4);
    assert(t.layers == 3 && t.banks == 4);
    assert(T::phase(5,0,0) == T::Prefill);
    assert(T::phase(5,5,0) == T::Decode);
    assert(T::phase(5,3,0) == T::Mixed);
    assert(T::phase(5,5,-1) == T::Draft);
    t.issue(0,0,0,T::Prefill);
    t.issue(0,0,0,T::Prefill);
    assert(t.count[T::Prefill].issued == 1 && t.count[T::Prefill].produced_passes == 2);
    assert(!t.landed(0,0,0,true)); // no recorded event / old complete event
    t.flush(0,0,0,0);
    assert(!t.landed(0,0,0,false)); // late DMA event
    assert(t.landed(0,0,0,true));
    assert(t.credit(0,0,0,2));
    assert(!t.credit(0,0,0,2));
    assert(!t.landed(0,0,0,true));
    assert(t.count[T::Prefill].credited == 1 && t.count[T::Prefill].credited_passes == 2);
    assert(t.duplicate_rejected == 1 && t.stale_event_rejected >= 2);
    // Completed out of order: do not credit later step before earlier landed.
    t.issue(1,0,0,T::Prefill); t.flush(0,1,1,0);
    t.issue(1,1,1,T::Decode); t.flush(1,1,1,1);
    assert(!t.landed(1,1,1,true));
    assert(t.count[T::Decode].completed == 1);
    assert(t.landed(1,0,0,true) && t.credit(1,0,0,1));
    assert(t.landed(1,1,1,true) && t.credit(1,1,1,1));
    assert(t.count[T::Decode].credited == 1);
    // A recycled ring slot must never credit a former generation.
    t.issue(2,0,0,T::Mixed); t.flush(0,2,2,0);
    t.issue(2,0,4,T::Decode); t.flush(0,2,2,4);
    assert(t.count[T::Mixed].overwritten == 1);
    assert(!t.landed(2,0,0,true));
    assert(t.landed(2,0,4,true));
    assert(!t.credit(2,0,0,1));
    assert(t.credit(2,0,4,1));
    // Contiguous DMA overwrites an old layer even if that layer never re-reported.
    t.issue(0,0,8,T::Prefill); t.flush(0,0,0,8);
    t.issue(2,0,12,T::Decode); t.flush(0,0,2,12);
    assert(t.count[T::Prefill].overwritten == 1);
    assert(!t.landed(0,0,8,true));
    // Forged completed bank cannot be consumed without readiness.
    t.issue(0,1,13,T::Decode);
    assert(!t.credit(0,1,13,1));
    assert(!t.landed(0,1,13,true));
    t.flush(1,0,0,13);
    assert(t.landed(0,1,13,true));
    assert(t.credit(0,1,13,1));
    // Last credited must not move backwards even if an event returns ready.
    t.issue(0,2,12,T::Decode); t.flush(2,0,0,12);
    assert(t.landed(0,2,12,true));
    assert(!t.credit(0,2,12,1));
    assert(t.order_rejected > 0);
    // Static/deterministic/no movable units still drain the same event-backed reader.
    StaticCtx ctx; std::vector<int> seen;
    assert(!bc_moe_remote_evidence_static_drain(ctx, seen, 3, false));
    assert(ctx.credited == 0);
    assert(bc_moe_remote_evidence_static_drain(ctx, seen, 3, true));
    assert(ctx.credited == 1 && seen[0] == 2);
    assert(bc_moe_remote_evidence_static_drain(ctx, seen, 3, true) && ctx.credited == 1);
    // Counters saturate instead of wrapping into false low counts.
    t.count[T::Draft].issued = std::numeric_limits<uint64_t>::max();
    t.issue(2,1,19,T::Draft);
    assert(t.count[T::Draft].issued == std::numeric_limits<uint64_t>::max());
    assert(t.overflow > 0);
    // Restart/reopen invalidates every earlier generation and duplicate state.
    t.reset(2, 7);
    assert(t.outstanding == 0 && t.count[T::Draft].issued == 0 && t.banks == 7);
    t.issue(0,0,0,T::Prefill); t.flush(0,0,0,0);
    assert(t.landed(0,0,0,true) && t.credit(0,0,0,1));
    StaticCtx owner;
    bc_moe_remote_evidence_open(&owner, 2, 7);
    bc_moe_remote_evidence_issue(&owner, 0, 0, 0, 3, 0, 0);
    bc_moe_remote_evidence_flush(&owner, 0, 0, 0, 0);
    bc_moe_remote_evidence_issue(&owner, 0, 2, 2, 3, 0, 0);
    bc_moe_remote_evidence_flush(&owner, 2, 0, 0, 2);
    bc_moe_remote_evidence_issue(&owner, 1, 1, 1, 3, 0, 0);
    int events[7] = {1, 1, 2, 0, 0, 0, 0};
    bc_moe_remote_evidence_exit(&owner, 0, 0, events);
    auto* observed = bc_moe_remote_evidence_get(&owner);
    assert(observed && observed->count[T::Prefill].completed == 1);
    assert(observed->count[T::Prefill].credited == 0);
    assert(observed->outstanding == 3);
    bc_moe_remote_evidence_close(&owner);
    assert(bc_moe_remote_evidence_get(&owner) == nullptr);
    return 0;
}
"""

MUTATIONS = [
    ("event-ready", "if (!event_ready) return false;", "if (false) return false;"),
    ("flush-issued", "if (!s || s->step != step || !s->flushed || s->dropped || s->credited)", "if (!s || s->step != step || false || s->dropped || s->credited)"),
    ("recycled-generation", "s->step != step || s->credited", "false || s->credited"),
    ("duplicate-credit", "s->credited || s->dropped", "false || s->dropped"),
    ("credit-readiness", "!s->completed || s->passes", "false || s->passes"),
    ("multi-pass", "if (s->step == step && !s->dropped && !s->credited)", "if (false)"),
    ("dma-clobber", "if (s->step != step) {", "if (false) {"),
    ("original-order", "earlier->step < step &&", "false &&"),
    ("last-credit-step", "last_credited[(size_t)layer] >= step", "false"),
    ("counter-overflow", "std::numeric_limits<uint64_t>::max() - v < n", "false"),
    ("exit-event-query", "rad_event_query(events[bank]) == 1", "true"),
    ("exit-not-flushed", "!s->flushed || s->dropped || s->completed", "false || s->dropped || s->completed"),
    ("exit-ready-check", "if (ready[(size_t)bank]) {", "if (true) {"),
    ("context-close", "bc_moe_remote_evidence_registry().erase(ctx);", "(void)ctx;"),
]

def compile_and_run(tmp: Path, header: str, name: str) -> int:
    (tmp / "bc_moe_remote_evidence.h").write_text(header)
    (tmp / "evidence_test.cpp").write_text(CPP)
    build = subprocess.run(["g++", "-std=c++20", "-O0", "-I", str(tmp), "-o",
                            str(tmp / "test"), str(tmp / "evidence_test.cpp")],
                           capture_output=True, text=True)
    assert build.returncode == 0, f"{name} compile: {build.stderr}"
    return subprocess.run([str(tmp / "test")], env={**os.environ, "RAD_ROUTING_EVIDENCE": "1"},
                          capture_output=True).returncode

def test_card_free_lifecycle_and_mutation_guards():
    header = (PACKAGE / "bc_moe_remote_evidence.h").read_text()
    with tempfile.TemporaryDirectory() as d:
        assert compile_and_run(Path(d), header, "baseline") == 0
        for name, old, new in MUTATIONS:
            assert header.count(old) == 1, name
            altered = header.replace(old, new)
            assert compile_and_run(Path(d), altered, name) != 0, f"{name}: survived mutation"
