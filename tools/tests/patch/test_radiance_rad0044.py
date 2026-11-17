"""VTP44: rad0044 expert route record (pinned v1.17, hardware-free mechanics).

The recorded ids are checked on the card (a recorded run's histograms equal --expert-stats'). These tests cover the
transform on the files the committed fixture holds (ctx.h, ctx.cpp, engine.cpp); the other four edits (config, bring-up,
option inventory) are applied, fail closed, by the lane build."""
from __future__ import annotations

import difflib
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
from bigcherry.patcher import apply_all  # noqa: E402

PACKAGE = REPO / "engines/radiance/patches/rad0044_expert_route_record"
PIN = REPO / "engines/radiance/tests/fixtures/19d4e8f"
FILES = ("core/runtime/ctx.h", "core/runtime/ctx.cpp", "core/engine.cpp")
DRAIN_ANCHOR = "        const int rf = route_flush();\n        if (rf < 0 && step_status_ >= 0) step_status_ = rf;\n    }\n"


def patches():
    spec = importlib.util.spec_from_file_location("rad0044", PACKAGE / "patch.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return [p for p in mod.PATCHES if p.path in FILES]


def _fixture(dst: Path) -> dict:
    for rel in FILES:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PIN / rel, dst / rel)
    return {rel: (dst / rel).read_text(encoding="utf-8") for rel in FILES}


def test_applies_adds_lines_and_is_idempotent():
    with tempfile.TemporaryDirectory(dir=REPO) as temp:
        root = Path(temp)
        before = _fixture(root)
        ps = patches()
        assert {p.path for p in ps} == set(FILES)
        first = apply_all(ps, root)
        assert all(p.ok for p in first), first
        assert all(e.status == "applied" for p in first for e in p.results), first
        after = {rel: (root / rel).read_text(encoding="utf-8") for rel in FILES}
        # Only the tape condition is changed in place; everything else is additions.
        for rel in FILES:
            removed = [l for l in difflib.unified_diff(before[rel].splitlines(), after[rel].splitlines(), n=0)
                       if l.startswith("-") and not l.startswith("---")]
            assert len(removed) <= 1, (rel, removed)
            if removed:
                assert "tape_on_ = record_passes_" in removed[0] and rel == "core/runtime/ctx.cpp"
        ctx = after["core/runtime/ctx.cpp"]
        assert "&& !taps && route_rec_cap_ == 0;" in ctx
        # Copy at the report, drain after the pass's flush, free with the other staging.
        report = ctx.index("route_record_copy(layer, r)")
        assert ctx.index("s.reported = step;") < report
        assert ctx.index("route_record_drain();", ctx.index("step_status_ = rf;")) > 0
        assert "rad_dev_free(route_rec_dev_" in ctx
        # The first problem is kept; only one-sequence prefill without drafting is recorded.
        assert "batch_->phase != RAD_PHASE_PREFILL || batch_->n_seq != 1 || batch_->n_spec != 0 || batch_->draft_pass != 0" in ctx
        engine = after["core/engine.cpp"]
        assert engine.index("xstats_close();") < engine.index("BC_VTP44_ROUTE_RECORD_WRITE")
        assert 'std::rename(tmp.c_str(), cfg_.expert_route_record.c_str())' in engine
        assert "RADROUTE1" in engine

        snapshot = after
        second = apply_all(ps, root)
        assert all(p.ok for p in second), second
        assert all(e.status == "already-applied" for p in second for e in p.results)
        assert {rel: (root / rel).read_text(encoding="utf-8") for rel in FILES} == snapshot


def test_anchor_drift_and_duplicate_fail_closed():
    ps = patches()
    for how in ("drift", "duplicate"):
        with tempfile.TemporaryDirectory(dir=REPO) as temp:
            root = Path(temp)
            src = _fixture(root)["core/runtime/ctx.cpp"]
            assert src.count(DRAIN_ANCHOR) == 1
            bad = src.replace(DRAIN_ANCHOR, DRAIN_ANCHOR.replace("rf < 0", "rf != 0") if how == "drift" else DRAIN_ANCHOR * 2)
            (root / "core/runtime/ctx.cpp").write_text(bad, encoding="utf-8")
            result = apply_all(ps, root)
            assert any(not p.ok for p in result), (how, result)
            assert (root / "core/runtime/ctx.cpp").read_text(encoding="utf-8") == bad
