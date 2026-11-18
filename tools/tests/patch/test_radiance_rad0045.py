"""VTP44: rad0045 expert route replay (pinned v1.17, hardware-free mechanics).

The replay itself is shown on the card (a replayed run's expert counts equal the record's histogram; KLD drops to the
cache's own). These tests cover the transform on the files the committed fixture holds (ctx.h, ctx.cpp, engine.cpp), on
top of rad0044 which it requires; the other edits (libr4d kernel/row/schema/shape, config, ABI, bring-up, block) are
applied, fail closed, by the lane build."""
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

PATCHES = REPO / "engines/radiance/patches"
PIN = REPO / "engines/radiance/tests/fixtures/19d4e8f"
FILES = ("core/runtime/ctx.h", "core/runtime/ctx.cpp", "core/engine.cpp")
BEGIN_ANCHOR = "    step_fail_.clear();\n"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, PATCHES / name / "patch.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return [p for p in mod.PATCHES if p.path in FILES]


def _fixture(dst: Path) -> dict:
    for rel in FILES:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PIN / rel, dst / rel)
    return {rel: (dst / rel).read_text(encoding="utf-8") for rel in FILES}


def _with_record(root: Path):
    first = apply_all(_load("rad0044_expert_route_record"), root)
    assert all(p.ok for p in first), first
    return {rel: (root / rel).read_text(encoding="utf-8") for rel in FILES}


def test_requires_the_record_patch():
    assert 'requires = ["rad0044_expert_route_record"]' in (PATCHES / "rad0045_expert_route_replay/patch.toml").read_text()


def test_applies_adds_lines_only_and_is_idempotent():
    with tempfile.TemporaryDirectory(dir=REPO) as temp:
        root = Path(temp)
        _fixture(root)
        before = _with_record(root)
        ps = _load("rad0045_expert_route_replay")
        assert {p.path for p in ps} == set(FILES)
        first = apply_all(ps, root)
        assert all(p.ok for p in first), first
        assert all(e.status == "applied" for p in first for e in p.results), first
        after = {rel: (root / rel).read_text(encoding="utf-8") for rel in FILES}
        for rel in FILES:
            removed = [l for l in difflib.unified_diff(before[rel].splitlines(), after[rel].splitlines(), n=0)
                       if l.startswith("-") and not l.startswith("---")]
            assert len(removed) <= 1, (rel, removed)   # only the tape condition is changed in place
        ctx = after["core/runtime/ctx.cpp"]
        assert "const int rp = route_replay_begin();" in ctx and "BC_VTP44_ROUTE_REPLAY_PASS_END" in ctx
        assert "&& route_rec_cap_ == 0 && !route_rp_;" in ctx
        # Every deviation fails the run; the pass is looked up through document starts, so the chunk size cannot matter.
        for text in ("only one-sequence prefill passes without drafting can be replayed",
                     "token positions do not continue the previous pass",
                     "the run has more documents than the record",
                     "a document is longer than the record's",
                     "the record's routed layers are not this model's"):
            assert text in ctx, text
        assert "step_status_ = rc;" in ctx and "route_rp_used_ += n;" in ctx
        # Fail closed: a layer never asked for its rows fails the pass; the admission error fails the step; a load before
        # admission is a failure, not "not replaying".
        assert "a routed layer did not take the recorded experts for exactly the pass's rows" in ctx
        assert "a layer asked for recorded experts before its pass was admitted" in ctx
        assert ctx.index("a layer asked for recorded experts before its pass was admitted") < ctx.index("return 0;   /* an MTP head's layer */")
        header = after["core/runtime/ctx.h"]
        assert header.index("struct RouteReplay") < header.index("struct CtxDesc")
        assert "const RouteReplay* route_replay = nullptr;" in header
        engine = after["core/engine.cpp"]
        assert "replayed %lld rows, the record has %lld" in engine
        assert "their token digests differ" in engine

        snapshot = after
        second = apply_all(ps, root)
        assert all(p.ok for p in second), second
        assert all(e.status == "already-applied" for p in second for e in p.results)
        assert {rel: (root / rel).read_text(encoding="utf-8") for rel in FILES} == snapshot


def test_anchor_drift_and_duplicate_fail_closed():
    ps = _load("rad0045_expert_route_replay")
    for how in ("drift", "duplicate"):
        with tempfile.TemporaryDirectory(dir=REPO) as temp:
            root = Path(temp)
            _fixture(root)
            src = _with_record(root)["core/runtime/ctx.cpp"]
            assert src.count(BEGIN_ANCHOR) == 1
            bad = src.replace(BEGIN_ANCHOR, BEGIN_ANCHOR.replace("clear", "reset") if how == "drift" else BEGIN_ANCHOR * 2)
            (root / "core/runtime/ctx.cpp").write_text(bad, encoding="utf-8")
            result = apply_all(ps, root)
            assert any(not p.ok for p in result), (how, result)
            assert (root / "core/runtime/ctx.cpp").read_text(encoding="utf-8") == bad
