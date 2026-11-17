"""VTP44: rad0043 maps a layer's protected experts in Engine::xstats_open (pinned v1.17, hardware-free mechanics).

The semantics (a Flash-Next run counts every selection, `ranks.unmapped` 0) are shown on the card; these tests cover the
transform: it applies to the pinned source, adds lines only, is idempotent, lands inside xstats_open, and fails closed
when the anchor drifts or is duplicated."""
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

PACKAGE = REPO / "engines/radiance/patches/rad0043_expert_stats_protected"
PIN = REPO / "engines/radiance/tests/fixtures/19d4e8f"
REL = "core/engine.cpp"
ANCHOR = "            for (int32_t& v : row) if (v == kUnset || v < 0) v = -1;\n"


def patches():
    spec = importlib.util.spec_from_file_location("rad0043", PACKAGE / "patch.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return list(mod.PATCHES)


def _fixture(dst: Path) -> str:
    target = dst / REL
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PIN / REL, target)
    return target.read_text(encoding="utf-8")


def test_applies_adds_lines_only_inside_xstats_open_and_is_idempotent():
    with tempfile.TemporaryDirectory(dir=REPO) as temp:
        root = Path(temp)
        before = _fixture(root)
        ps = patches()
        assert {p.path for p in ps} == {REL}
        first = apply_all(ps, root)
        assert all(p.ok for p in first), first
        assert all(e.status == "applied" for p in first for e in p.results), first
        after = (root / REL).read_text(encoding="utf-8")
        removed = [line for line in difflib.unified_diff(before.splitlines(), after.splitlines(), n=0)
                   if line.startswith("-") and not line.startswith("---")]
        assert not removed, removed[:10]
        assert after.count("xs_extra") > 1
        # The new loop sits after the rows are normalised and before the model's expert counts are read.
        start = after.index("int Engine::xstats_open()")
        end = after.index("xs_map_build(ids, n_model", start)
        normalise = after.index("if (v == kUnset || v < 0) v = -1;", start)
        added = after.index("xs_extra", start)
        assert start < normalise < added < end
        # It labels each slot from the tensor name and appends the layer-scope (protected) experts.
        assert "else if ((size_t)at < row.size()) row[(size_t)at] = xs_e;" in after
        assert "std::unique(g.begin(), g.end()));" in after

        snapshot = after
        second = apply_all(ps, root)
        assert all(p.ok for p in second), second
        assert all(e.status == "already-applied" for p in second for e in p.results)
        assert (root / REL).read_text(encoding="utf-8") == snapshot


def test_anchor_drift_and_duplicate_fail_closed():
    ps = patches()
    with tempfile.TemporaryDirectory(dir=REPO) as temp:
        root = Path(temp)
        source = _fixture(root)
        assert source.count(ANCHOR) == 1
        drifted = source.replace(ANCHOR, ANCHOR.replace("v < 0", "v <= -1"))
        (root / REL).write_text(drifted, encoding="utf-8")
        result = apply_all(ps, root)
        assert any(not p.ok for p in result), result
        assert (root / REL).read_text(encoding="utf-8") == drifted
    with tempfile.TemporaryDirectory(dir=REPO) as temp:
        root = Path(temp)
        source = _fixture(root)
        doubled = source.replace(ANCHOR, ANCHOR + ANCHOR)
        (root / REL).write_text(doubled, encoding="utf-8")
        result = apply_all(ps, root)
        assert any(not p.ok for p in result), result
        assert (root / REL).read_text(encoding="utf-8") == doubled
