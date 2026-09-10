"""HI166: patches/0850 adds an ordered per-verify-step speculative-decode
acceptance trace (server_slot_stats.draft_trace), because llama-server's
existing aggregate (draft_n, draft_n_accepted) scalars cannot distinguish
two genuinely different per-step work schedules that sum to the same
totals -- exactly the class of gap that let a real regression (HI141)
through the pre-promotion behavioral gate once."""

import importlib.util
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patcher import apply_all

ROOT = Path(__file__).resolve().parents[3]
PATCH_DIR = ROOT / "patches" / "0850_ordered_speculative_trace"

_spec = importlib.util.spec_from_file_location(
    "hi166_ordered_speculative_trace_patch", PATCH_DIR / "patch.py",
)
assert _spec and _spec.loader
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def _apply_to_copy(tmp_path: Path) -> dict[str, Path]:
    vendor = ROOT / "vendor" / "llama.cpp" / "tools" / "server"
    targets = {}
    for name in ("server-common.h", "server-common.cpp", "server-context.cpp"):
        target = tmp_path / "tools" / "server" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(vendor / name, target)
        targets[name] = target
    return targets


def test_patch_applies_cleanly_to_the_real_pinned_source(tmp_path):
    targets = _apply_to_copy(tmp_path)
    results = apply_all(_module.PATCHES, tmp_path)
    assert all(result.ok for result in results), [
        (r.edit_id, r.status, r.detail) for result in results for r in result.results
    ]

    header = targets["server-common.h"].read_text(encoding="utf-8")
    assert "std::vector<std::pair<uint32_t, uint32_t>> draft_trace;" in header
    assert "#include <utility>" in header

    common = targets["server-common.cpp"].read_text(encoding="utf-8")
    assert 'base["draft_trace"] = std::move(trace);' in common

    context = targets["server-context.cpp"].read_text(encoding="utf-8")
    assert "slot.stats.draft_trace.emplace_back(" in context


def test_patch_is_idempotent(tmp_path):
    targets = _apply_to_copy(tmp_path)
    first = apply_all(_module.PATCHES, tmp_path)
    assert all(result.ok for result in first)
    once = {name: t.read_text(encoding="utf-8") for name, t in targets.items()}
    second = apply_all(_module.PATCHES, tmp_path)
    assert all(result.ok for result in second)
    twice = {name: t.read_text(encoding="utf-8") for name, t in targets.items()}
    assert once == twice


def test_record_point_is_before_the_checkpoint_rollback_early_return(tmp_path):
    # HI166's whole point: recording after the rollback branch would use a
    # replay-truncated n_draft on the resumed iteration and double-count the
    # same logical verify decision. The recording line must appear BEFORE
    # the checkpoint-restore branch in source order.
    targets = _apply_to_copy(tmp_path)
    results = apply_all(_module.PATCHES, tmp_path)
    assert all(result.ok for result in results)
    context = targets["server-context.cpp"].read_text(encoding="utf-8")
    record_pos = context.index("slot.stats.draft_trace.emplace_back(")
    rollback_pos = context.index("slot.spec_is_replay = true;")
    assert record_pos < rollback_pos


def test_recording_is_guarded_on_not_spec_is_replay(tmp_path):
    targets = _apply_to_copy(tmp_path)
    results = apply_all(_module.PATCHES, tmp_path)
    assert all(result.ok for result in results)
    context = targets["server-context.cpp"].read_text(encoding="utf-8")
    # the guard must immediately precede the record call, not be some
    # unrelated earlier `if` in the function
    guard_and_record = context[
        context.index("if (!slot.spec_is_replay) {") :
        context.index("slot.stats.draft_trace.emplace_back(") + 40
    ]
    assert "emplace_back" in guard_and_record


def test_serialization_only_emits_when_trace_is_non_empty(tmp_path):
    # Mirrors the existing draft_n > 0 gate immediately above it -- a
    # non-speculative request must not gain a spurious empty draft_trace key.
    targets = _apply_to_copy(tmp_path)
    results = apply_all(_module.PATCHES, tmp_path)
    assert all(result.ok for result in results)
    common = targets["server-common.cpp"].read_text(encoding="utf-8")
    assert "if (!draft_trace.empty()) {" in common


def test_patch_fails_closed_on_missing_anchor(tmp_path):
    target = tmp_path / "tools" / "server" / "server-context.cpp"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("// unrelated content, no accept-and-verify block here\n", encoding="utf-8")
    (tmp_path / "tools" / "server" / "server-common.h").write_text("", encoding="utf-8")
    (tmp_path / "tools" / "server" / "server-common.cpp").write_text("", encoding="utf-8")

    results = apply_all(_module.PATCHES, tmp_path)
    assert not all(result.ok for result in results)
