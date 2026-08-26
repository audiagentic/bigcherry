"""HI67 slice 2a: patches/1222_hi67_deterministic_test_backend_ops_seed.py applies
cleanly and idempotently to the real vendored test-backend-ops.cpp, and the
patched source contains the contract the correctness-evidence generator
(slice 2c, not yet written) will depend on."""

import importlib.util
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patcher import apply_all

ROOT = Path(__file__).resolve().parents[3]

spec = importlib.util.spec_from_file_location(
    "hi67_deterministic_seed_patch",
    ROOT / "patches" / "1222_hi67_deterministic_test_backend_ops_seed.py",
)
assert spec and spec.loader
_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_module)


def _apply_to_copy(tmp_path: Path) -> Path:
    vendor = ROOT / "vendor" / "llama.cpp"
    target = tmp_path / "tests" / "test-backend-ops.cpp"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(vendor / "tests" / "test-backend-ops.cpp", target)
    return target


def test_applies_cleanly_to_the_real_pinned_source(tmp_path):
    target = _apply_to_copy(tmp_path)
    results = apply_all(_module.PATCHES, tmp_path)
    assert all(result.ok for result in results), [
        (r.edit_id, r.status, r.detail) for result in results for r in result.results
    ]
    text = target.read_text(encoding="utf-8")
    assert text.count("bigcherry_deterministic_mode") >= 2
    assert "BIGCHERRY_TEST_DETERMINISTIC_SEED" in text
    assert "BIGCHERRY_REF_DIGEST" in text


def test_apply_is_idempotent(tmp_path):
    _apply_to_copy(tmp_path)
    first = apply_all(_module.PATCHES, tmp_path)
    assert all(result.ok for result in first)
    second = apply_all(_module.PATCHES, tmp_path)
    assert all(result.ok for result in second)
    assert not any(result.changed for result in second), (
        "re-applying the patch to already-patched source must be a no-op"
    )


def test_default_behavior_is_unchanged_when_seed_is_unset():
    """The patch must not touch the non-deterministic path's own code -- the
    existing thread_local std::random_device engine and the parallel
    init_thread lambda stay exactly as upstream wrote them, only reachable
    now behind an `else` that requires BIGCHERRY_TEST_DETERMINISTIC_SEED."""
    for patch in _module.PATCHES:
        for edit in patch.edits:
            if edit.id == "hi67-deterministic-branch":
                assert "thread_local std::default_random_engine gen(std::random_device{}())" in edit.text
                assert "std::async(std::launch::async, init_thread, start, end)" in edit.text
                assert "} else {" in edit.text


def test_deterministic_branch_derives_one_call_index_reused_for_seed_and_digest():
    """Regression for a real bug caught during implementation: an earlier
    draft called bigcherry_next_call_index() a second time inside the
    fprintf, which both mis-logged the index (off by one from the one
    actually used to derive the seed) and silently consumed an extra
    counter increment, desynchronizing the call-index sequence from a
    paired native/candidate invocation."""
    for patch in _module.PATCHES:
        for edit in patch.edits:
            if edit.id == "hi67-deterministic-branch":
                assert edit.text.count("bigcherry_next_call_index()") == 1
                assert "const uint64_t call_index = bigcherry_next_call_index();" in edit.text
