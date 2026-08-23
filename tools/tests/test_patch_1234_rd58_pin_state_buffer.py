"""RD58: patches/1234 registers the state-restore buffer as portable
pinned host memory during multi-GPU prompt-cache/checkpoint restore, to
work around ROCm/rocm-systems#4817 (a real, still-open ROCm runtime
defect: an async H2D copy from pageable host memory can fault mid-
transfer with 2+ devices in one process). Verified against the real
pinned vendor source (not a synthetic fixture) that the guard is wired
in correctly, is idempotent, and deliberately deviates from the source
PR by silencing the failure-path log line (see the patch module's own
docstring for why)."""

import importlib.util
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry.patcher import apply_all


ROOT = Path(__file__).resolve().parents[2]


def _load_patch_module(patch_id: str):
    spec = importlib.util.spec_from_file_location(
        patch_id, ROOT / "patches" / f"{patch_id}.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RD58 = _load_patch_module("1234_rd58_pin_state_buffer_multigpu_restore")


def _apply_to_copy(tmp_path: Path) -> Path:
    vendor = ROOT / "vendor" / "llama.cpp"
    target = tmp_path / "src" / "llama-context.cpp"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(vendor / "src" / "llama-context.cpp", target)
    return target


def test_patch_applies_cleanly_to_the_real_pinned_source(tmp_path):
    target = _apply_to_copy(tmp_path)
    results = apply_all(RD58.PATCHES, tmp_path)
    assert all(result.ok for result in results), [
        (r.edit_id, r.status, r.detail) for result in results for r in result.results
    ]
    patched = target.read_text(encoding="utf-8")
    assert "rd58_pin_guard_t" in patched
    assert "LLAMA_STATE_SEQ_FLAGS_ON_DEVICE" in patched
    assert "size >= 1u << 20" in patched
    assert '"ggml_backend_register_host_buffer"' in patched
    assert '"ggml_backend_unregister_host_buffer"' in patched
    # RD58's own deliberate deviation from the source PR: no failure-path
    # log line -- the backend returning "no" to a register request is not
    # an error when GGML_CUDA_REGISTER_HOST is simply unset (the default).
    assert "could not pin state buffer" not in patched
    # But the success diagnostic must remain -- it is this item's
    # real-hardware activation proof.
    assert '"%s: pinned state buffer (%zu bytes) for restore\\n"' in patched
    # The guard must be declared before `io`, so its destructor (running
    # after the deferred H2D copies flush) unregisters only once the
    # restore is actually done.
    guard_pos = patched.index("rd58_pin_guard_t")
    io_pos = patched.index("std::unique_ptr<llama_io_read_i> io;")
    assert guard_pos < io_pos


def test_patch_is_idempotent(tmp_path):
    target = _apply_to_copy(tmp_path)
    first = apply_all(RD58.PATCHES, tmp_path)
    assert all(result.ok for result in first)
    once = target.read_text(encoding="utf-8")
    second = apply_all(RD58.PATCHES, tmp_path)
    assert all(result.ok for result in second)
    twice = target.read_text(encoding="utf-8")
    assert once == twice
