"""RD58: patches/1234 registers the state-restore buffer as portable
pinned host memory during multi-GPU prompt-cache/checkpoint restore, to
work around ROCm/rocm-systems#4817 (a real, still-open ROCm runtime
defect: an async H2D copy from pageable host memory can fault mid-
transfer with 2+ devices in one process). Verified against the real
pinned vendor source (not a synthetic fixture) that the guard is wired
in correctly, is idempotent, and deliberately deviates from the source
PR by silencing the failure-path log line (see the patch module's own
docstring for why).

Also covers the 2026-08-24 real-hardware fix: under -sm tensor (the
mode BigCherry's actual production dual-GPU topology uses), the real
per-device HIP backends are hidden behind a single Meta scheduling
device with no registry of its own, so the original registration loop
never activated there. The patch now unwraps a Meta device via two
previously-file-local ggml-backend-meta.cpp functions, exposed as
public API by a sibling edit in the same patch module."""

import importlib.util
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patcher import apply_all


ROOT = Path(__file__).resolve().parents[3]


def _load_patch_module(patch_id: str):
    spec = importlib.util.spec_from_file_location(
        patch_id, ROOT / "patches" / patch_id / "patch.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RD58 = _load_patch_module("1234_rd58_pin_state_buffer_multigpu_restore")

_FILES = (
    "src/llama-context.cpp",
    "ggml/include/ggml-backend.h",
    "ggml/src/ggml-backend-meta.cpp",
)


def _apply_to_copy(tmp_path: Path) -> dict[str, Path]:
    vendor = ROOT / "vendor" / "llama.cpp"
    targets = {}
    for rel in _FILES:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(vendor / rel, target)
        targets[rel] = target
    return targets


def test_patch_applies_cleanly_to_the_real_pinned_source(tmp_path):
    targets = _apply_to_copy(tmp_path)
    results = apply_all(RD58.PATCHES, tmp_path)
    assert all(result.ok for result in results), [
        (r.edit_id, r.status, r.detail) for result in results for r in result.results
    ]

    ctx = targets["src/llama-context.cpp"].read_text(encoding="utf-8")
    assert "rd58_pin_guard_t" in ctx
    assert "LLAMA_STATE_SEQ_FLAGS_ON_DEVICE" in ctx
    assert "size >= 1u << 20" in ctx
    assert '"ggml_backend_register_host_buffer"' in ctx
    assert '"ggml_backend_unregister_host_buffer"' in ctx
    # RD58's own deliberate deviation from the source PR: no failure-path
    # log line -- the backend returning "no" to a register request is not
    # an error when GGML_CUDA_REGISTER_HOST is simply unset (the default).
    assert "could not pin state buffer" not in ctx
    # But the success diagnostic must remain -- it is this item's
    # real-hardware activation proof.
    assert '"%s: pinned state buffer (%zu bytes) for restore\\n"' in ctx
    # The guard must be declared before `io`, so its destructor (running
    # after the deferred H2D copies flush) unregisters only once the
    # restore is actually done.
    guard_pos = ctx.index("rd58_pin_guard_t")
    io_pos = ctx.index("std::unique_ptr<llama_io_read_i> io;")
    assert guard_pos < io_pos
    # 2026-08-24 fix: Meta devices must be unwrapped to their real
    # per-device backends before the registration loop runs, or -sm
    # tensor (BigCherry's actual production topology) never activates.
    assert "GGML_BACKEND_DEVICE_TYPE_META" in ctx
    assert "ggml_backend_meta_dev_n_devs(dev)" in ctx
    assert "ggml_backend_meta_dev_simple_dev(dev, rd58_j)" in ctx

    header = targets["ggml/include/ggml-backend.h"].read_text(encoding="utf-8")
    assert "GGML_API size_t          ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev);" in header
    assert "GGML_API ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index);" in header

    meta_src = targets["ggml/src/ggml-backend-meta.cpp"].read_text(encoding="utf-8")
    # Must no longer be file-local `static` -- llama-context.cpp needs to
    # call these from outside this translation unit.
    assert "static size_t ggml_backend_meta_dev_n_devs" not in meta_src
    assert "static ggml_backend_dev_t ggml_backend_meta_dev_simple_dev" not in meta_src
    assert "size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {" in meta_src
    assert "ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {" in meta_src


def test_patch_is_idempotent(tmp_path):
    targets = _apply_to_copy(tmp_path)
    first = apply_all(RD58.PATCHES, tmp_path)
    assert all(result.ok for result in first)
    once = {rel: t.read_text(encoding="utf-8") for rel, t in targets.items()}
    second = apply_all(RD58.PATCHES, tmp_path)
    assert all(result.ok for result in second)
    twice = {rel: t.read_text(encoding="utf-8") for rel, t in targets.items()}
    assert once == twice
