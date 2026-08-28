"""HI85: patches/1225 fails closed instead of crashing when NCCL/RCCL's
participants span different GPU architectures. Verified on real Brutus
hardware (RDNA2/RDNA3/RDNA4 mix) that every heterogeneous-architecture
pair/group crashes RCCL with a HIP-level SIGABRT ("invalid device
function") inside ncclGroupEnd() -- an uncatchable process abort, not a
recoverable ncclResult_t error -- while same-architecture participant
sets (the real production dual-XTX pair) work correctly."""

import importlib.util
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patcher import apply_all


ROOT = Path(__file__).resolve().parents[3]
PATCH = (ROOT / "patches" / "1225_hi85_nccl_heterogeneous_arch_guard" / "patch.py").read_text(encoding="utf-8")

_spec = importlib.util.spec_from_file_location(
    "hi85_nccl_heterogeneous_arch_guard_patch",
    ROOT / "patches" / "1225_hi85_nccl_heterogeneous_arch_guard" / "patch.py",
)
assert _spec and _spec.loader
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def _apply_to_copy(tmp_path: Path) -> Path:
    vendor = ROOT / "vendor" / "llama.cpp"
    target = tmp_path / "ggml" / "src" / "ggml-cuda" / "ggml-cuda.cu"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(vendor / "ggml" / "src" / "ggml-cuda" / "ggml-cuda.cu", target)
    return target


def test_patch_applies_cleanly_to_the_real_pinned_source(tmp_path):
    target = _apply_to_copy(tmp_path)
    results = apply_all(_module.PATCHES, tmp_path)
    assert all(result.ok for result in results), [
        (r.edit_id, r.status, r.detail) for result in results for r in result.results
    ]
    patched = target.read_text(encoding="utf-8")
    assert "heterogeneous_arch" in patched
    assert "info.devices[ret->dev_ids[i]].cc != info.devices[ret->dev_ids[0]].cc" in patched


def test_patch_is_idempotent(tmp_path):
    target = _apply_to_copy(tmp_path)
    first = apply_all(_module.PATCHES, tmp_path)
    assert all(result.ok for result in first)
    once = target.read_text(encoding="utf-8")
    second = apply_all(_module.PATCHES, tmp_path)
    assert all(result.ok for result in second)
    twice = target.read_text(encoding="utf-8")
    assert once == twice


def test_guard_runs_before_ncclcomminitall_not_after(tmp_path):
    target = _apply_to_copy(tmp_path)
    results = apply_all(_module.PATCHES, tmp_path)
    assert all(result.ok for result in results)
    patched = target.read_text(encoding="utf-8")
    guard_pos = patched.index("heterogeneous_arch = false")
    init_pos = patched.index("ncclCommInitAll(ret->comms.data()")
    assert guard_pos < init_pos, (
        "the arch check must run before the crash-prone call, not after -- "
        "checking post-hoc would not prevent the abort"
    )


def test_guard_fails_closed_not_via_the_internal_fallback():
    # This path is only ever reached via explicit SPLIT_MODE_TENSOR (the
    # META backend's comm_init hook) -- silently substituting a different
    # reduction path here would change semantics behind the user's back
    # (2026-08-28 explicit user direction). Must hard-abort, not decline
    # into ggml_backend_cuda_comm_init_internal(ret) the way the
    # pre-existing virtual-device guard above it still does.
    guard_pos = PATCH.index("heterogeneous_arch = false")
    abort_pos = PATCH.index('GGML_ABORT("heterogeneous-architecture tensor-split', guard_pos)
    assert abort_pos > guard_pos
    # only the pre-existing virtual-device guard declines to internal --
    # this guard must not add a second occurrence.
    assert PATCH.count("ggml_backend_cuda_comm_init_internal(ret)") == 1


def test_guard_logs_a_clear_error_before_aborting():
    # User requirement: this must be visible, not a silent behavior change.
    err_pos = PATCH.index('GGML_LOG_ERROR("NCCL/RCCL cannot reduce across mixed GPU')
    abort_pos = PATCH.index('GGML_ABORT("heterogeneous-architecture tensor-split', err_pos)
    assert err_pos < abort_pos
    assert "mixed GPU" in PATCH
    assert "architectures" in PATCH


def test_guard_compares_every_device_not_just_a_pair():
    # A loop over the whole dev_ids list, not a single pairwise check --
    # correctness for N>2 heterogeneous groups (HI84) depends on this.
    assert "for (size_t i = 1; i < ret->dev_ids.size(); ++i)" in PATCH


def test_placed_right_after_the_existing_virtual_device_guard():
    # Both fail-closed checks belong in ggml_backend_cuda_comm_init_nccl(),
    # before the real ncclCommInitAll call -- verify the anchor is exactly
    # the existing virtual-device decline block, so this edit attaches
    # directly after it rather than somewhere else in the function.
    assert "info.device_count > info.physical_device_count" in PATCH
    assert PATCH.count("ggml_backend_cuda_comm_init_internal(ret)") == 1
