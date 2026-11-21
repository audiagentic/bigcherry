"""The ABI43 wide-verify replay guard applies and refuses unknown split geometry."""
from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
from bigcherry.patcher import apply_all

COMMIT = "3deabdaeca17b58bad90d332f4b204cf7bb8a10e"
SOURCE = "arch/qwen4exp_fp8/qwen4exp_fp8.cpp"
PACKAGE = REPO / "engines/radiance/patches/rad0047_mtp_verify_replay_summary/patch.py"
HELPER_PATH = Path(__file__).with_name("test_radiance_mtp_pass_replay.py")
spec = importlib.util.spec_from_file_location("mtp_replay_helpers", HELPER_PATH)
assert spec and spec.loader
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def _patch():
    package_spec = importlib.util.spec_from_file_location("rad0047_mtp_verify_replay_summary", PACKAGE)
    assert package_spec and package_spec.loader
    module = importlib.util.module_from_spec(package_spec)
    package_spec.loader.exec_module(module)
    return module.PATCH


def _candidate_source() -> Path | None:
    configured = os.environ.get("RADIANCE_CANDIDATE_SRC")
    if configured:
        return Path(configured)
    work_root = os.environ.get("BC_WORK_ROOT")
    if work_root:
        return Path(work_root) / "radiance-latest-bump/source"
    return None


def test_candidate_patch_is_exact_and_idempotent(tmp_path):
    checkout = _candidate_source()
    if checkout is None or not checkout.is_dir():
        pytest.skip("set RADIANCE_CANDIDATE_SRC to the frozen ABI43 source")
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True,
    ).strip()
    assert actual == COMMIT
    target = tmp_path / SOURCE
    target.parent.mkdir(parents=True)
    target.write_bytes(subprocess.check_output(
        ["git", "-C", str(checkout), "show", f"{COMMIT}:{SOURCE}"],
    ))
    first = apply_all([_patch()], tmp_path)
    assert first and all(result.ok for result in first), first
    assert [edit.status for result in first for edit in result.results] == ["applied"]
    patched = target.read_bytes()
    second = apply_all([_patch()], tmp_path)
    assert second and all(result.ok for result in second), second
    assert [edit.status for result in second for edit in result.results] == ["already-applied"]
    assert target.read_bytes() == patched


def test_wide_verify_with_unknown_full_kv_summary_is_not_recorded():
    scratch_root = Path(os.environ.get("BC_WORK_ROOT", REPO))
    if not scratch_root.is_dir():
        scratch_root = REPO
    with tempfile.TemporaryDirectory(prefix="rad0047-", dir=scratch_root) as scratch:
        tmp_path = Path(scratch)
        checkout = _candidate_source()
        if checkout is None or not checkout.is_dir():
            pytest.skip("set RADIANCE_CANDIDATE_SRC to the frozen ABI43 source")
        cxx = shutil.which("g++") or shutil.which("clang++")
        if not cxx:
            pytest.skip("no C++ compiler")

        target = tmp_path / SOURCE
        target.parent.mkdir(parents=True)
        target.write_bytes(subprocess.check_output(
            ["git", "-C", str(checkout), "show", f"{COMMIT}:{SOURCE}"],
        ))
        result = apply_all([_patch()], tmp_path)
        assert result and all(item.ok for item in result), result

        suffix = r"""
extern "C" uint64_t verify_key(int summary) {
    static RadCtx c;
    static RadKVGroupBatch kv;
    RadBatch b;
    b.phase = RAD_PHASE_DECODE;
    b.draft_pass = 0;
    b.n_spec = 7;
    b.n_tok = 8;
    b.n_seq = 1;
    b.max_q_len = 8;
    b.max_ctx_len = 2048;
    b.n_kv_groups = 1;
    b.kv = &kv;
    int32_t used[1] = {summary};
    return rad_arch_pass_key(&c, &b, used);
}
"""
        cpp = tmp_path / "verify.cpp"
        cpp.write_text(helpers.callback_from_source(target.read_text()) + suffix)
        shared = tmp_path / "verify.so"
        built = subprocess.run(
            [cxx, "-std=c++17", "-O0", "-shared", "-fPIC", str(cpp), "-o", str(shared)],
            capture_output=True, text=True, check=False,
        )
        assert built.returncode == 0, built.stderr
        lib = ctypes.CDLL(str(shared))
        lib.verify_key.argtypes = [ctypes.c_int]
        lib.verify_key.restype = ctypes.c_uint64
        assert lib.verify_key(0) == 0
        assert lib.verify_key(1504) != 0
