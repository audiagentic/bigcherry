"""Mechanics tests for RAD01 Radiance patch packages.

These intentionally load patch.py directly instead of the registry: until RAD01 PR 2,
registry discovery is rooted at engines/llamacpp/patches.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

from bigcherry.patcher import apply_all  # noqa: E402

PACKAGES = REPO / "engines" / "radiance" / "patches"
FIXTURE = REPO / "engines" / "radiance" / "tests" / "fixtures" / "89cee7c"

CASES = {
    "0001_drain_mover_before_arena_free": (
        "core/engine.cpp",
        "for (auto& r : ranks_) r->ctx.shutdown();",
        "for (auto& r : ranks_) r->ctx.shutdown_now();",
    ),
    "0002_moe_remote_handoff": (
        "arch/common/rad_block_moe_fp8.h",
        "rad_op op_ar = 0, op_add = 0;",
        "rad_op op_ar = 0, op_add = 0, int changed_shape;",
    ),
    "0003_mtp_full_head_switch": (
        "arch/common/rad_block_mtp_fp8.h",
        "if (!head || !lm) return false;",
        "if (!head || !lm) return true;",
    ),
}


# Git blob prefixes for the exact package outputs against Radiance 1.3.0
# (89cee7ce4); mismatches detect a port that merely matches guards while
# silently changes output. 0002 now contains the decode-row policy gate,
# post-gather mailbox add, and HyperConn fused-path exclusion needed by remote passes.
# 0003: diff body was amended to std::atoi(e) != 0 after its index line
# was generated; 62849e9 is stale and the corrected hunk produces 8e1f89e.
PIN_BLOBS = {
    "0001_drain_mover_before_arena_free": {
        "core/engine.cpp": ("3ba2831", "20767f7"),
    },
    "0002_moe_remote_handoff": {
        "abi/rad_moe_remote.h": (None, "9e5f03d"),
        "arch/common/rad_block_hc.h": ("c19c8b6", "14c5886"),
        "arch/common/rad_block_moe_fp8.h": ("51b7c28", "6794711"),
        "core/place/moe_remote.cpp": (None, "3705adf"),
    },
    "0003_mtp_full_head_switch": {
        "arch/common/rad_block_mtp_fp8.h": ("dac7f12", "8e1f89e"),
    },
}


def _git_blob_prefix(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()[:7]


def _load(package_id: str):
    path = PACKAGES / package_id / "patch.py"
    spec = importlib.util.spec_from_file_location(f"radiance_{package_id}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    patches = getattr(module, "PATCHES", None)
    if patches is None:
        patch = getattr(module, "PATCH", None)
        patches = (patch,) if patch is not None else ()
    if not patches:
        raise AssertionError(f"{package_id} exports neither PATCH nor PATCHES")
    return tuple(patches)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


class TestRadiancePatchPackages(unittest.TestCase):
    def _tree(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        shutil.copytree(FIXTURE, root, dirs_exist_ok=True)
        return temp, root

    def test_apply_output_and_idempotence(self):
        for package_id in CASES:
            with self.subTest(package=package_id):
                temp, root = self._tree()
                try:
                    patches = _load(package_id)
                    self.assertEqual({patch.path for patch in patches}, set(PIN_BLOBS[package_id]))
                    for relative, (pin_blob, _) in PIN_BLOBS[package_id].items():
                        target = root / relative
                        if pin_blob is None:
                            self.assertFalse(target.exists(), relative)
                        else:
                            self.assertEqual(_git_blob_prefix(target.read_bytes()), pin_blob, relative)
                    first = apply_all(list(patches), root)
                    self.assertTrue(all(result.ok for result in first), first)
                    self.assertTrue(
                        all(edit.status == "applied" for result in first for edit in result.results),
                        first,
                    )
                    for patch in patches:
                        text = (root / patch.path).read_text(encoding="utf-8")
                        for edit in patch.edits:
                            self.assertRegex(
                                text,
                                re.compile(edit.guard_pattern(), re.MULTILINE),
                                f"{package_id}:{patch.path}:{edit.id}",
                            )

                    if package_id == "0002_moe_remote_handoff":
                        self.assertTrue((root / "abi/rad_moe_remote.h").is_file())
                        self.assertTrue((root / "core/place/moe_remote.cpp").is_file())
                        moe = (root / "arch/common/rad_block_moe_fp8.h").read_text(encoding="utf-8")
                        # Remote hand-off must not globally change the normal router/top-k arithmetic.
                        self.assertIn(
                            "const bool rfused = op_router_fused && rows <= router_fused_rows;", moe
                        )
                        self.assertIn(
                            "} else if (op_topk_scatter && rows <= topk_scatter_rows) {", moe
                        )
                        self.assertNotIn("rfused = !remote_policy", moe)
                        self.assertNotIn("!remote_policy && op_topk_scatter", moe)
                        self.assertIn("{w.h.x}, {w.h.x});", moe)
                        pub = moe.index("RAD_ISSUE_N(c_, op_remote_publish, rows")
                        rescatter = moe.index("RAD_ISSUE_N(c_, op_scatter, rows", pub)
                        routed_gemm = moe.index("if (op_hq)", rescatter)
                        self.assertLess(pub, rescatter)
                        self.assertLess(rescatter, routed_gemm)
                        gather = moe.index("RAD_ISSUE_N(c_, op_gather, rows")
                        add = moe.index("RAD_ISSUE_N(c_, op_remote_add, rows")
                        ar = moe.index("RAD_ISSUE_N(c_, op_ar, rows * g.n_embd", add)
                        self.assertLess(gather, add)
                        self.assertLess(add, ar)
                        self.assertIn(
                            "if (remote_policy || !gather_taken(g, T, gather_out_rows, gather_out_rows6))",
                            moe,
                        )
                        self.assertIn("if (op_ar && (remote_policy ||", moe)
                        hc = (root / "arch/common/rad_block_hc.h").read_text(encoding="utf-8")
                        self.assertIn("if (!remote_policy && takes_gather(T)) {", hc)
                        self.assertIn(
                            "(!remote_policy && takes_ar(T)) ? ar_write(T) : op_write", hc
                        )
                    if package_id == "0003_mtp_full_head_switch":
                        text = (root / "arch/common/rad_block_mtp_fp8.h").read_text(encoding="utf-8")
                        self.assertIn("std::atoi(e) != 0", text)
                        self.assertNotIn("*e == 1", text)

                    for relative, (_, patched_blob) in PIN_BLOBS[package_id].items():
                        self.assertEqual(
                            _git_blob_prefix((root / relative).read_bytes()),
                            patched_blob,
                            f"{package_id}: {relative}: source differs from original Radiance diff",
                        )
                    after_first = _snapshot(root)
                    second = apply_all(list(patches), root)
                    self.assertTrue(all(result.ok for result in second), second)
                    self.assertTrue(
                        all(edit.status == "already-applied" for result in second for edit in result.results),
                        second,
                    )
                    self.assertEqual(_snapshot(root), after_first)
                finally:
                    temp.cleanup()

    def test_remote_helper_adds_after_gather_and_exposes_poison_proof(self):
        helper = REPO / "engines" / "radiance" / "components" / "libr3" / "helper"
        hook = (helper / "moe_remote_hook.cpp").read_text(encoding="utf-8")
        ops = (helper / "moe_mailbox_ops.hip").read_text(encoding="utf-8")
        self.assertIn('setenv("R3_MOE_REMOTE_MIN_ROWS", "0", 0)', hook)
        self.assertIn('setenv("R3_MOE_REMOTE_MAX_ROWS", mr.c_str(), 0)', hook)
        self.assertNotIn('setenv("R3_MOE_REMOTE_MAX_ROWS", mr.c_str(), 1)', hook)
        self.assertIn('{"out", RAD_OPD_INOUT, 0}', ops)
        self.assertIn('r.weight_bf16', ops)
        self.assertIn('acc += bf16_product(weight, value);', ops)
        self.assertIn('R3_MOE_HELPER poison', hook)
        gpu = (helper / "moe_remote_gpu.h").read_text(encoding="utf-8")
        self.assertIn('R3_MOE_HELPER_POISON', gpu)
        self.assertIn('poison ? (uint16_t)0x447au : down[i]', gpu)

    def test_missing_anchor_fails_closed_and_is_atomic(self):
        for package_id, (relative, before, after) in CASES.items():
            with self.subTest(package=package_id):
                temp, root = self._tree()
                try:
                    target = root / relative
                    text = target.read_text(encoding="utf-8")
                    self.assertEqual(text.count(before), 1)
                    target.write_text(text.replace(before, after), encoding="utf-8", newline="")
                    before_apply = _snapshot(root)

                    results = apply_all(list(_load(package_id)), root)
                    self.assertTrue(any(not result.ok for result in results), results)
                    self.assertEqual(_snapshot(root), before_apply)
                    if package_id == "0002_moe_remote_handoff":
                        self.assertFalse((root / "abi/rad_moe_remote.h").exists())
                        self.assertFalse((root / "core/place/moe_remote.cpp").exists())
                finally:
                    temp.cleanup()


if __name__ == "__main__":
    unittest.main()
