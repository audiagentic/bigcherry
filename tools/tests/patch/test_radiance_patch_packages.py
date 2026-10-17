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
FIXTURE = REPO / "engines" / "radiance" / "tests" / "fixtures" / "48f7b8e"

# rad0001 is superseded by radiance 1.7.0 (Engine::~Engine drains the mover stream itself) and is not applied.
# rad0002 is pending its 1.7.0 port (MEN12): two of its anchors no longer match, see test_rad0002_*.
CASES = {
    "rad0002_moe_remote_handoff": (
        "arch/common/rad_block_moe_fp8.h",
        "rad_op op_ar = 0, op_add = 0;",
        "rad_op op_ar = 0, op_add = 0, int changed_shape;",
    ),
    "rad0003_mtp_full_head_switch": (
        "arch/common/rad_block_mtp_fp8.h",
        "if (!head || !lm) return false;",
        "if (!head || !lm) return true;",
    ),
}


# Git blob prefixes of the exact package outputs against Radiance 1.7.0 (48f7b8e, the release commit 1e968f6 plus two
# NOTICE-only commits); mismatches detect a port that merely matches guards while silently changes output.
# The pin side is checked for every package. A patched side of None means the package does not apply on 1.7.0 yet.
PIN_BLOBS = {
    "rad0002_moe_remote_handoff": {
        "abi/rad_moe_remote.h": (None, None),
        "arch/common/rad_block_hc.h": ("44c538f", None),
        "arch/common/rad_block_moe_fp8.h": ("240b82b", None),
        "core/place/moe_remote.cpp": (None, None),
        "core/place/planner.cpp": ("ac68e59", None),
        "core/engine_bringup.cpp": ("06c100e", None),
    },
    "rad0003_mtp_full_head_switch": {
        "arch/common/rad_block_mtp_fp8.h": ("c571d45", "235cebb"),
    },
}

#: rad0002's edits whose anchors no longer match on 1.7.0 (analysis: docs/reference/engines/radiance/PORT_1.7.md).
RAD0002_UNPORTED_EDITS = {"add-remote-before-all-reduce", "remote-owned-unit-unplaced"}

#: packages whose full output is asserted on 1.7.0
APPLIED = tuple(package_id for package_id in CASES if package_id != "rad0002_moe_remote_handoff")


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
        for package_id in APPLIED:
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

                    if package_id == "rad0003_mtp_full_head_switch":
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

    def test_rad0002_pin_blobs_and_unported_edits_on_1_7_0(self):
        """rad0002 is pending its 1.7.0 port (MEN12): exactly two anchors fail, and nothing is written."""
        package_id = "rad0002_moe_remote_handoff"
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
            before = _snapshot(root)
            results = apply_all(list(patches), root)
            failed = {edit.edit_id for result in results for edit in result.failed}
            self.assertEqual(failed, RAD0002_UNPORTED_EDITS)
            self.assertEqual(_snapshot(root), before)
        finally:
            temp.cleanup()

    @unittest.expectedFailure
    def test_rad0002_applies_with_its_handoff_output_on_1_7_0(self):
        """Pending the rad0002 port (MEN12, docs/reference/engines/radiance/PORT_1.7.md); flips to an unexpected
        success once it applies, at which point fill the patched blobs into PIN_BLOBS and drop the decorator."""
        temp, root = self._tree()
        try:
            results = apply_all(list(_load("rad0002_moe_remote_handoff")), root)
            self.assertTrue(all(result.ok for result in results), results)
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
            self.assertIn(
                "rad_moe_remote_layer_map(l, (int)ne, remote_slot.data());", moe
            )
            self.assertIn(
                "remote_slot[(size_t)reg[(size_t)s]] = (int32_t)s;", moe
            )
            self.assertNotIn(
                "n_reg == (int64_t)c.n_expert", moe
            )
            abi = (root / "abi/rad_moe_remote.h").read_text(encoding="utf-8")
            core = (root / "core/place/moe_remote.cpp").read_text(encoding="utf-8")
            self.assertIn("rad_moe_remote_set_layer_map_hook", abi)
            self.assertIn("rad_moe_remote_mask_only(void);", abi)
            self.assertIn("g_layer_map_hook.store(hook, std::memory_order_release);", core)
            self.assertIn('std::getenv("R3_MOE_HELPER_MASK_ONLY")', core)
            op_hq = moe.index("if (op_hq)")
            pub = moe.index("RAD_ISSUE_N(c_, op_remote_publish, rows")
            routed_gemm = moe.index("const int64_t n4[4]", pub)
            self.assertLess(op_hq, pub)
            self.assertLess(pub, routed_gemm)
            self.assertIn("brow_slice(remote_q, r0, rows, g.n_embd)", moe)
            self.assertIn("brow_slice(remote_s, r0, rows, g.n_embd / RAD_FP8_BLOCK)", moe)
            self.assertIn("brows(remote_q, T)", moe)
            self.assertIn("if (remote_active)", moe)
            shadow = moe.index("RAD_ISSUE_N(c_, op_remote_shadow, rows")
            capture = moe.index("RAD_ISSUE_N(c_, op_remote_mask_capture, rows")
            restore = moe.index("RAD_ISSUE_N(c_, op_remote_mask_restore, rows")
            add = moe.index("RAD_ISSUE_N(c_, op_remote_add, rows")
            gather = moe.index("RAD_ISSUE_N(c_, op_gather, rows")
            ar = moe.index("RAD_ISSUE_N(c_, op_ar, rows * g.n_embd", add)
            self.assertLess(capture, restore)
            self.assertLess(restore, shadow)
            self.assertLess(shadow, add)
            self.assertLess(add, gather)
            self.assertLess(gather, ar)
            self.assertIn(
                "else if (!gather_taken(g, T, gather_out_rows, gather_out_rows6))",
                moe,
            )
            self.assertIn("if (op_ar && (remote_active ||", moe)
            self.assertIn("const bool remote_shadow = remote_policy && rad_moe_remote_shadow();", moe)
            self.assertIn("remote_active && rad_moe_remote_mask_only();", moe)
            self.assertIn("gfold ? brow_slice(w.sout, r0, rows, g.n_embd) : RAD_NONE", moe)
            self.assertIn("gfold ? brow_slice(w.sgate, r0, rows, 1) : RAD_NONE", moe)
            hc = (root / "arch/common/rad_block_hc.h").read_text(encoding="utf-8")
            self.assertIn("const bool remote_active =", hc)
            self.assertIn("!rad_moe_remote_shadow();", hc)
            self.assertIn("if (!remote_active && takes_gather(T)) {", hc)
            self.assertIn(
                "(!remote_active && takes_ar(T)) ? ar_write(T) : op_write", hc
            )
        finally:
            temp.cleanup()

    def test_remote_helper_merges_in_gather_order_and_exposes_diagnostics(self):
        helper = REPO / "engines" / "radiance" / "components" / "libr3" / "helper"
        hook = (helper / "moe_remote_hook.cpp").read_text(encoding="utf-8")
        ops = (helper / "moe_mailbox_ops.hip").read_text(encoding="utf-8")
        self.assertIn('setenv("R3_MOE_REMOTE_MIN_ROWS", "0", 0)', hook)
        self.assertIn('setenv("R3_MOE_REMOTE_MAX_ROWS", mr.c_str(), 0)', hook)
        self.assertNotIn('setenv("R3_MOE_REMOTE_MAX_ROWS", mr.c_str(), 1)', hook)
        self.assertIn('{"sorted", RAD_OPD_IN, 0}', ops)
        self.assertIn('{"down", RAD_OPD_IN, 0}', ops)
        self.assertIn('{"ew", RAD_OPD_IN, 0}', ops)
        self.assertIn('{"shared", RAD_OPD_IN, 1}', ops)
        self.assertIn('{"shared_gate", RAD_OPD_IN, 1}', ops)
        self.assertIn('{"out", RAD_OPD_OUT, 0}', ops)
        self.assertIn('for (uint32_t slot = 0; slot < top_k; ++slot)', ops)
        self.assertIn('if (src) acc += bf16_to_f32(ew[flat]) * bf16_to_f32(src[col]);', ops)
        self.assertIn('acc += bf16_to_f32(__builtin_nontemporal_load(result + col))', ops)
        self.assertIn('const uint16_t gathered = f32_to_bf16_rn(acc);', ops)
        self.assertIn(
            'f32_to_bf16_rn(bf16_to_f32(part) + bf16_to_f32(gathered))', ops
        )
        self.assertNotIn('float acc = bf16_to_f32(dst[col]);', ops)
        self.assertIn('R3_MOE_HELPER poison', hook)
        self.assertIn('R3_MOE_HELPER_SHADOW enabled', hook)
        self.assertIn('R3_MOE_HELPER_MASK_ONLY enabled', hook)
        self.assertIn('R3_MOE_HELPER_SHADOW layer=', hook)
        self.assertIn('hipMalloc mask-only engine rows', hook)
        self.assertIn('"moe_mailbox_mask_ref"', ops)
        self.assertIn('if (mask_only == 1u) src = mask_ref + (uint64_t)flat * hidden;', ops)
        self.assertIn('r.weight_bf16 != ew[flat]', ops)
        self.assertIn('if (mask_only && m && local_index[slot] >= 0) __builtin_trap();', ops)
        self.assertIn('hipMemsetAsync(d->mask_ref, 0', ops)
        self.assertIn('dlsym(RTLD_DEFAULT, "rad_moe_remote_set_layer_map_hook")', hook)
        self.assertIn('g.set_layer_map(publish_layer_map)', hook)
        self.assertIn('layer_map_consumed', hook)
        self.assertIn('slot_map[(uint32_t)le] < 0', ops)
        self.assertIn('{"aq", RAD_OPD_IN, 0}', ops)
        self.assertIn('{"as", RAD_OPD_IN, 0}', ops)
        self.assertIn('{"aq_base", RAD_OPD_IN, 0}', ops)
        self.assertIn('"moe_mailbox_shadow"', ops)
        self.assertIn('q->row_base', ops)
        gpu = (helper / "moe_remote_gpu.h").read_text(encoding="utf-8")
        self.assertIn('R3_MOE_HELPER_POISON', gpu)
        self.assertIn('poison == 1u ? (uint16_t)0x447au : poison == 2u ? (uint16_t)0 : (uint16_t)(u >> 16)', gpu)
        self.assertIn('build_layer_table', gpu)
        self.assertIn('slot_of_expert', gpu)
        self.assertNotIn('model_.expert_slot', gpu)
        self.assertIn('stage_compare_activation', gpu)
        self.assertIn('q, act_q, act_s', gpu)
        helper_main = (helper / "r3_moe_helper.cpp").read_text(encoding="utf-8")
        self.assertIn("--synthetic-protect", helper_main)
        self.assertIn("did not shift an owned TP class", helper_main)
        self.assertNotIn("regular_slot", helper_main)
        component = (REPO / "engines" / "radiance" / "components" / "libr3" / "component.toml").read_text(encoding="utf-8")
        self.assertIn("[setting.R3_MOE_HELPER_SHADOW]", component)
        self.assertIn("[setting.R3_MOE_HELPER_MASK_ONLY]", component)
        self.assertIn('status = "diagnostic"', component)

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
                    if package_id == "rad0002_moe_remote_handoff":
                        self.assertFalse((root / "abi/rad_moe_remote.h").exists())
                        self.assertFalse((root / "core/place/moe_remote.cpp").exists())
                finally:
                    temp.cleanup()


if __name__ == "__main__":
    unittest.main()
