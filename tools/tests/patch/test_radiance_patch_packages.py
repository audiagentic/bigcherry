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
RAD0002 = "rad0002_moe_remote_handoff"
RAD0003 = "rad0003_mtp_full_head_switch"
RAD0004 = "rad0004_moe_remote_diagnostics"

#: package -> (file, a line that must be unique in it as applied, its replacement that breaks the anchor)
CASES = {
    RAD0002: (
        "arch/common/rad_block_moe_fp8.h",
        "rad_op op_ar = 0, op_add = 0;",
        "rad_op op_ar = 0, op_add = 0, int changed_shape;",
    ),
    RAD0003: (
        "arch/common/rad_block_mtp_fp8.h",
        "if (!head || !lm) return false;",
        "if (!head || !lm) return true;",
    ),
    RAD0004: (
        "arch/common/rad_block_moe_fp8.h",
        "if (shared) sh_down.step(c_, w.sff, w.sout, T, r0, rows);",
        "if (shared) sh_down.step(c_, w.sff, w.sout, T, r0, rows + 0);",
    ),
}

#: packages that must already be applied for a package to apply (patch.toml `requires`)
PREREQUISITES = {RAD0004: (RAD0002,)}


# Git blob prefixes of the exact package outputs against Radiance 1.7.0 (48f7b8e, the release commit 1e968f6 plus two
# NOTICE-only commits); mismatches detect a port that merely matches guards while silently changes output.
# (before, after): `before` is the file as the package finds it (None: the package creates it; for a package with
# prerequisites, after they are applied), `after` the file with the package applied.
PIN_BLOBS = {
    RAD0002: {
        "abi/bc_moe_remote.h": (None, "a0dd765"),
        "arch/common/bc_moe_remote_block.h": (None, "f56d783"),
        "arch/common/rad_block_hc.h": ("44c538f", "076f308"),
        "arch/common/rad_block_moe_fp8.h": ("240b82b", "f0a772d"),
        "core/place/bc_moe_remote.cpp": (None, "1babb84"),
        "core/place/bc_moe_remote_place.h": (None, "e89a83b"),
        "core/place/planner.cpp": ("ac68e59", "8e7f945"),
        "core/engine_bringup.cpp": ("06c100e", "5446799"),
    },
    RAD0003: {
        "arch/common/rad_block_mtp_fp8.h": ("c571d45", "235cebb"),
    },
    RAD0004: {
        "arch/common/bc_moe_remote_diag.h": (None, "3998d25"),
        "arch/common/rad_block_moe_fp8.h": ("f0a772d", "e8defe4"),
    },
}

#: the core files rad0002 edits (everything else it ships is a file of its own)
RAD0002_CORE_FILES = (
    "arch/common/rad_block_hc.h",
    "arch/common/rad_block_moe_fp8.h",
    "core/engine_bringup.cpp",
    "core/place/planner.cpp",
)


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

    def _prepared(self, package_id):
        """A fresh fixture tree with the package's prerequisites applied."""
        temp, root = self._tree()
        for required in PREREQUISITES.get(package_id, ()):
            results = apply_all(list(_load(required)), root)
            self.assertTrue(all(result.ok for result in results), results)
        return temp, root

    def test_apply_output_and_idempotence(self):
        for package_id in CASES:
            with self.subTest(package=package_id):
                temp, root = self._prepared(package_id)
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

                    if package_id == RAD0003:
                        text = (root / "arch/common/rad_block_mtp_fp8.h").read_text(encoding="utf-8")
                        self.assertIn("std::atoi(e) != 0", text)
                        self.assertNotIn("*e == 1", text)

                    for relative, (_, patched_blob) in PIN_BLOBS[package_id].items():
                        self.assertEqual(
                            _git_blob_prefix((root / relative).read_bytes()),
                            patched_blob,
                            f"{package_id}: {relative}: package output changed",
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

    def test_rad0004_does_not_apply_without_rad0002(self):
        """The diagnostics hang off rad0002's extension point; on the bare pin they fail closed and write nothing."""
        temp, root = self._tree()
        try:
            before = _snapshot(root)
            results = apply_all(list(_load(RAD0004)), root)
            self.assertTrue(any(not result.ok for result in results), results)
            self.assertEqual(_snapshot(root), before)
        finally:
            temp.cleanup()

    @staticmethod
    def _added_lines(before: str, after: str) -> list[str]:
        import difflib

        return [
            line[1:]
            for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0)
            if line.startswith("+") and not line.startswith("+++")
        ]

    def test_rad0002_core_edits_are_thin_calls_into_the_extension_point(self):
        """Policy lives in rad0002's own files and the plugin: every added code line in an upstream file calls the
        extension point (the one exception is rank_units' preplaced filter, which cannot move out of the core)."""
        temp, root = self._tree()
        try:
            patches = _load(RAD0002)
            core_edits = {p.path: len(p.edits) for p in patches if not p.create}
            self.assertEqual(set(core_edits), set(RAD0002_CORE_FILES))
            self.assertEqual(sum(core_edits.values()), 14)
            self.assertEqual(sum(1 for p in patches if p.create), 4)
            self.assertTrue(all(result.ok for result in apply_all(list(patches), root)))
            for relative in RAD0002_CORE_FILES:
                added = self._added_lines(
                    (FIXTURE / relative).read_text(encoding="utf-8"),
                    (root / relative).read_text(encoding="utf-8"),
                )
                code = [line.strip() for line in added if line.strip() and not line.strip().startswith(("/*", "*"))]
                for line in code:
                    if "preplaced()" in line or line.startswith("for (int32_t i = 0; i < (int32_t)p->units.size()"):
                        self.assertEqual(relative, "core/place/planner.cpp")
                        continue
                    self.assertRegex(
                        line, r"bc_moe_remote|bc_remote|BcMoeRemote|&&$|^!gather_taken", f"{relative}: {line}"
                    )
                text = (root / relative).read_text(encoding="utf-8")
                self.assertNotIn("rad_moe_remote", text, relative)
                self.assertNotIn('getenv("R3_MOE', text, relative)
                # the shadow compare and mask-only machinery is rad0004's, not rad0002's
                self.assertNotIn("moe_mailbox_shadow", text, relative)
                self.assertNotIn("moe_mailbox_mask_ref", text, relative)
        finally:
            temp.cleanup()

    def test_rad0002_handoff_output_on_1_7_0(self):
        temp, root = self._tree()
        try:
            self.assertTrue(all(result.ok for result in apply_all(list(_load(RAD0002)), root)))
            moe = (root / "arch/common/rad_block_moe_fp8.h").read_text(encoding="utf-8")
            # Remote hand-off must not globally change the normal router/top-k arithmetic.
            self.assertIn("const bool rfused = op_router_fused && rows <= router_fused_rows;", moe)
            self.assertIn("} else if (op_topk_scatter && rows <= topk_scatter_rows) {", moe)
            self.assertNotIn("rfused = !", moe)
            # the call sites, in pass order: route -> (grouped GEMMs) -> merge replaces gather -> all-reduce
            declare = moe.index("RAD_ARCH_TRY(bc_remote.declare(b, *this, reg, l));")
            self.assertLess(moe.index('op_add = rw(b, RAD_OP(b, "add",'), declare)
            self.assertLess(declare, moe.index("xp_parts.assign("))
            after_route = moe.index("bc_remote.after_route(c_, *this, T, r0, rows, ecnt);")
            n4 = moe.index("const int64_t n4[4]")
            hq = moe.index("RAD_ISSUE_N(c_, op_hq, rows")
            self.assertLess(hq, after_route)
            self.assertLess(after_route, n4)
            merge = moe.index("if (!bc_remote.merge(c_, *this, T, r0, rows) &&")
            gather = moe.index("RAD_ISSUE_N(c_, op_gather, rows")
            ar = moe.index("if (own_ar && issues_ar(T, rows)) issue_ar(c_, r0, rows, false);")
            self.assertLess(n4, merge)
            self.assertLess(merge, gather)
            self.assertLess(gather, ar)
            self.assertIn("return op_ar && (bc_remote.active(T) ||", moe)
            hc = (root / "arch/common/rad_block_hc.h").read_text(encoding="utf-8")
            self.assertIn(
                "bool takes_ar(int64_t T) const { return ar_write(T) != 0 && !bc_moe_remote_hc_open(gsrc.top_k, T); }",
                hc,
            )
            planner = (root / "core/place/planner.cpp").read_text(encoding="utf-8")
            self.assertIn("if (bc_moe_remote_pin_unit(mu, in_.config->max_tok)) continue;", planner)
            self.assertIn("bc_moe_remote_mark_footprint(units);", planner)
            self.assertIn("if (!p->units[(size_t)i].preplaced()) order->push_back(i);", planner)
            bringup = (root / "core/engine_bringup.cpp").read_text(encoding="utf-8")
            placeholder = "if (bc_moe_remote_publish_placeholder(cfg_.max_tok, w, e, *file_, r->mover, (rad_weight)i)) continue;"
            self.assertIn(placeholder, bringup)
            self.assertLess(
                bringup.index(placeholder),
                bringup.index("if ((w.tier == Tier::Mapped || w.tier == Tier::SSD) && (!direct || !e)) {"),
            )

            abi = (root / "abi/bc_moe_remote.h").read_text(encoding="utf-8")
            core = (root / "core/place/bc_moe_remote.cpp").read_text(encoding="utf-8")
            for symbol in ("bc_moe_remote_register", "bc_moe_remote_wanted", "bc_moe_remote_wanted_upto",
                           "bc_moe_remote_layer_map", "bc_moe_remote_skip_unit", "bc_moe_remote_shadow",
                           "bc_moe_remote_mask_only", "bc_moe_remote_diag_build"):
                self.assertRegex(abi, rf"\b{symbol}\(")
                self.assertRegex(core, rf'extern "C" \w+ {symbol}\(')
            # Diagnostics are compiled out of a production build and refused loudly, never silently ignored.
            self.assertIn("#ifdef BC_MOE_REMOTE_DIAG", core)
            self.assertIn("static constexpr bool kDiagBuild = false;", core)
            self.assertIn("need the diagnostic build", core)
            self.assertIn("plugin->struct_size != sizeof(BcMoeRemotePlugin)", core)
            self.assertIn('env_on("R3_MOE_WORKER_SKIP_OWNED")', core)
            block = (root / "arch/common/bc_moe_remote_block.h").read_text(encoding="utf-8")
            self.assertIn("moe_mailbox_publish", block)
            self.assertIn("moe_mailbox_add", block)
            self.assertNotIn("moe_mailbox_shadow", block)
            self.assertNotIn("moe_mailbox_mask_ref", block)
            self.assertIn("brows(q, T));", block)
            self.assertIn("m.gfold ? brow_slice(w.sout, r0, rows, g.n_embd) : RAD_NONE", block)
            self.assertIn("m.gfold ? brow_slice(w.sgate, r0, rows, 1) : RAD_NONE", block)
        finally:
            temp.cleanup()

    def test_rad0004_diagnostics_are_compiled_out_without_the_macro(self):
        temp, root = self._prepared(RAD0004)
        try:
            pre = (root / "arch/common/rad_block_moe_fp8.h").read_text(encoding="utf-8")
            self.assertTrue(all(result.ok for result in apply_all(list(_load(RAD0004)), root)))
            post = (root / "arch/common/rad_block_moe_fp8.h").read_text(encoding="utf-8")
            added = self._added_lines(pre, post)
            # every line it adds to the upstream header sits inside an #ifdef BC_MOE_REMOTE_DIAG ... #endif
            depth = 0
            for line in added:
                text = line.strip()
                if text == "#ifdef BC_MOE_REMOTE_DIAG":
                    depth += 1
                elif text == "#endif":
                    depth -= 1
                else:
                    self.assertEqual(depth, 1, f"diagnostic line outside the macro guard: {line}")
            self.assertEqual(depth, 0)
            self.assertEqual(sum(1 for line in added if line.strip() == "#ifdef BC_MOE_REMOTE_DIAG"), 4)
            self.assertEqual(
                {p.path for p in _load(RAD0004)},
                {"arch/common/bc_moe_remote_diag.h", "arch/common/rad_block_moe_fp8.h"},
            )
            diag = (root / "arch/common/bc_moe_remote_diag.h").read_text(encoding="utf-8")
            self.assertIn("#error", diag)
            for name in ("moe_mailbox_shadow", "moe_mailbox_mask_ref", "bc_moe_remote_mask_only()",
                         "bc_moe_remote_shadow()", "bc_moe_remote_diag_build()"):
                self.assertIn(name, diag)
            self.assertIn("base.publish", diag)
        finally:
            temp.cleanup()

    def test_plugin_registration_struct_matches_the_engine_abi(self):
        """libr3moe mirrors BcMoeRemotePlugin (it builds against an unpatched tree); the two must not drift."""
        temp, root = self._tree()
        try:
            self.assertTrue(all(result.ok for result in apply_all(list(_load(RAD0002)), root)))
            abi = (root / "abi/bc_moe_remote.h").read_text(encoding="utf-8")
            hook = (REPO / "engines/radiance/components/moe-remote/moe_remote_hook.cpp").read_text(encoding="utf-8")

            def fields(text: str, opener: str) -> list[str]:
                body = text.split(opener, 1)[1].split("}", 1)[0]
                body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
                return [re.sub(r"\s+", " ", f.strip()) for f in body.split(";") if f.strip()]

            self.assertEqual(
                fields(abi, "typedef struct BcMoeRemotePlugin {"),
                fields(hook, "struct EnginePlugin {"),
            )
        finally:
            temp.cleanup()

    def test_remote_helper_merges_in_gather_order_and_exposes_diagnostics(self):
        helper = REPO / "engines" / "radiance" / "components" / "moe-remote"
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
        self.assertIn('R3_MOE_WORKER poison', hook)
        self.assertIn('R3_MOE_WORKER_SHADOW enabled', hook)
        self.assertIn('R3_MOE_WORKER_MASK_ONLY enabled', hook)
        self.assertIn('R3_MOE_WORKER_SHADOW layer=', hook)
        self.assertIn('hipMalloc mask-only engine rows', hook)
        self.assertIn('"moe_mailbox_mask_ref"', ops)
        self.assertIn('if (mask_only == 1u) src = mask_ref + (uint64_t)flat * hidden;', ops)
        self.assertIn('r.weight_bf16 != ew[flat]', ops)
        self.assertIn('if (mask_only && m && local_index[slot] >= 0) __builtin_trap();', ops)
        self.assertIn('hipMemsetAsync(d->mask_ref, 0', ops)
        self.assertIn('dlsym(RTLD_DEFAULT, "bc_moe_remote_register")', hook)
        self.assertIn('static const EnginePlugin plugin{sizeof(EnginePlugin), publish_layer_map, owner_query};', hook)
        self.assertIn('g.engine_register(&plugin)', hook)
        self.assertNotIn('rad_moe_remote', hook)
        self.assertIn('layer_map_consumed', hook)
        self.assertIn('slot_map[(uint32_t)le] < 0', ops)
        self.assertIn('{"aq", RAD_OPD_IN, 0}', ops)
        self.assertIn('{"as", RAD_OPD_IN, 0}', ops)
        self.assertIn('{"aq_base", RAD_OPD_IN, 0}', ops)
        self.assertIn('"moe_mailbox_shadow"', ops)
        self.assertIn('q->row_base', ops)
        gpu = "".join(f.read_text(encoding="utf-8") for f in [helper / "moe_remote_gpu.h", *sorted(helper.glob("moe_lane_*.h"))])
        self.assertIn('R3_MOE_WORKER_POISON', gpu)
        self.assertIn('poison == 1u ? (uint16_t)0x447au : poison == 2u ? (uint16_t)0 : (uint16_t)(u >> 16)', gpu)
        self.assertIn('build_layer_table', gpu)
        self.assertIn('slot_of_expert', gpu)
        self.assertNotIn('model_.expert_slot', gpu)
        self.assertIn('stage_compare_activation', gpu)
        self.assertIn('q, act_q, act_s', gpu)
        helper_main = "".join(f.read_text(encoding="utf-8") for f in [helper / "r3_moe_worker.cpp", *sorted(helper.glob("worker_*.h"))])
        self.assertIn("--synthetic-protect", helper_main)
        self.assertIn("did not shift an owned TP class", helper_main)
        self.assertNotIn("regular_slot", helper_main)
        component = (REPO / "engines" / "radiance" / "components" / "moe-remote" / "component.toml").read_text(encoding="utf-8")
        self.assertIn("[setting.R3_MOE_WORKER_SHADOW]", component)
        self.assertIn("[setting.R3_MOE_WORKER_MASK_ONLY]", component)
        self.assertIn('status = "diagnostic"', component)

    def test_missing_anchor_fails_closed_and_is_atomic(self):
        for package_id, (relative, before, after) in CASES.items():
            with self.subTest(package=package_id):
                temp, root = self._prepared(package_id)
                try:
                    target = root / relative
                    text = target.read_text(encoding="utf-8")
                    self.assertEqual(text.count(before), 1)
                    target.write_text(text.replace(before, after), encoding="utf-8", newline="")
                    before_apply = _snapshot(root)

                    results = apply_all(list(_load(package_id)), root)
                    self.assertTrue(any(not result.ok for result in results), results)
                    self.assertEqual(_snapshot(root), before_apply)
                    for patch in _load(package_id):
                        if patch.create:
                            self.assertFalse((root / patch.path).exists(), patch.path)
                finally:
                    temp.cleanup()


if __name__ == "__main__":
    unittest.main()
