"""Mechanics tests for RAD01 Radiance patch packages.

These intentionally load patch.py directly instead of the registry: until RAD01 PR 2,
registry discovery is rooted at engines/llamacpp/patches.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

from bigcherry.patcher import apply_all  # noqa: E402

PACKAGES = REPO / "engines" / "radiance" / "patches"
FIXTURE = REPO / "engines" / "radiance" / "tests" / "fixtures" / "4a61d72"

# rad0001 is superseded by radiance 1.7.0 (Engine::~Engine drains the mover stream itself) and is not applied.
RAD0002 = "rad0002_moe_remote_handoff"
RAD0003 = "rad0003_mtp_full_head_switch"
RAD0004 = "rad0004_moe_remote_diagnostics"
RAD0009 = "rad0009_serving_manifest"
RAD0011 = "rad0011_plan_memory"

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
    RAD0009: (
        "core/main.cpp",
        "int main(int argc, char** argv) {",
        "int main(int argc, char** argv, char** envp) {",
    ),
}

#: packages that must already be applied for a package to apply (patch.toml `requires`)
PREREQUISITES = {RAD0004: (RAD0002,)}


# Git blob prefixes of the exact package outputs against Radiance 1.14.0 (the release commit 4a61d72);
# mismatches detect a port that merely matches guards while silently changes output.
# (before, after): `before` is the file as the package finds it (None: the package creates it; for a package with
# prerequisites, after they are applied), `after` the file with the package applied.
PIN_BLOBS = {
    RAD0002: {
        "abi/bc_moe_remote.h": (None, "d3748b9"),
        "arch/common/bc_moe_remote_block.h": (None, "271a9a6"),
        "arch/common/rad_block_hc.h": ("3a602e5", "d1124d5"),
        "arch/common/rad_block_moe_fp8.h": ("614db0e", "5777cc8"),
        "core/engine.cpp": ("7544310", "c0c5357"),
        "core/engine.h": ("b10743c", "11473eb"),
        "core/engine_bringup.cpp": ("3ecdba8", "fe411f0"),
        "core/place/bc_moe_remote.cpp": (None, "381430e"),
        "core/place/bc_moe_remote_engine.h": (None, "cfb316a"),
        "core/place/bc_moe_remote_place.h": (None, "da896b6"),
        "core/place/planner.cpp": ("2ea4220", "7944888"),
        "core/runtime/ctx.cpp": ("ff0f746", "2ff379e"),
    },
    RAD0003: {
        "arch/common/rad_block_mtp_fp8.h": ("c571d45", "235cebb"),
    },
    RAD0004: {
        "arch/common/bc_moe_remote_diag.h": (None, "3998d25"),
        "arch/common/rad_block_moe_fp8.h": ("5777cc8", "feb3865"),
    },
    RAD0009: {
        "core/bc_serving_manifest.h": (None, "4ad1a40"),
        "core/bc_serving_manifest.cpp": (None, "124ef4d"),
        "core/main.cpp": ("4cb6d43", "7feb7b6"),
    },
}

#: the core files rad0002 edits (everything else it ships is a file of its own)
RAD0002_CORE_FILES = (
    "arch/common/rad_block_hc.h",
    "arch/common/rad_block_moe_fp8.h",
    "core/engine.cpp",
    "core/engine.h",
    "core/engine_bringup.cpp",
    "core/place/planner.cpp",
    "core/runtime/ctx.cpp",
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
        extension point, including the preplaced filter."""
        temp, root = self._tree()
        try:
            patches = _load(RAD0002)
            core_edits = {p.path: len(p.edits) for p in patches if not p.create}
            self.assertEqual(set(core_edits), set(RAD0002_CORE_FILES))
            self.assertEqual(sum(core_edits.values()), 28)
            self.assertEqual(sum(1 for p in patches if p.create), 5)
            self.assertTrue(all(result.ok for result in apply_all(list(patches), root)))
            for relative in RAD0002_CORE_FILES:
                added = self._added_lines(
                    (FIXTURE / relative).read_text(encoding="utf-8"),
                    (root / relative).read_text(encoding="utf-8"),
                )
                code = [line.strip() for line in added if line.strip() and not line.strip().startswith(("/*", "*"))]
                for line in code:
                    self.assertRegex(
                        line, r"bc_moe_remote|bc_remote|BcMoeRemote", f"{relative}: {line}"
                    )
                text = (root / relative).read_text(encoding="utf-8")
                self.assertNotIn("rad_moe_remote", text, relative)
                self.assertNotIn('getenv("R3_MOE', text, relative)
                # the shadow compare and mask-only machinery is rad0004's, not rad0002's
                self.assertNotIn("moe_mailbox_shadow", text, relative)
                self.assertNotIn("moe_mailbox_mask_ref", text, relative)
        finally:
            temp.cleanup()

    def test_rad0002_handoff_output_on_the_pin(self):
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
            self.assertIn("bc_moe_remote_rank_units(p, order);", planner)
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

    def test_rad0002_actual_declare_policy_to_engine_admission_gate(self):
        # Compile the ACTUAL managed-patch policy functions, not a duplicated
        # Python parser of MIN_ROWS/MAX_ROWS. Feed their output to the exact
        # header called by moe_remote_hook.cpp::admit_ticket. No GPU required.
        temp, root = self._tree()
        try:
            self.assertTrue(all(result.ok for result in apply_all(list(_load(RAD0002)), root)))
            core = (root / "core/place/bc_moe_remote.cpp").read_text(encoding="utf-8")
            policy_helpers = core[core.index("#ifdef BC_MOE_REMOTE_DIAG"):
                                  core.index("std::mutex g_mu;")] + "\n}\n"
            policy_entry = core[core.index('extern "C" int bc_moe_remote_wanted('):
                                core.index('extern "C" int bc_moe_remote_shadow(')]
            gate_header = REPO / "engines/radiance/components/moe-remote"
            hook = (gate_header / "moe_remote_hook.cpp").read_text(encoding="utf-8")
            self.assertIn('dlsym(RTLD_DEFAULT,"bc_moe_remote_wanted_upto")', hook)
            self.assertIn('admission_bootstrap(\n        g.remote_possible,g.bootstrap_ready_ranks,mask)', hook)
            program = """
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <vector>
#include "engine_admission_gate.h"
""" + policy_helpers + policy_entry + """
int main(int argc, char** argv) {
    if (argc != 4) return 7;
    setenv("R3_MOE_REMOTE_MIN_ROWS", argv[1], 1);
    setenv("R3_MOE_REMOTE_MAX_ROWS", argv[2], 1);
    const bool declared = bc_moe_remote_wanted_upto(128) != 0;
    const bool any_remote = bc_moe_remote_wanted_upto(LLONG_MAX) != 0;
    const bool expect = argv[3][0] == '1';
    if (declared != expect || any_remote != expect) return 8;
    std::atomic<uint32_t> ready{0};
    using r3::epoch::AdmissionBootstrap;
    const auto before = r3::epoch::admission_bootstrap(any_remote, ready, 3);
    if (before != (expect ? AdmissionBootstrap::Uninitialized :
                            AdmissionBootstrap::LocalOnly)) return 9;
    ready.store(1u,std::memory_order_release);
    if (r3::epoch::admission_bootstrap(any_remote, ready, 3) !=
        (expect ? AdmissionBootstrap::Uninitialized :
                  AdmissionBootstrap::LocalOnly)) return 10;
    ready.store(3u,std::memory_order_release);
    if (r3::epoch::admission_bootstrap(any_remote, ready, 3) !=
        (expect ? AdmissionBootstrap::Ready :
                  AdmissionBootstrap::LocalOnly)) return 11;
    return 0;
}
"""
            with tempfile.TemporaryDirectory() as build:
                exe = Path(build) / "policy_gate"
                compiled = subprocess.run(["c++", "-std=c++17", "-x", "c++",
                    "-", "-o", str(exe), "-I", str(gate_header)],
                    input=program, text=True, capture_output=True, check=False)
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                for lo, hi, allowed in (("1", "0", "0"), ("1", "64", "1")):
                    case = subprocess.run([str(exe), lo, hi, allowed],
                                          capture_output=True, text=True, check=False)
                    self.assertEqual(case.returncode, 0,
                                     f"actual engine policy min={lo} max={hi}: {case.stderr}")
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

    def test_admission_bridge_preserves_parent_across_issue_ahead(self):
        """Exercise the generated bridge with fake scheduler/device callbacks."""
        import subprocess

        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        temp, root = self._tree()
        try:
            self.assertTrue(all(result.ok for result in apply_all(list(_load(RAD0002)), root)))
            stub = root / "card-free" / "device"
            stub.mkdir(parents=True)
            (stub / "device.h").write_text(
                "#pragma once\n"
                "enum { RAD_OK=0, RAD_E_STATE=-1, RAD_E_DEVICE=-2 };\n",   # no rad_dev_set: the bridge never switches device
                encoding="utf-8",
            )
            binary = root / "admission-bridge-test"
            source = REPO / "engines/radiance/tests/admission_bridge_test.cpp"
            compiled = subprocess.run(
                [compiler, "-std=c++17", "-I", str(root / "card-free"),
                 "-I", str(root / "abi"), "-I", str(root / "core/place"),
                 str(source), "-o", str(binary)],
                capture_output=True, text=True,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
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
                body = body.replace("struct BcMoeRemoteTicket", "StepTicket")
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
        self.assertIn("static const EnginePlugin plugin{sizeof(EnginePlugin), publish_layer_map, owner_query, burst_transition,", hook)
        self.assertIn("before_pass, admit_ticket, bind_ticket, cancel_ticket,", hook)
        self.assertIn("host_done_ticket, rank_done_ticket, poll_tickets,", hook)
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


class TestServingManifestPatch(unittest.TestCase):
    """rad0009: the engine starts from a serving manifest; thin hooks into core/bc_serving_manifest.*."""

    def _applied(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        shutil.copytree(FIXTURE, root, dirs_exist_ok=True)
        self.assertTrue(all(result.ok for result in apply_all(list(_load(RAD0009)), root)))
        return temp, root

    def test_main_edits_are_thin_calls_into_the_extension_point(self):
        patches = _load(RAD0009)
        # config.cpp (the option table) is deliberately not touched: --manifest is handled before config_parse
        self.assertEqual({p.path for p in patches if not p.create}, {"core/main.cpp"})
        self.assertEqual(sum(1 for p in patches if p.create), 2)
        temp, root = self._applied()
        try:
            added = TestRadiancePatchPackages._added_lines(
                (FIXTURE / "core/main.cpp").read_text(encoding="utf-8"),
                (root / "core/main.cpp").read_text(encoding="utf-8"),
            )
            code = [line.strip() for line in added if line.strip() and not line.strip().startswith(("/*", "*"))]
            self.assertEqual(
                code,
                [
                    '#include "bc_serving_manifest.h"',
                    "if (!bc_serving_engine_argv(&argc, &argv)) return 2;",
                    "if (!bc_serving_mark_given(&cfg.given)) return 2;",
                ],
            )
        finally:
            temp.cleanup()

    def test_manifest_is_loaded_before_config_parse_and_marked_after_it(self):
        temp, root = self._applied()
        try:
            main = (root / "core/main.cpp").read_text(encoding="utf-8")
            argv = main.index("bc_serving_engine_argv(&argc, &argv)")
            parse = main.index("rad::config_parse(argc, argv, &cfg)")
            usage_exit = main.index("if (s < 0)  { rad::config_usage(argv[0]); return 2; }")
            mark = main.index("if (!bc_serving_mark_given(&cfg.given)) return 2;")
            self.assertLess(argv, parse)
            self.assertLess(usage_exit, mark)
            self.assertNotIn("--manifest", (root / "core/main.cpp").read_text(encoding="utf-8").split("int main")[1])
        finally:
            temp.cleanup()

    def test_manifest_to_argv_logic_compiled_and_run(self):
        """Compile the patch's own bc_serving_manifest.cpp against a stub of config_option_names
         and run the card-free test."""
        compiler = shutil.which("g++") or shutil.which("c++")
        candidates = [os.environ.get("BC_NLOHMANN_INCLUDE", ""), str(REPO / "vendor" / "llama.cpp" / "vendor")]
        include = next((c for c in candidates if c and (Path(c) / "nlohmann" / "json.hpp").is_file()), None)
        if compiler is None or include is None:
            message = "needs a C++ compiler and nlohmann/json.hpp (set BC_NLOHMANN_INCLUDE to its parent dir)"
            if os.environ.get("CI"):
                self.fail(message)   # in CI the test must run, never skip
            self.skipTest(message)
        temp, root = self._applied()
        try:
            exe = Path(temp.name) / "manifest_argv_test"
            scratch = Path(temp.name) / "scratch"
            scratch.mkdir()
            test = REPO / "engines" / "radiance" / "tests" / "serving_manifest" / "manifest_argv_test.cpp"
            build = subprocess.run(
                [compiler, "-std=c++20", "-Wall", "-Werror", f"-I{root / 'core'}", f"-I{include}",
                 str(root / "core" / "bc_serving_manifest.cpp"), str(test), "-o", str(exe)],
                capture_output=True, text=True,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            # radiance's real option table (g_flags at the pin), so the C++ test uses exactly the engine's option names
            # and value shapes, not a hand-written list
            config = (FIXTURE / "core" / "config.cpp").read_text(encoding="utf-8")
            table = config[config.index("static const Flag g_flags[] = {"):]
            table = table[: table.index("\n};")]
            rows = re.findall(r'\{\s*"(--?[a-z0-9-]+)",\s*("[^"]*"|nullptr)', table)
            self.assertGreater(len(rows), 100)
            options = Path(scratch) / "options.tsv"
            # config_option_names() leaves out --help, as the engine does (core/config.cpp)
            options.write_text("".join(f"{n}\t{'-' if a == 'nullptr' else a.strip(chr(34))}\n"
                                       for n, a in rows if n != "--help"))
            run = subprocess.run([str(exe), str(scratch), str(test.parent / "bc_serve_dry_run.json"), str(options)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("ok", run.stdout)
        finally:
            temp.cleanup()


class TestPlanMemoryPatch(unittest.TestCase):
    """rad0011: the engine's memory plan as a serving manifest v1 EnginePlan: `--plan-memory FILE` measures it,
    `--plan-check FILE --plan-check-floor-ppm N` checks it at start-up; thin hooks into core/bc_plan_*.* and a fail-closed
    link of the shared bc_serving_v1 library into rad_core."""

    SERVING = REPO / "engines" / "radiance" / "components" / "serving"

    def _applied(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        shutil.copytree(FIXTURE, root, dirs_exist_ok=True)
        self.assertTrue(all(result.ok for result in apply_all(list(_load(RAD0009)) + list(_load(RAD0011)), root)))
        return temp, root

    def _added_code(self, root, path):
        added = TestRadiancePatchPackages._added_lines(
            (FIXTURE / path).read_text(encoding="utf-8"), (root / path).read_text(encoding="utf-8"))
        return [line.strip() for line in added if line.strip() and not line.strip().startswith(("/*", "*", "#"))]

    def test_upstream_edits_are_thin_calls_into_the_extension_point(self):
        patches = _load(RAD0011)
        self.assertEqual({p.path for p in patches if not p.create},
                         {"core/main.cpp", "core/engine_bringup.cpp", "core/CMakeLists.txt"})
        self.assertEqual({p.path for p in patches if p.create},
                         {"core/bc_plan_memory.h", "core/bc_plan_memory.cpp", "core/bc_plan_check.h",
                          "core/bc_plan_check.cpp"})
        temp, root = self._applied()
        try:
            self.assertEqual(self._added_code(root, "core/main.cpp"), [
                "if (int rc = bc_plan_memory_take_flag(&argc, &argv)) return rc;",
                "if (!bc_serving_engine_argv(&argc, &argv)) return 2;",
                "if (!bc_serving_mark_given(&cfg.given)) return 2;",
            ])
            self.assertEqual(self._added_code(root, "core/engine_bringup.cpp"), [
                "bc_plan_memory_note(cfg_, facts, vb, emb_bytes, wfs, ngram_bytes, ck_slot, host_arenas, kv_floor, ctx, "
                "ranks_[0]->program);",
                "RAD_TRY(bc_plan_memory_done());",
            ])
        finally:
            temp.cleanup()

    def test_flag_is_taken_before_the_manifest_reader_and_the_check_precedes_the_pools(self):
        temp, root = self._applied()
        try:
            main = (root / "core/main.cpp").read_text(encoding="utf-8")
            self.assertLess(main.index("bc_plan_memory_take_flag(&argc, &argv)"),
                            main.index("bc_serving_engine_argv(&argc, &argv)"))
            bringup = (root / "core/engine_bringup.cpp").read_text(encoding="utf-8")
            self.assertIn("    RAD_TRY(plan());\n    RAD_TRY(bc_plan_memory_done());\n    RAD_TRY(load_weights());", bringup)
            resolve = bringup.index("RAD_TRY(vram_budget_resolve(cfg_, facts, &vb));")
            note = bringup.index("bc_plan_memory_note(cfg_, facts, vb,")
            self.assertLess(resolve, note)
            # RSC08 S4: the live plan is checked right after the VRAM split, before any pool is configured
            self.assertLess(note, bringup.index("r->pools.configure("))
            self.assertLess(note, bringup.index("int Engine::start()"))
        finally:
            temp.cleanup()

    def test_the_core_link_is_added_and_fails_closed(self):
        temp, root = self._applied()
        try:
            cmake = (root / "core/CMakeLists.txt").read_text(encoding="utf-8")
            guard = cmake.index("if(NOT DEFINED BC_SERVING_V1_CMAKE)")
            self.assertLess(cmake.index("target_link_libraries(rad_core PUBLIC rad_abi"), guard)
            self.assertIn("message(FATAL_ERROR", cmake[guard:guard + 400])
            self.assertLess(guard, cmake.index('include("${BC_SERVING_V1_CMAKE}")'))
            self.assertLess(cmake.index('include("${BC_SERVING_V1_CMAKE}")'),
                            cmake.index("target_link_libraries(rad_core PUBLIC bc_serving_v1)"))
            self.assertEqual(self._added_code(root, "core/CMakeLists.txt"), [
                "if(NOT DEFINED BC_SERVING_V1_CMAKE)",
                'message(FATAL_ERROR "rad0011: -DBC_SERVING_V1_CMAKE=<bigcherry>/engines/radiance/components/serving/bc_serving_v1.cmake "',
                '"is required (the shared serving manifest v1 library); the BigCherry lane passes it")',
                "endif()",
                'set(BC_SERVING_V1_NLOHMANN_INCLUDE "${CMAKE_SOURCE_DIR}/vendor")',
                'include("${BC_SERVING_V1_CMAKE}")',
                "target_link_libraries(rad_core PUBLIC bc_serving_v1)",
            ])
        finally:
            temp.cleanup()

    def test_both_builds_compile_one_source_list(self):
        """The serving component and radiance's core include the same bc_serving_v1.cmake: the library is defined there
        and nowhere else, and the lane hands the engine that file."""
        for cmake in (self.SERVING / "CMakeLists.txt", *self.SERVING.glob("*.cmake")):
            text = cmake.read_text(encoding="utf-8")
            if cmake.name == "bc_serving_v1.cmake":
                self.assertEqual(text.count("add_library(bc_serving_v1 "), 1)
                self.assertEqual(text.count("add_library(bc_jcs "), 1)
            else:
                self.assertNotIn("add_library(bc_serving_v1", text, cmake.name)
                self.assertNotIn("add_library(bc_jcs", text, cmake.name)
        self.assertIn('include("${CMAKE_CURRENT_SOURCE_DIR}/bc_serving_v1.cmake")',
                      (self.SERVING / "CMakeLists.txt").read_text(encoding="utf-8"))
        engine = tomllib.loads((REPO / "engines" / "radiance" / "engine.toml").read_text(encoding="utf-8"))
        self.assertEqual(engine["build"]["repo-paths"]["BC_SERVING_V1_CMAKE"],
                         "engines/radiance/components/serving/bc_serving_v1.cmake")
        listed = re.findall(r'"\$\{_bc_v1_dir\}/([^"]+\.cpp)"', (self.SERVING / "bc_serving_v1.cmake").read_text(encoding="utf-8"))
        self.assertTrue(listed and all((self.SERVING / f).is_file() for f in listed), listed)

    def test_second_apply_is_a_no_op(self):
        temp, root = self._applied()
        try:
            names = ("core/main.cpp", "core/engine_bringup.cpp", "core/CMakeLists.txt")
            before = {p: (root / p).read_text(encoding="utf-8") for p in names}
            self.assertTrue(all(r.ok for r in apply_all(list(_load(RAD0009)) + list(_load(RAD0011)), root)))
            self.assertEqual(before, {p: (root / p).read_text(encoding="utf-8") for p in before})
        finally:
            temp.cleanup()

    def test_without_rad0009_it_fails_closed_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copytree(FIXTURE, root, dirs_exist_ok=True)
            results = apply_all(list(_load(RAD0011)), root)
            self.assertFalse(all(r.ok for r in results))
            self.assertEqual((root / "core/main.cpp").read_text(encoding="utf-8"),
                             (FIXTURE / "core/main.cpp").read_text(encoding="utf-8"))
            self.assertFalse((root / "core/bc_plan_memory.cpp").exists())

    def test_plan_flags_compiled_and_run(self):
        """Compile the patch's own bc_plan_check.cpp with the shared v1 sources, under radiance's own warning flags
        (-Wall -Wextra, C++20) plus -Werror, and run its card-free test: the flags, the expected plan read, the one
        start-up line and its refusals."""
        compiler = shutil.which("g++") or shutil.which("c++")
        candidates = [os.environ.get("BC_NLOHMANN_INCLUDE", ""), str(REPO / "vendor" / "llama.cpp" / "vendor")]
        include = next((c for c in candidates if c and (Path(c) / "nlohmann" / "json.hpp").is_file()), None)
        if compiler is None or include is None:
            message = "needs a C++ compiler and nlohmann/json.hpp (set BC_NLOHMANN_INCLUDE to its parent dir)"
            if os.environ.get("CI"):
                self.fail(message)   # in CI the test must run, never skip
            self.skipTest(message)
        temp, root = self._applied()
        try:
            work = Path(temp.name)
            flags = ["-std=c++20", "-O0", "-Wall", "-Wextra", "-Wno-unused-parameter", "-Wno-missing-field-initializers",
                     "-Werror", f"-I{root / 'core'}", f"-I{root / 'abi'}", f"-I{self.SERVING}", f"-I{self.SERVING / 'v1'}",
                     f"-I{self.SERVING / 'generated'}", f"-I{include}"]
            sources = [root / "core" / "bc_plan_check.cpp",
                       REPO / "engines" / "radiance" / "tests" / "plan_memory" / "plan_flags_test.cpp",
                       *(self.SERVING / "v1" / n for n in ("error.cpp", "strict_json.cpp", "manifest.cpp", "admission.cpp")),
                       self.SERVING / "generated" / "serving_codec.cpp", self.SERVING / "jcs.cpp", self.SERVING / "sha256.cpp"]
            objects = []
            procs = []
            for index, source in enumerate(sources):
                obj = work / f"{index}-{source.stem}.o"
                objects.append(obj)
                procs.append((source, subprocess.Popen([compiler, *flags, "-c", str(source), "-o", str(obj)],
                                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
            for source, proc in procs:
                _, err = proc.communicate()
                self.assertEqual(proc.returncode, 0, f"{source.name}: {err}")
            exe = work / "plan_flags_test"
            link = subprocess.run([compiler, *map(str, objects), "-o", str(exe)], capture_output=True, text=True)
            self.assertEqual(link.returncode, 0, link.stderr)
            vectors = REPO / "docs" / "design" / "serving-manifest-v1" / "admission-vectors.json"
            scratch = work / "scratch"
            scratch.mkdir()
            run = subprocess.run([str(exe), str(vectors), str(scratch)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("ok", run.stdout)
        finally:
            temp.cleanup()


