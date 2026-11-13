"""VTP36 (rad0031-rad0033): the exact two-shot's timeline; the three-rank two-shot's pieces, staggered steps and
duplex weights; its second hop written once through the host mailbox.

Card-free. The pinned-source test applies the vtp14-mixed-tp chain up to rad0033 to radiance v1.17.0 when a checkout
with that tag is reachable (RADIANCE_SRC, or the workspace's vendor/radiance), and is skipped otherwise.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bigcherry.patcher import apply_all
from test_vtp14_n1_rank_code_selection import PIN_COMMIT, radiance_checkout

PATCHES = REPO / "engines/radiance/patches"
TL = PATCHES / "rad0031_ts_timeline"
DX = PATCHES / "rad0032_ts3_duplex"
ONCE = PATCHES / "rad0033_ts3_once"


def load(pkg: Path):
    spec = importlib.util.spec_from_file_location(pkg.name, pkg / "patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PATCHES)


def chain(last: Path = ONCE):
    with (REPO / "engines/radiance/recipes.toml").open("rb") as f:
        names = tomllib.load(f)["patch-set"]["vtp14-mixed-tp"]["patches"]
    return names[: names.index(last.name) + 1]


def texts(pkg: Path):
    return {e.id: e.text for f in load(pkg) for e in f.edits}


class Ts3AllReduceTests(unittest.TestCase):
    def test_metadata_and_recipe(self):
        for pkg, order, kind, requires in (
                (TL, 31, "diagnostic", ["rad0021_vtp14_weighted_twoshot_3rank"]),
                (DX, 32, "enhancement", ["rad0024_ts3_decode", "rad0031_ts_timeline"]),
                (ONCE, 33, "enhancement", ["rad0032_ts3_duplex"])):
            with (pkg / "patch.toml").open("rb") as f:
                meta = tomllib.load(f)
            self.assertEqual((meta["order"], meta["state"], meta["kind"]), (order, "untested", kind))
            self.assertEqual(meta["plan-ids"], ["VTP36"])
            self.assertEqual(meta["requires"], requires)
            self.assertIn("**Plan item:** VTP36", (pkg / "SUMMARY.md").read_text())
        names = chain()
        for earlier in ("rad0021_vtp14_weighted_twoshot_3rank", "rad0024_ts3_decode", TL.name):
            self.assertLess(names.index(earlier), names.index(DX.name))
        self.assertLess(names.index(DX.name), names.index(ONCE.name))

    def test_timeline_is_inert_without_the_variable(self):
        t = texts(TL)
        self.assertIn('std::getenv("R4D_AR_TIMELINE")', t["tstl-host"])
        self.assertIn("if (!tstl_dir()) return nullptr;", t["tstl-host"])
        # upstream's own per-launch timeline wins; the ring is taken only without it
        self.assertIn("timeline ? nullptr : tstl_ring(seq_ctrs, (int)rank, WS)", t["tstl-launch"])
        self.assertIn("if (!tl && tlring && b < R4D_TSTL_BLOCKS", t["tstl-kstart"])
        # the entry comes from the device sequence number, not a host counter (replay-safe)
        self.assertIn("(s_seq % R4D_TSTL_RING)", t["tstl-kstart"])
        # one ring file name per rank, apart from rad0023's one-shot rings
        self.assertIn("ar-ts-r%d-ws%d-p%d-i%zu.bin", t["tstl-host"])

    def test_axes_default_to_the_current_launch(self):
        t = texts(DX)
        for tag in ("ar", "ts4lane", "lane"):
            axes = t[f"ts3p-axes-{tag}"]
            self.assertIn('{ "ts_pieces", kArTsPieces, NELEM(kArTsPieces), 1,', axes)
            self.assertIn('{ "ts_duplex", kArTsDuplex, NELEM(kArTsDuplex), 0,', axes)
            self.assertIn('{ "ts_stagger", kArTsDuplex, NELEM(kArTsDuplex), 0,', axes)
            self.assertIn('{ "ts_pieces_min", kArTsPiecesMin, NELEM(kArTsPiecesMin), 0,', axes)
        self.assertIn("kArTsPieces[] = { 1, 2, 4, 8 };", t["ts3p-values"])
        self.assertIn("kArTsDuplex[] = { 0, 1 };", t["ts3p-values"])
        launch = t["ts3p-launch"]
        # the duplex table only where the settled weights apply and its shard pair fits the slot
        self.assertIn("if (weights && c.in->shard_wd_ok && r4d_cfg_ar_ts_duplex(a) != 0 &&", launch)
        self.assertIn("c.in->shard_wd, off) * 16 <= c.in->slot_bytes)", launch)
        # pieces from the instance's piece floor; below it one piece, as now
        self.assertIn("r4d_cfg_ar_ts_pieces_min(a) > 0 ? r4d_cfg_ar_ts_pieces_min(a) : (long)in->pieces_min;", launch)
        self.assertIn("c.numel >= ts_floor ? (long)r4d_cfg_ar_ts_pieces(a) : 1L;", launch)
        self.assertIn("kArTsPiecesMin[] = { 0, 524288, 1048576, 2097152, 4194304, 8388608 };", t["ts3p-values"])
        # the staggered order means something only in pieces, and only odd blocks take it
        self.assertIn("if (ts_pieces > 1 && r4d_cfg_ar_ts_stagger(a) != 0) ts_opts |= R4D_TS_OPT_STAGGER;", launch)
        self.assertIn("(ts_opts & R4D_TS_OPT_STAGGER) != 0 && (b & 1) != 0;", t["ts3p-kstep"])
        # no model, card or world constant beyond the three-rank row this serves
        self.assertIn("r4d_ar_weights_duplex(gbs, world, wd)", t["ts3p-settle"])
        self.assertNotIn("64,64,0", t["ts3p-settle"])

    def test_write_once_is_host_mailbox_only_and_off_by_default(self):
        t = texts(ONCE)
        for tag in ("ar", "ts4lane", "lane"):
            self.assertIn('{ "ts_once", kArTsDuplex, NELEM(kArTsDuplex), 0,', t[f"ts3o-axis-{tag}"])
        self.assertIn("if (in->host && r4d_cfg_ar_ts_once(a) != 0) ts_opts |= R4D_TS_OPT_ONCE;", t["ts3o-launch"])
        # the writer's own slot in its own scratch; the reader takes the owner's scratch at the same slot index
        self.assertIn("ts_self + ((size_t)parity * WS + rank) * (size_t)slot_stride16 + sh_max;", t["ts3o-kself"])
        # never a pointer derived from the restrict my_base: that made every fold load wait for pending stores
        self.assertNotIn("const_cast", t["ts3o-kself"])
        self.assertIn("(int4*)my_scratch /* BC_VTP36_TS3O_LAUNCH_ARG */)", t["ts3o-launch-arg"])
        # the once form is a loop of its own, chosen outside it; the upstream loop keeps its text
        self.assertIn("if (ts_once) reduce2_once(j);\n    else reduce2(j);", t["ts3o-kstore"])
        self.assertEqual(t["ts3o-kstore"].count("ts_self_b[k] = o;"), 1)
        self.assertIn("(ts_once ? (const int4*)peers.scratch[p] : my_base) +", t["ts3o-kload"])
        self.assertIn("((size_t)parity * WS + owner) * (size_t)slot_stride16 + sh_max;", t["ts3o-kload"])
        # off, the per-peer copies are the upstream stores
        self.assertIn("for (int p = 0; p < NPEERS; ++p) pa[p][sh_max + k] = o;  // region B", t["ts3o-kstore"])

    def test_guards_are_unique_and_not_prefixes(self):
        mine = [e.guard for pkg in (TL, DX, ONCE) for f in load(pkg) for e in f.edits]
        self.assertEqual(len(mine), len(set(mine)))
        for a in mine:
            for b in mine:
                self.assertFalse(a != b and b.startswith(a), (a, b))
        others = [e.guard for name in chain(TL)[:-1] for f in load(PATCHES / name) for e in f.edits]
        for a in mine:
            for b in others:
                self.assertFalse(a == b or a.startswith(b) or b.startswith(a), (a, b))

    def test_applies_to_pinned_radiance_source(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("no pinned radiance checkout; set RADIANCE_SRC")
        files = [f for name in chain() for f in load(PATCHES / name)]
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            root = Path(d)
            for path in {f.path for f in files if not f.create}:
                blob = subprocess.run(["git", "-C", str(checkout), "show", f"{PIN_COMMIT}:{path}"],
                                      capture_output=True, check=True).stdout
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_bytes(blob)
            for f in files:
                result = apply_all([f], root)
                self.assertTrue(all(r.ok for r in result), result)
            for f in load(ONCE):
                text = (root / f.path).read_text()
                for e in f.edits:
                    self.assertIn(e.text, text, e.id)
            kernel = (root / "libr4d/r4d_ar_twoshot_Nrank_exact.hip").read_text()
            entry = (root / "libr4d/r4d_ar_entry.hip").read_text()
            rows = (root / "libr4d/r4d_rows.cpp").read_text()
            # the kernel body, its parameter macro and its one launch macro all carry the ring
            self.assertEqual(kernel.count("long long* tl_all, long long* tlring, int ts_opts,"), 1)
            self.assertEqual(kernel.count("pieces_arg, tl_all, tlring, ts_opts, ts_self\n"), 1)
            self.assertEqual(kernel.count("(long long*)timeline, tlring,"), 1)
            self.assertEqual(kernel.count("(int)ts_opts /* BC_VTP36_TS3P_LAUNCH_ARG */,"), 1)
            # the split form keeps upstream's reduce; the lockstep form's two orders take the chooser
            self.assertEqual(kernel.count("ts_reduce(j);"), 2)
            self.assertEqual(kernel.count("      reduce2(j);\n"), 1)
            for guard in ("BC_VTP36_TSTL_KSTART", "BC_VTP36_TSTL_HOST", "BC_VTP36_TS3P_KSTEP", "BC_VTP36_TS3O_KSTORE",
                          "BC_VTP36_TS3O_KLOAD"):
                self.assertEqual(kernel.count(guard), 1, guard)
            # rad0021's wrapper is kept (its launch stays one piece); the entry calls the pieced one
            self.assertEqual(kernel.count("void r4d_ar_twoshot_3rank_exact_w("), 1)
            self.assertEqual(kernel.count("void r4d_ar_twoshot_3rank_exact_wp("), 1)
            self.assertEqual(entry.count("r4d_ar_twoshot_3rank_exact_wp(weights, ts_pieces, ts_opts,"), 1)
            self.assertNotIn("r4d_ar_twoshot_3rank_exact_w(weights,", entry)
            # every all-reduce tunable table that declares ts_min declares the two axes
            self.assertEqual(rows.count('{ "ts_pieces",'), rows.count('{ "ts_min",'))
            for axis in ("ts_pieces_min", "ts_duplex", "ts_stagger", "ts_once"):
                self.assertEqual(rows.count('{ "%s",' % axis), rows.count('{ "ts_min",'), axis)
            for pkg in (TL, DX, ONCE):
                again = apply_all(load(pkg), root)
                self.assertTrue(all(e.status == "already-applied" for f in again for e in f.results))


if __name__ == "__main__":
    unittest.main()
