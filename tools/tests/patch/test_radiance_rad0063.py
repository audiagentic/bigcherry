"""rad0063: the vocabulary-head builder cuts the head for any rank count (padding with dropped rows)."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bigcherry.patcher import apply_all  # noqa: E402
from _vtp_promotion_support import CANDIDATE_COMMIT, load_patch_files  # noqa: E402
from test_vtp14_n1_rank_code_selection import radiance_checkout  # noqa: E402

PACKAGE = "rad0063_vhead_world_ranks"
WIDTHS = [4, 5, 6, 7]
K = 16


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_source(src: Path, mod, V: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.integers(0, 65535, size=(V, K), dtype=np.uint16)
    src.mkdir()
    mod.st_write(str(src / "model.safetensors"), [("lm_head.weight", W, "BF16")])
    (src / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {"lm_head.weight": "model.safetensors"}}))
    return W


def write_classes(path: Path, cls: np.ndarray):
    V = cls.size
    padded = np.zeros(((V + 3) // 4) * 4, dtype=np.uint8)
    padded[:V] = cls
    packed = sum((padded[j::4].astype(np.uint8) << (2 * j)) for j in range(4)).astype(np.uint8)
    path.write_bytes(f"radiance-vhead-1 n_vocab={V} widths={','.join(map(str, WIDTHS))}\n".encode() + packed.tobytes())


def read_head(dst: Path, mod):
    out = {}
    path = dst / "vhead.safetensors"
    raw = path.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    hdr = json.loads(raw[8:8 + n])
    base = 8 + n
    for name, meta in hdr.items():
        a, b = meta["data_offsets"]
        dt = np.int32 if meta["dtype"] == "I32" else np.uint16
        out[name] = np.frombuffer(raw[base + a:base + b], dtype=dt).reshape(meta["shape"])
    return out


class VheadWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkout = radiance_checkout()
        if checkout is None:
            raise unittest.SkipTest("set RADIANCE_SRC to a checkout with the pinned Radiance commit")
        cls.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        cls.root = Path(cls.tmp.name) / "radiance"
        (cls.root / "tools").mkdir(parents=True)
        shown = subprocess.check_output(["git", "-C", str(checkout), "show", f"{CANDIDATE_COMMIT}:tools/vhead.py"])
        (cls.root / "tools/vhead.py").write_bytes(shown)
        (cls.root / "tools/vhead_orig.py").write_bytes(shown)
        cls.files = load_patch_files(PACKAGE)
        cls.first = apply_all(cls.files, cls.root)
        cls.orig = load_module(cls.root / "tools/vhead_orig.py", "vhead_orig")
        cls.new = load_module(cls.root / "tools/vhead.py", "vhead_new")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_tool(self, mod, src, dst, cpath, *args):
        argv = ["vhead.py", *args, str(src), str(dst), str(cpath)]
        old = sys.argv
        sys.argv = argv
        try:
            mod.main()
        finally:
            sys.argv = old

    def test_patch_applies_once_and_is_idempotent_and_fails_closed(self):
        self.assertTrue(self.first and all(r.ok for r in self.first), self.first)
        second = apply_all(self.files, self.root)
        self.assertTrue(all(r.ok for r in second), second)
        self.assertEqual((self.root / "tools/vhead.py").read_text().count("def rank_shares("), 1)
        broken = Path(self.tmp.name) / "broken"
        (broken / "tools").mkdir(parents=True)
        text = (self.root / "tools/vhead_orig.py").read_text().replace("half = V // 2", "half = V // 3")
        (broken / "tools/vhead.py").write_text(text)
        res = apply_all(self.files, broken)
        self.assertFalse(all(r.ok for r in res))

    def test_shares_follow_the_engine_rule(self):
        self.assertEqual(self.new.rank_shares(248320, 3), [(0, 82816), (82816, 165632), (165632, 248320)])
        self.assertEqual([e - b for b, e in self.new.rank_shares(248320, 3)], [82816, 82816, 82688])
        self.assertEqual(self.new.rank_shares(248320, 2), [(0, 124160), (124160, 248320)])
        self.assertEqual(self.new.vocab_per_rank(248320, 4), 62080)

    def test_world_two_is_byte_identical_to_the_old_tool(self):
        V = 2048
        half = V // 2
        base = Path(self.tmp.name) / "w2"
        base.mkdir()
        probe = write_source(base / "src", self.new, V, 1)
        # equal counts per class in each half, each a multiple of 256: 256-row runs of one class each
        order = np.array([0, 1, 2, 3])
        rng = np.random.default_rng(5)
        halves = []
        for _ in range(2):
            runs = rng.permutation(np.repeat(order, [1, 1, 1, 1]))
            blocks = np.repeat(np.concatenate([runs, rng.permutation(np.repeat(order, 1))]), 256)
            halves.append(rng.permutation(blocks[:half]))
        cls = np.concatenate(halves)
        # force the halves to the same class counts: take the first half's multiset for both
        cls = np.concatenate([halves[0], rng.permutation(halves[0])])
        self.assertEqual([int((cls[:half] == i).sum()) % 256 for i in range(4)], [0, 0, 0, 0])
        write_classes(base / "c.cls", cls)
        self.run_tool(self.orig, base / "src", base / "old", base / "c.cls")
        self.run_tool(self.new, base / "src", base / "new", base / "c.cls")
        self.assertEqual((base / "old/vhead.safetensors").read_bytes(), (base / "new/vhead.safetensors").read_bytes())
        self.assertEqual((base / "old/model.safetensors.index.json").read_text(),
                         (base / "new/model.safetensors.index.json").read_text())
        del probe

    def test_world_three_slices_are_equal_padded_in_share_and_consistent(self):
        V = 2560            # not divisible by 3: shares 896, 896, 768 (128-row units)
        world = 3
        base = Path(self.tmp.name) / "w3"
        base.mkdir()
        W = write_source(base / "src", self.new, V, 2)
        rng = np.random.default_rng(9)
        cls = rng.choice(4, size=V, p=[0.5, 0.15, 0.15, 0.2]).astype(np.int64)
        write_classes(base / "c.cls", cls)
        self.run_tool(self.new, base / "src", base / "dst", base / "c.cls", "--world", str(world))
        head = read_head(base / "dst", self.new)
        shares = self.new.rank_shares(V, world)
        vrow = head["lm_head.vrow"][:, 0]
        seen = np.zeros(V, dtype=int)
        for i, k in enumerate(WIDTHS):
            ids = head[f"lm_head.k{k}.ids"][:, 0].astype(np.int64)
            w = head[f"lm_head.k{k}.weight"]
            self.assertEqual(ids.size % (256 * world), 0, k)
            n = ids.size // world
            for r, (b, e) in enumerate(shares):
                part = ids[r * n:(r + 1) * n]
                real = part[part >= 0]
                self.assertTrue(((real >= b) & (real < e)).all())
                self.assertTrue((np.diff(real) > 0).all())                       # ascending id
                self.assertTrue((part[real.size:] == -1).all())                   # pads trail, outside every share
                self.assertTrue((cls[real] == i).all())
                self.assertTrue(np.array_equal(w[r * n:r * n + real.size], W[real]))
                pad_w = w[r * n + real.size:(r + 1) * n]
                own = W[real] if real.size else W[b:b + 1]
                for row in pad_w:                                                # pad weights repeat this rank's rows
                    self.assertTrue((own == row).all(axis=1).any())
                seen[real] += 1
                # the engine's direct store: id - off as unsigned below the share keeps exactly the real rows
                off, share = b, e - b
                kept = ((part.astype(np.int64) - off) & 0xFFFFFFFF) < share
                self.assertTrue(np.array_equal(kept, part >= 0))
        self.assertTrue((seen == 1).all())
        # the staging gather: vrow[v] gives (class width, rank-major row); this rank reads row - rank * rows
        for r, (b, e) in enumerate(shares):
            for v in range(e - b):
                word = int(vrow[b + v]) & 0xFFFFFFFF
                k, row = word >> 28, word & 0x0FFFFFFF
                ids = head[f"lm_head.k{k}.ids"][:, 0]
                n = ids.size // world
                j = row - r * n
                self.assertTrue(0 <= j < n)
                self.assertEqual(int(ids[r * n + j]), b + v)
        # the logit of every id from either path equals the dense row's dot product
        x = np.random.default_rng(3).integers(-3, 4, size=K)
        dense = W.astype(np.int64) @ x
        direct = np.full(V, -1, dtype=np.int64)
        for r, (b, e) in enumerate(shares):
            for k in WIDTHS:
                ids = head[f"lm_head.k{k}.ids"][:, 0].astype(np.int64)
                w = head[f"lm_head.k{k}.weight"].astype(np.int64)
                n = ids.size // world
                for t in range(n):
                    v = (ids[r * n + t] - b) & 0xFFFFFFFF
                    if v < e - b:
                        direct[b + v] = int(w[r * n + t] @ x)
        self.assertTrue(np.array_equal(direct, dense))

    def test_world_one_and_empty_class_slice(self):
        V = 640
        base = Path(self.tmp.name) / "w1"
        base.mkdir()
        write_source(base / "src", self.new, V, 4)
        cls = np.zeros(V, dtype=np.int64)
        cls[:3] = 1                                    # a class held by one rank only
        write_classes(base / "c.cls", cls)
        self.run_tool(self.new, base / "src", base / "dst", base / "c.cls", "--world", "2")
        head = read_head(base / "dst", self.new)
        ids = head["lm_head.k5.ids"][:, 0]
        self.assertEqual(ids.size, 2 * 256)
        self.assertEqual(sorted(ids[ids >= 0].tolist()), [0, 1, 2])
        self.assertTrue((ids[256:] == -1).all())      # the second rank holds none: all pad


if __name__ == "__main__":
    unittest.main()
