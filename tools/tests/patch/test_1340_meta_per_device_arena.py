"""Offline mechanics tests for 1340_meta_per_device_arena on the composed production source."""

from __future__ import annotations

from collections import Counter
import importlib.util
import re
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.core import paths  # noqa: E402
from bigcherry.patch.pinned_source import production_source  # noqa: E402
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_PID = "1340_meta_per_device_arena"
_META = "ggml/src/ggml-backend-meta.cpp"
_BACKEND = "ggml/src/ggml-backend.cpp"
_ALLOC = "ggml/src/ggml-alloc.c"
_ALLOC_H = "ggml/include/ggml-alloc.h"

_BC_FLAG = "ggml/src/bc-meta-arena-flag.h"
_BC_SIMPLE = "ggml/src/bc-meta-arena-simple-tensors.h"
_BC_GRAPH = "ggml/src/bc-meta-arena-graph.h"
_BC_VBUFFER = "ggml/src/bc-meta-arena-vbuffer.h"
_BC_RESERVE = "ggml/src/bc-meta-arena-reserve.h"
_BC_BIND = "ggml/src/bc-meta-arena-bind.h"
_CREATED = {_BC_FLAG, _BC_SIMPLE, _BC_GRAPH, _BC_VBUFFER, _BC_RESERVE, _BC_BIND}


def _load():
    path = _REPO / "engines/llamacpp/patches" / _PID / "patch.py"
    spec = importlib.util.spec_from_file_location("patch_1340", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MERGED = _load()
_P1339 = [p for p in _MERGED.PATCHES if getattr(p, "description", "").startswith("1339:")]
_P1340 = [p for p in _MERGED.PATCHES if getattr(p, "description", "").startswith("1340:")]
_INPUTS = sorted({p.path for p in _P1339 + _P1340 if not p.create})
_TOUCHED = (_META, _BACKEND, _ALLOC, _ALLOC_H)

_INCLUDES = {
    _BC_FLAG: '#include "bc-meta-arena-flag.h"  // BigCherry 1340 owned definitions\n\n',
    _BC_SIMPLE: '#include "bc-meta-arena-simple-tensors.h"  // BigCherry 1340 owned definitions\n\n',
    _BC_GRAPH: '#include "bc-meta-arena-graph.h"  // BigCherry 1340 owned definitions\n\n',
    _BC_VBUFFER: '#include "bc-meta-arena-vbuffer.h"  // BigCherry 1340 owned definitions\n\n',
    _BC_RESERVE: '#include "bc-meta-arena-reserve.h"  // BigCherry 1340 owned definitions\n\n',
    _BC_BIND: '#include "bc-meta-arena-bind.h"  // BigCherry 1340 owned definitions\n\n',
}


def _lines(*texts: str) -> Counter[str]:
    return Counter("".join(line.split()) for text in texts for line in text.splitlines() if line.strip())


@unittest.skipUnless(paths.llama_root().is_dir(), "pinned vendor checkout not present")
class Patch1340Mechanics(unittest.TestCase):
    def _root(self, td):
        root = production_source("llamacpp", "bigcherry", _INPUTS, Path(td), stop_before=_PID)
        base = apply_all(_P1339, root)
        self.assertTrue(all(r.ok for r in base), [e.detail for r in base for e in r.failed])
        return root

    @staticmethod
    def _read(root, path):
        return (root / path).read_text(encoding="utf-8")

    def test_apply_composed_source_owned_files_guards_identity_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            before = {p: self._read(root, p) for p in _INPUTS}

            for fp in _P1340:
                if fp.create:
                    self.assertIn(fp.path, _CREATED)
                    self.assertFalse((root / fp.path).exists())
                    continue
                for edit in fp.edits:
                    self.assertEqual(edit.expect_matches, 1, edit.id)
                    self.assertIsNotNone(edit.guard, edit.id)
                    self.assertIsNone(re.search(edit.guard_pattern(), before[fp.path], re.MULTILINE), (fp.path, edit.id))

            res = apply_all(_P1340, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])

            after = {p: self._read(root, p) for p in _TOUCHED}
            owned = {p: self._read(root, p) for p in _CREATED}
            meta, alloc = after[_META], after[_ALLOC]

            self.assertEqual(sum(text.count("\n") for text in owned.values()), 446)
            self.assertEqual(sum(1 for text in owned.values() for line in text.splitlines() if line.strip()), 403)

            definitions = {
                _BC_FLAG: ("bool ggml_backend_meta_per_device_arena_enabled() {",),
                _BC_SIMPLE: (
                    "static const std::vector<ggml_tensor *> * ggml_backend_meta_buffer_simple_tensors(",
                    "static struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(",
                ),
                _BC_GRAPH: (
                    "ggml_backend_t ggml_backend_meta_simple_backend(",
                    "static void ggml_backend_meta_arena_map_graph(",
                    "bool ggml_backend_meta_reserve_graph(",
                    "bool ggml_backend_meta_alloc_graph(",
                    "void ggml_backend_meta_rotate_graph_containers(",
                ),
                _BC_VBUFFER: (
                    "static struct vbuffer * ggml_vbuffer_alloc(",
                    "static struct vbuffer * ggml_vbuffer_alloc_grow(",
                ),
                _BC_RESERVE: (
                    "bool ggml_gallocr_reserve(",
                    "bool ggml_gallocr_reserve_grow(",
                ),
                _BC_BIND: (
                    "bool ggml_gallocr_alloc_graph_reuse(",
                    "bool ggml_gallocr_alloc_graph_reuse_from(",
                    "bool ggml_gallocr_alloc_graph(",
                ),
            }
            for path, needles in definitions.items():
                target = meta if path in {_BC_FLAG, _BC_SIMPLE, _BC_GRAPH} else alloc
                for needle in needles:
                    self.assertIn(needle, owned[path], (path, needle))
                    self.assertNotIn(needle, target, (path, needle))

            for path, include in _INCLUDES.items():
                target = meta if path in {_BC_FLAG, _BC_SIMPLE, _BC_GRAPH} else alloc
                self.assertEqual(target.count(include.strip()), 1)

            self.assertIn("bufs.resize(n_simple_bufts, nullptr);", meta)
            self.assertIn("std::vector<arena_plan_t>            arena_plans; // BigCherry 1340 (MSM02)", meta)
            self.assertIn("bc.arena_plans.clear();", meta)
            self.assertIn("bc_gallocr_replan_reasons[3]++;", alloc)
            self.assertNotIn("static bool ggml_gallocr_needs_realloc(", alloc)
            self.assertIn("GGML_API bool ggml_gallocr_reserve_grow(", after[_ALLOC_H])
            self.assertIn("ggml_backend_meta_alloc_graph(sched->backends[i], &sched->graph)", after[_BACKEND])

            graph = owned[_BC_GRAPH]
            self.assertIn("bc.arena_plans[i_plan].out_sig != out_sig", graph)
            self.assertEqual(graph.count("out_sig = ggml_backend_meta_arena_out_sig(simple_graph);"), 2)
            self.assertIn("ret->flags |= (t->flags & GGML_TENSOR_FLAG_OUTPUT);", graph)
            self.assertIn("ggml_gallocr_reserve_grow(bcj.arena_galloc.get(), &simple_graph)", graph)
            self.assertIn("ggml_gallocr_alloc_graph_reuse_from(plan, bcj.arena_galloc.get(), &simple_graph)", graph)

            i = meta.index('#include "bc-meta-arena-simple-tensors.h"')
            self.assertGreater(meta.find("static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(", i), i)
            i = meta.index('#include "bc-meta-arena-flag.h"')
            self.assertGreater(meta.find("static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(", i), i)
            i = alloc.index('#include "bc-meta-arena-vbuffer.h"')
            self.assertGreater(alloc.find("static void ggml_vbuffer_tensor_alloc(", i), i)
            i = alloc.index('#include "bc-meta-arena-reserve.h"')
            self.assertGreater(alloc.find("static void ggml_gallocr_init_tensor(", i), i)
            i = alloc.index('#include "bc-meta-arena-bind.h"')
            self.assertGreater(alloc.find("size_t ggml_gallocr_get_buffer_size(", i), i)

            legacy = dict(after)
            for path in (_BC_FLAG, _BC_SIMPLE, _BC_GRAPH):
                legacy[_META] = legacy[_META].replace(_INCLUDES[path], owned[path], 1)
            for path in (_BC_VBUFFER, _BC_RESERVE, _BC_BIND):
                legacy[_ALLOC] = legacy[_ALLOC].replace(_INCLUDES[path], owned[path], 1)
            new_lines = _lines(*(after[p] for p in _TOUCHED), *(owned[p] for p in sorted(_CREATED)))
            legacy_lines = _lines(*(legacy[p] for p in _TOUCHED))
            include_lines = _lines(*_INCLUDES.values())
            self.assertEqual(new_lines, legacy_lines + include_lines)
            self.assertEqual(sum(new_lines.values()), sum(legacy_lines.values()) + 6)

            texts = dict(after)
            texts.update(owned)
            for fp in _P1340:
                for edit in fp.edits:
                    self.assertIsNotNone(re.search(edit.guard_pattern(), texts[fp.path], re.MULTILINE), (fp.path, edit.id))

            snapshot = {p: self._read(root, p) for p in _INPUTS + sorted(_CREATED)}
            second = apply_all(_P1340, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(snapshot, {p: self._read(root, p) for p in snapshot})

    def test_missing_owned_block_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            path = root / _ALLOC
            source = self._read(root, _ALLOC)
            anchor = "static struct vbuffer * ggml_vbuffer_alloc(ggml_backend_buffer_type_t buft,"
            self.assertEqual(source.count(anchor), 1)
            drifted = source.replace(anchor, anchor.replace("ggml_vbuffer_alloc", "ggml_vbuffer_alloc_drifted"), 1)
            path.write_text(drifted, encoding="utf-8", newline="\n")
            snapshot = {p: self._read(root, p) for p in _INPUTS}

            res = apply_all(_P1340, root)
            self.assertFalse(all(r.ok for r in res))
            self.assertEqual(snapshot, {p: self._read(root, p) for p in _INPUTS})
            for created in _CREATED:
                self.assertFalse((root / created).exists())

    def test_production_set_and_env_docs_unchanged(self):
        with (_REPO / "config/recipes.toml").open("rb") as handle:
            recipes = tomllib.load(handle)
        production = recipes["patch-set"]["validated-enhancements"]["patches"]
        self.assertNotIn("1339_meta_memory_report", production)
        self.assertIn(_PID, production)
        self.assertNotIn("meta-memory", recipes["experiment"])
        self.assertEqual(
            [doc.name for doc in _MERGED.DOCUMENTED_ENV_DOCS],
            ["BIGCHERRY_META_MEM", "BIGCHERRY_META_PER_DEVICE_ARENA"],
        )


if __name__ == "__main__":
    unittest.main()
