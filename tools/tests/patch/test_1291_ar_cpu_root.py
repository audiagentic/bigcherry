"""Offline mechanics tests for 1291_ar_cpu_root on the composed production source."""

from __future__ import annotations

from collections import Counter
import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.core import paths  # noqa: E402
from bigcherry.patch.pinned_source import production_source  # noqa: E402
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_PID = "1291_ar_cpu_root"
_CUDA = "ggml/src/ggml-cuda/ggml-cuda.cu"
_ARG = "common/arg.cpp"
_BENCH = "tools/llama-bench/llama-bench.cpp"
_BC = "ggml/src/ggml-cuda/bc-ar-cpu-root.cuh"
_PROVIDER = "ggml/src/ggml-cuda/bc-allreduce-provider-cli.cuh"
_INPUTS = [_CUDA, _ARG, _BENCH, _PROVIDER]


def _load():
    spec = importlib.util.spec_from_file_location("patch_1291", _REPO / "engines/llamacpp/patches" / _PID / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load()


def _lines(*texts: str) -> Counter[str]:
    return Counter("".join(line.split()) for text in texts for line in text.splitlines() if line.strip())


@unittest.skipUnless(paths.llama_root().is_dir(), "pinned vendor checkout not present")
class Patch1291Mechanics(unittest.TestCase):
    def _root(self, td):
        return production_source("llamacpp", "bigcherry", _INPUTS, Path(td), stop_before=_PID)

    @staticmethod
    def _read(root, path):
        return (root / path).read_text(encoding="utf-8")

    def test_apply_composed_source_owned_file_guards_identity_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            before = {p: self._read(root, p) for p in _INPUTS}

            for fp in _P.PATCHES:
                if fp.path == _BC:
                    continue
                source = before[fp.path]
                for edit in fp.edits:
                    self.assertIsNone(re.search(edit.guard, source, re.M), (fp.path, edit.id))

            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            cuda = self._read(root, _CUDA)
            arg = self._read(root, _ARG)
            bench = self._read(root, _BENCH)
            provider = self._read(root, _PROVIDER)
            owned = self._read(root, _BC)
            self.assertEqual(owned, _P._IMPL)
            self.assertEqual(sum(1 for line in owned.splitlines() if line.strip()), 407)

            include_text = '#include "bc-ar-cpu-root.cuh"  // BigCherry 1291 implementation\n\n'
            self.assertEqual(cuda.count('#include "bc-ar-cpu-root.cuh"'), 1)
            for definition in (
                "static __global__ void bc_cpu_root_produce(",
                "static __global__ void bc_cpu_root_consume(",
                "static void bc_cpu_root_worker(bc_cpu_root * cr)",
                "static bool ggml_backend_cuda_comm_try_allreduce_cpu_root(",
                "static void ggml_backend_cuda_comm_init_cpu_root(",
            ):
                self.assertNotIn(definition, cuda)
                self.assertIn(definition, owned)

            self.assertIn("bc_cpu_root *               cpu_root = nullptr;  // BigCherry 1291", cuda)
            self.assertIn("bc_cpu_root_free(cpu_root);  // BigCherry 1291", cuda)
            self.assertIn("ggml_backend_cuda_comm_init_cpu_root(ret);  // BigCherry 1291", cuda)
            include = cuda.index('#include "bc-ar-cpu-root.cuh"')
            comm_init = cuda.index("static void * ggml_backend_cuda_comm_init(", include)
            self.assertLess(include, comm_init)

            legacy_cuda = cuda.replace(include_text, _P._IMPL + "\n", 1)
            legacy_lines = _lines(legacy_cuda, arg, bench, provider)
            new_lines = _lines(cuda, arg, bench, provider, owned)
            include_line = _lines(include_text)
            self.assertEqual(new_lines, legacy_lines + include_line)
            self.assertEqual(sum(new_lines.values()), sum(legacy_lines.values()) + 1)

            texts = {_CUDA: cuda, _ARG: arg, _BENCH: bench, _PROVIDER: provider, _BC: owned}
            for fp in _P.PATCHES:
                for edit in fp.edits:
                    self.assertIsNotNone(re.search(edit.guard, texts[fp.path], re.M), (fp.path, edit.id))

            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(texts, {p: self._read(root, p) for p in texts})

    def test_missing_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            path = root / _CUDA
            before = self._read(root, _CUDA)
            anchor = "static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {"
            self.assertEqual(before.count(anchor), 1)
            drifted = before.replace(anchor, anchor.replace("ggml_backend_cuda_comm_init", "ggml_backend_cuda_comm_init_drifted"), 1)
            path.write_text(drifted, encoding="utf-8", newline="\n")
            snapshots = {p: self._read(root, p) for p in _INPUTS}
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))
            self.assertEqual(snapshots, {p: self._read(root, p) for p in _INPUTS})
            self.assertFalse((root / _BC).exists())


if __name__ == "__main__":
    unittest.main()
