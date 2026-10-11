"""Offline mechanics tests for 1360_ar_transfer_stream, composed after production plus its declared dependencies."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.core import paths  # noqa: E402
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch.pinned_source import pinned_checkout, production_order, production_source  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_HAVE_VENDOR = paths.llama_root().is_dir()
_V = pinned_checkout() if _HAVE_VENDOR else paths.llama_root()
_FILES = ("ggml/src/ggml-cuda/ggml-cuda.cu",)


def _load(pid: str):
    spec = importlib.util.spec_from_file_location(
        "patch_" + pid, _REPO / "engines" / "llamacpp" / "patches" / pid / "patch.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load("1360_ar_transfer_stream")


@unittest.skipUnless(all((_V / p).exists() for p in _FILES), "pinned vendor checkout not present")
class Patch1360Mechanics(unittest.TestCase):
    def _composed_base(self, td):
        """Production source plus 1360's dependency closure, with 1360 itself left for the test."""
        ids = production_order()
        for required in ("0840_hybrid_allreduce_dispatch", "0910_feature_sets", "1291_ar_cpu_root"):
            self.assertIn(required, ids)
        return production_source(_FILES, td, without="1360_ar_transfer_stream")

    def test_applies_after_composed_base_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._composed_base(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])

            src = (root / _FILES[0]).read_text(encoding="utf-8")
            self.assertEqual(src.count("BIGCHERRY_PATCH_HIT patch=1360_ar_transfer_stream"), 1)
            self.assertIn("static bool bc_1360_ar_transfer_begin(", src)
            self.assertIn("static void bc_1360_ar_transfer_end(", src)
            self.assertIn("ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];", src)
            self.assertIn("cudaStreamCreateWithFlags(&comm_ctx->bc_ar_transfer_streams[i], cudaStreamNonBlocking)", src)
            self.assertIn("cudaEventCreateWithFlags(&comm_ctx->bc_ar_transfer_in[slot][i], cudaEventDisableTiming)", src)
            self.assertIn("cudaStreamDestroy(bc_ar_transfer_streams[i])", src)

            begin = src.index("static bool bc_1360_ar_transfer_begin(")
            pre = src.index("cudaEventRecord(state.e_in[i], cuda_ctx->stream())", begin)
            transfer_wait = src.index("cudaStreamWaitEvent(state.transfer[i], state.e_in[i], 0)", pre)
            collective = src.index("NCCL_CHECK(ncclAllReduce(", transfer_wait)
            out = src.index("cudaEventRecord(state.e_out[i], state.transfer[i])", collective)
            end = src.index("static void bc_1360_ar_transfer_end(", out)
            compute_wait = src.index("cudaStreamWaitEvent(cuda_ctx->stream(), state.e_out[i], 0)", end)
            post = src.index("to_fp32(state.tmp[i].get()", compute_wait)
            self.assertLess(pre, transfer_wait)
            self.assertLess(transfer_wait, collective)
            self.assertLess(collective, out)
            self.assertLess(out, compute_wait)
            self.assertLess(compute_wait, post)

            # The default-off entry branch precedes, rather than replaces, the original stock body.
            entry = src.index("static bool ggml_backend_cuda_comm_allreduce_nccl(")
            flag = src.index('getenv("BIGCHERRY_AR_TRANSFER_STREAM")', entry)
            stock = src.index("    const int64_t ne = ggml_nelements(tensors[0]);", flag)
            self.assertLess(flag, stock)
            self.assertIn(
                "NCCL_CHECK(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, "
                "comm_ctx->comms[i], cuda_ctx->stream()));",
                src,
            )

            before = src
            again = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in again), [e.detail for r in again for e in r.failed])
            self.assertEqual(before, (root / _FILES[0]).read_text(encoding="utf-8"))

    def test_changed_nccl_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._composed_base(td)
            p = root / _FILES[0]
            drifted = p.read_text(encoding="utf-8").replace(
                "static bool ggml_backend_cuda_comm_allreduce_nccl(\n",
                "static bool ggml_backend_cuda_comm_allreduce_nccl_drifted(\n",
                1,
            )
            p.write_text(drifted, encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))
            self.assertEqual(drifted, p.read_text(encoding="utf-8"))

    def test_env_doc(self):
        self.assertEqual([doc.name for doc in _P.ENV_DOCS], ["BIGCHERRY_AR_TRANSFER_STREAM"])
        self.assertEqual([doc.default for doc in _P.ENV_DOCS], ["0 (off)"])


if __name__ == "__main__":
    unittest.main()
