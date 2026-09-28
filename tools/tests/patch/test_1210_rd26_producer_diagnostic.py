"""Unit tests for 1210's validation producer diff-context diagnostic helper
(PRBE20 2026-09-28: added after two source-verified fix attempts both left
the same unexplained byte-480 mismatch, with the scratch .gguf files that
would explain it deleted before anyone could look).

Only ``_diff_context`` is tested here -- it's pure, file-local, and has no
GPU/build dependency. The producer's ``run()`` entrypoint is exercised only
by real hardware campaigns.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_REPO = Path(__file__).resolve().parents[3]
_PRODUCER_FILE = (
    _REPO
    / "patches/1210_rd26_bitidentical_decode_verify_standalone/validation/producer.py"
)
_spec = importlib.util.spec_from_file_location("rd26_producer", _PRODUCER_FILE)
assert _spec is not None and _spec.loader is not None
_producer = importlib.util.module_from_spec(_spec)
# _RunRecord is a frozen dataclass; dataclasses.wrap() resolves annotations
# via sys.modules[cls.__module__], so the module must be registered there
# before exec_module runs the class body.
sys.modules[_spec.name] = _producer
_spec.loader.exec_module(_producer)


class DiffContextTests(unittest.TestCase):
    def _write(self, td: Path, name: str, data: bytes) -> Path:
        path = td / name
        path.write_bytes(data)
        return path

    def test_reports_offset_and_window_around_a_float_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            prefix = b"\x00" * 100
            left = prefix + struct.pack("<f", 1.5) + b"\x00" * 100
            right = prefix + struct.pack("<f", 1.75) + b"\x00" * 100
            left_path = self._write(td, "left.gguf", left)
            right_path = self._write(td, "right.gguf", right)

            ctx = _producer._diff_context(left_path, right_path, offset=100, radius=8)

            self.assertEqual(ctx["offset"], 100)
            self.assertEqual(ctx["window_start"], 92)
            self.assertGreater(ctx["differing_bytes_in_window"], 0)
            # The float32 reading at the mismatch should recover the two
            # distinct values actually written, proving the interpretation
            # is aligned correctly relative to the reported offset.
            self.assertIn(1.5, [round(v, 4) for v in ctx["left_as_float32"]])
            self.assertIn(1.75, [round(v, 4) for v in ctx["right_as_float32"]])

    def test_window_clamps_at_start_of_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            left = struct.pack("<i", 7) + b"\x00" * 20
            right = struct.pack("<i", 8) + b"\x00" * 20
            left_path = self._write(td, "left.gguf", left)
            right_path = self._write(td, "right.gguf", right)

            # offset=0 with a large radius must not seek negative.
            ctx = _producer._diff_context(left_path, right_path, offset=0, radius=64)

            self.assertEqual(ctx["window_start"], 0)
            self.assertEqual(ctx["left_as_int32"][0], 7)
            self.assertEqual(ctx["right_as_int32"][0], 8)

    def test_diagnostic_artifact_name_is_declared_in_producer_manifest(self):
        manifest = (
            _REPO
            / "patches/1210_rd26_bitidentical_decode_verify_standalone/validation/producer.toml"
        ).read_text(encoding="utf-8")
        self.assertIn(_producer._DIAGNOSTIC_ARTIFACT_NAME, manifest)


if __name__ == "__main__":
    unittest.main()
