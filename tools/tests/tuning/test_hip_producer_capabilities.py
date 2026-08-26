"""HI121 M1b: tests for hip_capabilities.py's registry and source-owned
declaration reader."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning.capabilities import CapabilityMask128  # noqa: E402
from bigcherry.tuning import hip_capabilities as hc  # noqa: E402


def _write_types_h(tmp_path: Path, body: str) -> Path:
    root = tmp_path / "vendor" / "llama.cpp"
    types_dir = root / "ggml" / "src" / "ggml-cuda"
    types_dir.mkdir(parents=True, exist_ok=True)
    (types_dir / "hip-autotune-types.h").write_text(body, encoding="utf-8")
    return root


_REAL_DECLARATION = (
    "#define GGML_HIP_PRODUCER_CAPABILITIES_LO UINT64_C(0x000000000000001f)\n"
    "#define GGML_HIP_PRODUCER_CAPABILITIES_HI UINT64_C(0x0000000000000000)\n"
)


class KnownCapabilityMaskTests(unittest.TestCase):
    def test_hip_capability_mask_combines_bits(self):
        mask = hc.hip_capability_mask([hc.HipCapability.CORE_SIGNATURE_V1, hc.HipCapability.FUSION_X_BIAS_PRESENCE_V1])
        self.assertEqual(mask.value, 0b11)

    def test_known_hip_capability_mask_matches_real_head(self):
        # Every capability bit the registry currently understands, matching
        # the real declaration added to hip-autotune-types.h this round.
        self.assertEqual(hc.known_hip_capability_mask(), CapabilityMask128.from_hex(
            "0000000000000000000000000000001f"
        ))


class LoadDeclaredProducerCapabilitiesTests(unittest.TestCase):
    def test_real_declaration_parses_to_all_five_bits(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_types_h(Path(tmp), _REAL_DECLARATION)
            mask = hc.load_declared_producer_capabilities(root)
            self.assertEqual(mask.to_hex(), "0000000000000000000000000000001f")
            self.assertTrue(mask.contains(hc.hip_capability_mask([
                hc.HipCapability.CORE_SIGNATURE_V1,
                hc.HipCapability.FUSION_X_BIAS_PRESENCE_V1,
                hc.HipCapability.FUSION_GATE_BIAS_PRESENCE_V1,
                hc.HipCapability.FUSION_X_SCALE_PRESENCE_V1,
                hc.HipCapability.FUSION_GATE_SCALE_PRESENCE_V1,
            ])))

    def test_against_the_real_repo_head(self):
        # Confirms the declaration actually added to the real
        # hip-autotune-types.h this round matches the registry exactly.
        repo_root = Path(__file__).resolve().parents[3]
        vendor_root = repo_root / "src"
        mask = hc.load_declared_producer_capabilities(vendor_root)
        self.assertEqual(mask, hc.known_hip_capability_mask())

    def test_missing_file_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(hc.HipCapabilityError):
                hc.load_declared_producer_capabilities(Path(tmp))

    def test_missing_lo_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_types_h(
                Path(tmp),
                "#define GGML_HIP_PRODUCER_CAPABILITIES_HI UINT64_C(0x0000000000000000)\n",
            )
            with self.assertRaises(hc.HipCapabilityError):
                hc.load_declared_producer_capabilities(root)

    def test_missing_hi_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_types_h(
                Path(tmp),
                "#define GGML_HIP_PRODUCER_CAPABILITIES_LO UINT64_C(0x000000000000001f)\n",
            )
            with self.assertRaises(hc.HipCapabilityError):
                hc.load_declared_producer_capabilities(root)

    def test_duplicated_declaration_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_types_h(Path(tmp), _REAL_DECLARATION + _REAL_DECLARATION)
            with self.assertRaises(hc.HipCapabilityError):
                hc.load_declared_producer_capabilities(root)

    def test_unknown_bit_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_types_h(
                Path(tmp),
                "#define GGML_HIP_PRODUCER_CAPABILITIES_LO UINT64_C(0x000000000000003f)\n"
                "#define GGML_HIP_PRODUCER_CAPABILITIES_HI UINT64_C(0x0000000000000000)\n",
            )
            with self.assertRaises(hc.HipCapabilityError):
                hc.load_declared_producer_capabilities(root)


if __name__ == "__main__":
    unittest.main()
