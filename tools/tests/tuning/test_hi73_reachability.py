"""HI73: workload-shape reachability in inventory + candidate catalog.

Dispatch derives two shape facts that decide whether whole candidate families
are reachable at all (hip-autotune-dispatch.cu):

- MMQ ``fallback`` rows execute only when ``ne0[1] % 128 != 0``.
- MMF executes only for float src0 + F32 activation + even K +
  ``1 <= ncols_dst <= 16``.

The inventory records which types actually observed those shapes, and the
catalog skips candidate rows the inventory proves unreachable (restricted
profiles only). A pre-HI73 inventory (fields absent) or a partially-evaluated
one (field null) must NOT skip anything.

Run with: python -m unittest tools.tests.test_hi73_reachability
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.tuning import catalog as catalog # noqa: E402
from bigcherry.tuning import schema as schema # noqa: E402
from bigcherry import inventory  # noqa: E402
from bigcherry.inventory import Record  # noqa: E402


# ------------------------------------------------------------------ fixtures

HEADER = {
    "kind": "header",
    "source_revision": "abcdef1234567890",
    "manifest_hash": "deadbeef00112233",
    "signature_schema": 1,
    "hardware_schema": 1,
    "variant_set": "workload-max",
}

Q8_0, F32, F16 = 8, 0, 1  # ggml_type ids


def _obs(src0, ne0, ned, native="mmq:native:v1", src1=F32, flags=0):
    return {
        "kind": "observation",
        "hardware": "a" * 32,
        "signature": "b" * 32,
        "native": native,
        "canonical": {
            "op": "MUL_MAT",
            "src0_type": src0,
            "src1_type": src1,
            "dst_type": F32,
            "ne0": ne0,
            "ned": ned,
            "flags": flags,
        },
        "hardware_key": {
            "architecture_code": "gfx1201",
            "wave_size": 32,
            "compute_units": 60,
            "feature_flags": 1,
        },
        "calls": 1,
        "est_bytes": 64,
        "devices": [0],
    }


def _record(*observations):
    return Record(header=dict(HEADER), observations=list(observations))


# ------------------------------------------------------------------ inventory

class TestInventoryReachability(unittest.TestCase):
    def test_mmq_fallback_type_recorded_from_non_aligned_ne01(self):
        rec = _record(
            _obs(Q8_0, [64, 512], [64, 128]),        # 512 % 128 == 0
            _obs(Q8_0, [64, 32], [64, 32]),          # 32 % 128 != 0
        )
        inv = inventory.build_inventory(rec)
        self.assertEqual(inv["mmq_fallback_types"], ["q8_0"])

    def test_mmq_fallback_empty_when_all_shapes_aligned(self):
        rec = _record(_obs(Q8_0, [64, 512], [64, 128]))
        inv = inventory.build_inventory(rec)
        self.assertEqual(inv["mmq_fallback_types"], [])

    def test_mmf_eligible_float_op_with_f32_activation(self):
        # f16 weights, f32 activations, even K, dst width 4 (in [1, 16]).
        rec = _record(_obs(F16, [512, 256], [512, 4], native="mmf:native:v1"))
        inv = inventory.build_inventory(rec)
        self.assertEqual(inv["mmf_eligible_types"], ["f16"])

    def test_mmf_not_eligible_when_activation_not_f32(self):
        rec = _record(_obs(F16, [512, 256], [512, 4], src1=F16))
        inv = inventory.build_inventory(rec)
        self.assertEqual(inv["mmf_eligible_types"], [])

    def test_mmf_not_eligible_when_width_above_16(self):
        rec = _record(_obs(F16, [512, 256], [512, 32], native="mmf:native:v1"))
        inv = inventory.build_inventory(rec)
        self.assertEqual(inv["mmf_eligible_types"], [])

    def test_mmf_not_eligible_when_k_odd(self):
        rec = _record(_obs(F16, [511, 256], [511, 4], native="mmf:native:v1"))
        inv = inventory.build_inventory(rec)
        self.assertEqual(inv["mmf_eligible_types"], [])

    def test_mmf_uses_expert_dimension_for_mul_mat_id(self):
        # HAS_IDS = bit 3: ncols_dst = ned[2].
        rec = _record(_obs(F16, [512, 256], [512, 128, 8], flags=(1 << 3)))
        inv = inventory.build_inventory(rec)
        self.assertEqual(inv["mmf_eligible_types"], ["f16"])

    def test_missing_ne0_makes_field_null_not_empty(self):
        obs = _obs(Q8_0, [64, 32], [64, 32])
        del obs["canonical"]["ne0"]
        inv = inventory.build_inventory(_record(obs))
        self.assertIsNone(inv["mmq_fallback_types"])

    def test_missing_src1_makes_mmf_field_null(self):
        obs = _obs(F16, [512, 256], [512, 4], native="mmf:native:v1")
        del obs["canonical"]["src1_type"]
        inv = inventory.build_inventory(_record(obs))
        self.assertIsNone(inv["mmf_eligible_types"])


# ------------------------------------------------------------------ catalog

# A minimal config header with one fb0 and one fb1 row for q8_0.
_FAKE_MMQ_HEADER = """
static const mmq_config CASE_TABLE[] = {
    CASE(GGML_TYPE_Q8_0, 256, 2, 128, 8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q8_0, 256, 2, 128, 16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
};
"""

_ARCH = "gfx1201"


def _fake_cuda_dir(tmp: Path) -> Path:
    cuda = tmp / "ggml-cuda"
    cuda.mkdir(parents=True)
    header = f"mmq-config-{schema.ARCHITECTURE_FAMILY[_ARCH]}.cuh"
    (cuda / header).write_text(_FAKE_MMQ_HEADER, encoding="utf-8")
    return cuda


def _inventory(**overrides) -> catalog.Inventory:
    base: dict[str, object] = {
        "mmq_types": {"q8_0"}, "mmvq_types": {"q8_0"},
        "mmvf_types": set(), "mmf_types": set(),
        "widths": {1, 2, 4, 8}, "uses_blas": True,
    }
    base.update(overrides)
    return catalog.Inventory(**base)  # type: ignore[arg-type]


class TestCatalogReachability(unittest.TestCase):
    def _rows(self, tmp, inv):
        cuda = _fake_cuda_dir(tmp)
        rows = catalog.enumerate_mmq(cuda, [_ARCH], ["GGML_TYPE_Q8_0"], inv,
                                     restrict=True)
        return sorted(c.stable_name for c in rows)

    def test_fb1_row_skipped_when_never_reached(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            names = self._rows(tmp, _inventory(mmq_fallback_types=set()))
        fb1 = [n for n in names if ":fb1" in n]
        fb0 = [n for n in names if ":fb0" in n]
        self.assertEqual(fb1, [])
        self.assertEqual(len(fb0), 1)

    def test_fb1_row_kept_when_reached(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            names = self._rows(tmp, _inventory(mmq_fallback_types={"q8_0"}))
        self.assertEqual(len([n for n in names if ":fb1" in n]), 1)

    def test_fb1_row_kept_when_inventory_unknown(self):
        # Pre-HI73 inventory: field absent (None) -- nothing may be skipped.
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            names = self._rows(tmp, _inventory(mmq_fallback_types=None))
        self.assertEqual(len([n for n in names if ":fb1" in n]), 1)

    def test_unrestricted_profile_ignores_reachability(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            cuda = _fake_cuda_dir(tmp)
            rows = catalog.enumerate_mmq(
                cuda, [_ARCH], ["GGML_TYPE_Q8_0"],
                _inventory(mmq_fallback_types=set()), restrict=False)
        self.assertEqual(len([c for c in rows if c.config["fallback"]]), 1)

    def test_mmf_type_skipped_when_not_eligible(self):
        # f16 is observable (in the float union) but no operation met the
        # eligibility window -> every mmf:f16 candidate is skipped.
        inv = _inventory(mmf_eligible_types=set(),
                         mmvf_types={"f16"}, mmf_types={"f16"})
        rows = catalog.enumerate_mmf([_ARCH], inv, restrict=True)
        self.assertNotIn("f16", {c.config["type"] for c in rows})

    def test_mmf_type_kept_when_eligible(self):
        inv = _inventory(mmf_eligible_types={"f16"},
                         mmvf_types={"f16"}, mmf_types={"f16"})
        rows = catalog.enumerate_mmf([_ARCH], inv, restrict=True)
        self.assertIn("f16", {c.config["type"] for c in rows})

    def test_mmf_type_kept_when_inventory_unknown(self):
        inv = _inventory(mmf_eligible_types=None,
                         mmvf_types={"f16"}, mmf_types={"f16"})
        rows = catalog.enumerate_mmf([_ARCH], inv, restrict=True)
        self.assertIn("f16", {c.config["type"] for c in rows})


class TestUnreachableReporting(unittest.TestCase):
    def test_names_skipped_fb1_rows_and_reason(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            cuda = _fake_cuda_dir(tmp)
            inv = _inventory(mmq_fallback_types=set(),
                             mmf_eligible_types=set(),
                             mmvf_types={"f16"}, mmf_types={"f16"})
            out = catalog.unreachable_candidates(
                cuda, [_ARCH], ["GGML_TYPE_Q8_0"], inv)
        fb1 = [e for e in out if e["family"] == "mmq"]
        self.assertEqual(len(fb1), 1)
        self.assertIn("fb1", fb1[0]["stable_name"])
        self.assertIn("ne0[1] % 128", fb1[0]["reason"])
        mmf = [e for e in out if e["family"] == "mmf"]
        self.assertEqual(len(mmf), 1)
        self.assertIn("f16", mmf[0]["stable_name"])

    def test_nothing_reported_for_unknown_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            cuda = _fake_cuda_dir(tmp)
            inv = _inventory(mmq_fallback_types=None,
                             mmf_eligible_types=None)
            out = catalog.unreachable_candidates(
                cuda, [_ARCH], ["GGML_TYPE_Q8_0"], inv)
        self.assertEqual(out, [])

    def test_nothing_reported_for_reached_shapes(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            cuda = _fake_cuda_dir(tmp)
            inv = _inventory(mmq_fallback_types={"q8_0"},
                             mmf_eligible_types={"f16"},
                             mmvf_types={"f16"}, mmf_types={"f16"})
            out = catalog.unreachable_candidates(
                cuda, [_ARCH], ["GGML_TYPE_Q8_0"], inv)
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
