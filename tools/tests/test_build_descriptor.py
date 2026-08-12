"""Build-descriptor generation and validation (HI49)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry import autotune_catalog as catalog  # noqa: E402


def _manifest(variant_set: str, candidates: list[dict]) -> dict:
    manifest = {
        "artifact_version": 1,
        "variant_set": variant_set,
        "source_revision": "a" * 40,
        "architectures": ["gfx1100"],
        "signature_schema_version": 1,
        "hardware_schema_version": 1,
        "candidates": candidates,
    }
    manifest["summary"] = {
        "total": len(candidates),
        "by_family": _counts(candidates, "family"),
        "by_source_class": _counts(candidates, "source_class"),
    }
    manifest["manifest_hash"] = "b" * 32
    return manifest


def _counts(candidates: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in candidates:
        out[c[key]] = out.get(c[key], 0) + 1
    return dict(sorted(out.items()))


def _native_candidates() -> list[dict]:
    return [{"family": family, "source_class": "native_wrapper",
             "stable_name": f"{family}:native:v1"} for family in catalog.schema.FAMILIES]


class ProfileDescriptorTests(unittest.TestCase):
    def test_inventory_accepts_only_native_wrappers(self):
        manifest = _manifest("inventory", _native_candidates())
        descriptor = catalog.build_descriptor(manifest)
        self.assertEqual(descriptor["candidate_count"], len(catalog.schema.FAMILIES))
        self.assertEqual(descriptor["schema_version"], 1)

    def test_workload_rejects_five_candidate_inventory(self):
        manifest = _manifest("workload-max", _native_candidates())
        with self.assertRaises(catalog.CatalogError):
            catalog.build_descriptor(manifest)

    def test_profile_mismatch_is_fatal(self):
        manifest = _manifest("inventory", _native_candidates())
        descriptor = catalog.build_descriptor(manifest)
        with self.assertRaises(catalog.CatalogError):
            catalog.validate_profile_descriptor(descriptor, "workload-max")

    def test_generated_header_embeds_canonical_descriptor(self):
        manifest = _manifest("inventory", _native_candidates())
        manifest["build_descriptor"] = catalog.build_descriptor(manifest)
        header = catalog.render_build_hash(manifest)
        descriptor = manifest["build_descriptor"]
        self.assertIn(f'"{descriptor["descriptor_hash"]}"', header)
        self.assertIn(descriptor["manifest_hash"], header)


class InventoryIdentityTests(unittest.TestCase):
    def test_identity_is_order_independent(self):
        candidates = _native_candidates()
        forward = catalog.build_descriptor(_manifest("inventory", candidates))
        backward = catalog.build_descriptor(_manifest("inventory", list(reversed(candidates))))
        self.assertEqual(forward["descriptor_hash"], backward["descriptor_hash"])


if __name__ == "__main__":
    unittest.main()
