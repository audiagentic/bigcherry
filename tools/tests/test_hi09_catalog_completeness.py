"""Offline catalog-completeness contracts for the HI09 MMVQ generator."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bigcherry import autotune_catalog as catalog  # noqa: E402
from bigcherry import autotune_schema as schema  # noqa: E402
from bigcherry import paths  # noqa: E402


def full_manifest(architectures: list[str]) -> dict:
    return catalog.build_manifest(
        paths.llama_root(None),
        variant_set="full-max",
        architectures=architectures,
        inventory=None,
        source_revision="hi09-test",
    )


class HI09CatalogCompletenessTests(unittest.TestCase):
    def test_staged_profile_keeps_native_fallback_after_resource_filtering(self):
        """Resource filtering must never remove the correctness baseline."""
        manifest = full_manifest(schema.ARCHITECTURE_GROUPS["rdna3"])
        generated = [c for c in manifest["candidates"]
                     if c["source_class"] == "new_generated_variant"]
        blacklist = {
            (candidate["stable_name"], arch): ("register_spill",)
            for candidate in generated
            for arch in candidate["architectures"]
        }
        filtered = catalog.apply_resource_blacklist(
            [catalog.Candidate(
                c["stable_name"], c["family"], c["source_class"],
                list(c["architectures"]), dict(c["config"]))
             for c in manifest["candidates"]],
            blacklist,
        )
        natives = [c for c in filtered
                   if c.source_class == "native_wrapper"]
        self.assertEqual({c.family for c in natives}, set(schema.FAMILIES))
        self.assertEqual(len(natives), len(schema.FAMILIES))
        self.assertEqual(
            {arch for c in natives for arch in c.architectures},
            set(schema.ARCHITECTURE_GROUPS["rdna3"]),
        )

    def test_every_generated_mmvq_candidate_has_registry_and_instance(self):
        """A generated geometry must be both registered and compilable."""
        manifest = full_manifest(schema.ARCHITECTURE_GROUPS["rdna3"])
        generated = [c for c in manifest["candidates"]
                     if c["family"] == "mmvq"
                     and c["source_class"] == "new_generated_variant"]
        registry = catalog.render_registry(manifest)
        instances = catalog.render_mmvq_instances(manifest)

        for candidate in generated:
            config = candidate["config"]
            self.assertIn('"' + candidate["stable_name"] + '"', registry)
            marker = (
                f"DECL_MMVQ_AUTOTUNE_CASE_SK(GGML_TYPE_{config['type'].upper()}, "
                f"{config['width']}, {config['nwarps']}, "
                f"{config['rows_per_block']}, {int(config['small_k'])})"
            )
            self.assertIn(marker, instances, candidate["stable_name"])

        # The generated lookup must not contain a geometry absent from the
        # catalog. This catches stale/manual entries as well as missing ones.
        lookup = re.findall(
            r"type == GGML_TYPE_(\w+) && width == (\d+) && "
            r"nwarps == (\d+) &&\s+rows_per_block == (\d+) && "
            r"small_k == (true|false)",
            instances,
        )
        expected = {
            (c["config"]["type"].upper(), str(c["config"]["width"]),
             str(c["config"]["nwarps"]), str(c["config"]["rows_per_block"]),
             "true" if c["config"]["small_k"] else "false")
            for c in generated
        }
        self.assertEqual(set(lookup), expected)

    def test_architecture_masks_are_order_independent_and_bounded(self):
        """Equivalent target sets must produce identical deterministic masks."""
        ordered = schema.ARCHITECTURE_GROUPS["rdna3"]
        reversed_order = list(reversed(ordered))
        first = full_manifest(ordered)
        second = full_manifest(reversed_order)
        first_by_name = {c["stable_name"]: c for c in first["candidates"]}
        second_by_name = {c["stable_name"]: c for c in second["candidates"]}
        self.assertEqual(set(first_by_name), set(second_by_name))
        target_mask = schema.architecture_mask(ordered)
        for name, candidate in first_by_name.items():
            other = second_by_name[name]
            self.assertEqual(candidate["architectures"], other["architectures"])
            self.assertEqual(candidate["architecture_mask"],
                             other["architecture_mask"])
            self.assertNotEqual(candidate["architecture_mask"], 0)
            self.assertEqual(candidate["architecture_mask"] & ~target_mask, 0)

    def test_supported_coverage_is_present_and_is_not_inventory_limited(self):
        manifest = catalog.build_manifest(
            paths.llama_root(None),
            variant_set="workload-max",
            architectures=schema.ARCHITECTURE_GROUPS["rdna3"],
            inventory=catalog.Inventory(mmq_types={"q8_0"}, widths={1}),
            source_revision="hi61-test",
        )
        coverage = manifest["supported_coverage"]
        self.assertIn("q6_k", coverage["supported_types"])
        self.assertIn("q6_k", coverage["by_type"])
        self.assertFalse(coverage["by_type"]["q6_k"]["observed"])
        self.assertGreater(coverage["by_type"]["q6_k"]["alternative_count"], 0)


if __name__ == "__main__":
    unittest.main()
