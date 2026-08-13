"""Every dimension a candidate names must reach its variant params.

This test exists because that invariant has been broken twice, both times
silently, and both times expensively.

`small_k` was a candidate dimension from the start: it appears in the stable
name (`...:sk1:...`) and it selects a distinct compiled instance. But
`_variant_initialiser` did not pack it, so the registry could *name* an
instance it had no way to *ask for*. The dispatch layer would request the
`sk0` geometry whatever the name said. On the real MTP workload the `sk1`
geometries turned out to be the most valuable variants in the catalog
(review RV10: `w6:nw4:rpb4:sk1` at 22%), so a one-line packing omission was
hiding the best result the project had.

`src0_type` was the same failure a second time. Every family names itself for
one src0 type, every launch path dispatches on the runtime type, and the type
never reached the descriptor -- so eligibility could not check it and
candidates were measured on signatures of other types (review RV01). That one
surfaced as a fatal abort in one family and as silently misattributed winners
in two others.

Both are the same shape: **the stable name says X and the packed params do
not carry X**. Neither was catchable by reading the code, because the name and
the packing are written in different functions and each looks correct alone.
Both were found on hardware, hours in. This test finds the whole class in
milliseconds, with no GPU.

It deliberately parses the *name* rather than reading `config` twice. Checking
`config` against params derived from `config` would pass even when the name
promises a dimension nothing implements, which is precisely the situation that
occurred.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bigcherry import autotune_catalog as catalog  # noqa: E402
from bigcherry import autotune_schema as schema  # noqa: E402
from bigcherry import paths  # noqa: E402

# Dimensions a stable name can carry, and where each must land in
# ggml_hip_variant_params. Keyed by family so a family reading the wrong slot
# is also caught -- MMQ's J and MMVQ's nwarps share `primary`, and swapping
# them would launch a valid-looking wrong kernel.
NAME_FIELDS = {
    "mmvq": {"w": "width", "nw": "primary", "rpb": "secondary", "sk": "small_k"},
    "mmq":  {"j": "primary", "fb": "fallback"},
    "mmvf": {"w": "width", "bs": "primary"},
    "mmf":  {"w": "width", "nw": "primary"},
}

TOKEN_RE = re.compile(r"^([a-z]+)(\d+)$")


def manifest_for_tests() -> dict:
    """Build a catalog with every dimension populated.

    `full-max` is used rather than the workload profiles because it ignores the
    observed inventory, so every family and every geometry is present. A test
    that only saw the shapes one workload happened to execute would not have
    caught `small_k`, which no inventory forces into existence.
    """
    root = paths.llama_root(None)
    return catalog.build_manifest(
        root,
        variant_set="full-max",
        architectures=schema.ARCHITECTURE_GROUPS["rdna3"],
        inventory=None,
        source_revision="test",
    )


class TestNameMatchesParams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = manifest_for_tests()

    def test_every_named_dimension_reaches_the_params(self):
        """A numeric token in the stable name must equal its params slot."""
        checked = 0
        for candidate in self.manifest["candidates"]:
            family = candidate["family"]
            if candidate["source_class"] == "native_wrapper":
                continue                      # no fixed variant, params are zero
            mapping = NAME_FIELDS.get(family)
            if mapping is None:
                continue                      # blas carries no geometry
            fields = catalog.variant_fields(candidate)

            # stable name: family:type:tok:tok:...:vN
            for token in candidate["stable_name"].split(":")[2:-1]:
                match = TOKEN_RE.match(token)
                if match is None:
                    continue                  # non-numeric, e.g. an sram tag
                prefix, value = match.group(1), int(match.group(2))
                slot = mapping.get(prefix)
                if slot is None:
                    continue
                self.assertEqual(
                    fields[slot], value,
                    f"{candidate['stable_name']}: the name promises "
                    f"{prefix}={value} but variant params carry "
                    f"{slot}={fields[slot]}. The registry can name this "
                    f"variant and cannot ask for it.")
                checked += 1

        # Guard against the test silently passing because it matched nothing.
        self.assertGreater(checked, 100,
                           "expected to check many dimensions; the name format "
                           "or the catalog changed shape")

    def test_small_k_is_carried(self):
        """`sk1` in the name must set small_k. Regression: it did not."""
        seen = 0
        for candidate in self.manifest["candidates"]:
            if ":sk1:" not in candidate["stable_name"]:
                continue
            seen += 1
            self.assertEqual(
                catalog.variant_fields(candidate)["small_k"], 1,
                f"{candidate['stable_name']} names small_k and does not carry "
                f"it; the most valuable geometries on the real workload are "
                f"sk1 (RV10)")
        self.assertGreater(seen, 0, "no sk1 candidates generated at all")

    def test_src0_type_is_carried(self):
        """Every typed family must carry its type. Regression: RV01."""
        for candidate in self.manifest["candidates"]:
            if candidate["source_class"] == "native_wrapper":
                continue
            if candidate["family"] == "blas":
                continue                      # no per-type identity by design
            name_type = candidate["stable_name"].split(":")[1]
            self.assertEqual(
                catalog.variant_fields(candidate)["src0_type"],
                f"GGML_TYPE_{name_type.upper()}",
                f"{candidate['stable_name']} names a type its variant params "
                f"do not carry; eligibility cannot then reject a signature of "
                f"another type (RV01)")


class TestGeneratedInstancesMatchRegistry(unittest.TestCase):
    """Every generated MMVQ candidate must have an instance, and vice versa.

    The dispatch layer aborts rather than falling back when it cannot find a
    compiled instance, which is the right behaviour and makes a mismatch fatal
    at runtime. Catching it here costs nothing.
    """

    def test_mmvq_candidates_and_instances_agree(self):
        manifest = manifest_for_tests()
        named = {c["stable_name"] for c in manifest["candidates"]
                 if c["family"] == "mmvq"
                 and c["source_class"] == "new_generated_variant"}
        rendered = catalog.render_mmvq_instances(manifest)
        for name in named:
            self.assertIn(
                name, rendered,
                f"{name} is in the registry with no compiled instance; the "
                f"launcher aborts on a missing instance rather than falling "
                f"back, so this is fatal at runtime")


class TestRenderedRegistryAgreesWithManifest(unittest.TestCase):
    """The generated registry must be a lossless projection of the catalog."""

    def test_registry_rows_match_manifest_candidates_and_native_coverage(self):
        manifest = manifest_for_tests()
        rendered = catalog.render_registry(manifest)
        rows = re.findall(r'\{\s+(\d+)u, "([^"]+)"', rendered)

        expected = [(str(index), candidate["stable_name"])
                    for index, candidate in enumerate(manifest["candidates"])]
        self.assertEqual(
            rows,
            expected,
            "generated registry rows must preserve manifest order, stable "
            "names, and runtime ids",
        )
        self.assertIn(
            f"#define GGML_HIP_AUTOTUNE_CANDIDATE_COUNT {len(expected)}",
            rendered,
        )

        native_by_family = {family: 0 for family in schema.FAMILIES}
        for candidate in manifest["candidates"]:
            if candidate["source_class"] == "native_wrapper":
                native_by_family[candidate["family"]] += 1
        self.assertEqual(native_by_family, {family: 1 for family in schema.FAMILIES})

    def test_registry_architecture_masks_and_native_fallbacks_match(self):
        """The compiled registry must preserve eligibility and fallback coverage.

        A stable name/ID agreement is not enough: the architecture mask is the
        runtime gate which decides whether a candidate can even be considered.
        A stale mask can silently remove a valid candidate, or make it appear
        valid on a GPU for which it was never generated.  Native wrappers are
        the fail-safe path, so each family must cover every target architecture
        in the manifest.
        """
        manifest = manifest_for_tests()
        rendered = catalog.render_registry(manifest)
        rows = re.findall(
            r'\{\s+(\d+)u, "([^"]+)",\s+'
            r'(GGML_HIP_FAMILY_\w+),\s+(GGML_HIP_SOURCE_\w+),\s+(\d+),\n'
            r'\s+UINT64_C\((\d+)\),\s+\d+,\s+\d+,\s+(\d+),',
            rendered,
        )
        self.assertEqual(len(rows), len(manifest["candidates"]))

        target_mask = schema.architecture_mask(manifest["architectures"])
        native_families: dict[str, int] = {}
        for row, candidate in zip(rows, manifest["candidates"]):
            runtime_id, stable_name, family_enum, source_enum, version, mask, native = row
            self.assertEqual(int(runtime_id), manifest["candidates"].index(candidate))
            self.assertEqual(stable_name, candidate["stable_name"])
            self.assertEqual(int(version), candidate["implementation_version"])
            self.assertEqual(int(mask), schema.architecture_mask(candidate["architectures"]))
            self.assertNotEqual(int(mask), 0)
            self.assertEqual(int(mask) & ~target_mask, 0)
            self.assertEqual(int(native), int(candidate["source_class"] == "native_wrapper"))
            expected_family = "GGML_HIP_FAMILY_" + candidate["family"].upper()
            self.assertEqual(family_enum, expected_family)
            if candidate["source_class"] == "native_wrapper":
                self.assertEqual(source_enum, "GGML_HIP_SOURCE_NATIVE_WRAPPER")
                self.assertEqual(int(mask), target_mask)
                native_families[candidate["family"]] = native_families.get(candidate["family"], 0) + 1

        self.assertEqual(native_families, {family: 1 for family in schema.FAMILIES})


if __name__ == "__main__":
    unittest.main()
