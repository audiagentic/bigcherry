"""The serving manifest v1 contract (docs/design/serving-manifest-v1): integer-only, closed, and self-consistent.

The C++ contract (registry, generator, reader) is checked against these files in later slices; this test keeps the
files themselves honest: no non-integer number anywhere, every object closed, the examples valid, and each example's
content_digest reproducible by an independent canonicaliser.
"""
from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "docs/design/serving-manifest-v1"
SCHEMA = CONTRACT / "serving-manifest.v1.schema.json"
EXAMPLES = ("example.flashnext-4gpu.json", "example.lab-experiment.json")


def _load(name: str) -> dict:
    return json.loads((CONTRACT / name).read_text(encoding="utf-8"))


def _walk(node, path=""):
    yield path, node
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{path}/{index}")


def _jcs_ascii(document: dict) -> bytes:
    """RFC 8785 for an ASCII-only, integer-only document: sorted keys (UTF-16 order equals code-point order for ASCII),
    no whitespace, integers in plain decimal. Refuses anything outside that subset rather than approximate it."""
    for path, value in _walk(document):
        if isinstance(value, float):
            raise AssertionError(f"{path}: non-integer number")
        if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2**53 - 1:
            raise AssertionError(f"{path}: beyond I-JSON")
    text = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if not text.isascii():
        raise AssertionError("non-ASCII content needs a full UTF-16-ordering canonicaliser")
    return text.encode("utf-8")


class ServingManifestV1ContractTests(unittest.TestCase):
    def test_schema_has_no_non_integer_number_type(self) -> None:
        offenders = [path for path, node in _walk(json.loads(SCHEMA.read_text(encoding="utf-8")))
                     if isinstance(node, dict) and (node.get("type") == "number" or
                                                    (isinstance(node.get("type"), list) and "number" in node["type"]))]
        self.assertEqual(offenders, [], "v1 numbers are integers only (spec §8, §11)")

    def test_every_schema_object_is_closed(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        open_objects = [path for path, node in _walk(schema)
                        if isinstance(node, dict) and node.get("type") == "object" and "properties" in node
                        and node.get("additionalProperties") is not False and "unevaluatedProperties" not in node]
        self.assertEqual(open_objects, [], "v1 is closed (spec §8)")

    def test_examples_are_integer_only(self) -> None:
        for name in EXAMPLES:
            with self.subTest(name=name):
                self.assertEqual([p for p, v in _walk(_load(name)) if isinstance(v, float)], [])

    def test_example_digest_is_reproduced_independently(self) -> None:
        for name in EXAMPLES:
            with self.subTest(name=name):
                manifest = _load(name)
                recorded = manifest["metadata"].pop("content_digest")
                self.assertEqual(recorded, "sha256:" + hashlib.sha256(_jcs_ascii(manifest)).hexdigest())

    def test_amendments_are_applied(self) -> None:
        for name in EXAMPLES:
            with self.subTest(name=name):
                keys = {entry["key"]: entry["value"] for entry in _load(name)["spec"]["calibration"]}
                self.assertEqual(keys["host_reserve.total_fraction_ppm"], 50000)
                self.assertEqual(keys["kv_backing_floor.fraction_ppm"], 600000)
                self.assertNotIn("host_reserve.total_fraction", keys)
                self.assertNotIn("kv_backing_floor.fraction", keys)
        schema = SCHEMA.read_text(encoding="utf-8")
        self.assertIn('"balance_speeds_milli"', schema)
        self.assertNotIn('"balance_speeds"', schema)
        self.assertIn("## 11. Amendments against the review draft", (CONTRACT / "spec.md").read_text(encoding="utf-8"))

    def test_admission_uses_the_required_kv(self) -> None:
        """§5.2 as amended (§11, amendment 5): live claimable >= statics + experts + stage + kv_required, where
        kv_required is the larger of one full-length session and the calibrated backing floor of kv_ceiling."""
        spec = (CONTRACT / "spec.md").read_text(encoding="utf-8")
        self.assertIn("`live.vram.claimable >= expected.vram.statics + expected.vram.experts + expected.vram.stage + "
                      "kv_required`", spec)
        self.assertIn("5. **Admission uses the required KV, not the measured share**", spec)
        for name in EXAMPLES:
            with self.subTest(name=name):
                manifest = _load(name)
                expected = manifest["spec"]["plan"]["engine_memory"]["expected"]
                vram = expected["vram"]
                ppm = next((c["value"] for c in manifest["spec"]["calibration"]
                            if c["key"] == "kv_backing_floor.fraction_ppm"), 0)
                kv_required = max(expected["kv_one_session"]["bytes"], -(-vram["kv_ceiling"] * ppm // 1_000_000))
                required = vram["statics"] + vram["experts"] + vram["stage"] + kv_required
                # each example is admissible against its own measured plan under the requirement rule
                self.assertLessEqual(required, vram["claimable"])
                # the measured split leaves no margin, which is why it is not the admission input
                self.assertEqual(vram["statics"] + vram["kv"] + vram["experts"] + vram["stage"], vram["claimable"])

    def test_schema_and_examples_validate(self) -> None:
        try:
            import jsonschema
        except ImportError:
            if os.environ.get("CI"):
                self.fail("jsonschema is required in CI for the v1 contract test")
            self.skipTest("jsonschema not installed (CI installs it)")
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
        for name in EXAMPLES:
            with self.subTest(name=name):
                errors = [f"{list(e.path)}: {e.message}" for e in validator.iter_errors(_load(name))]
                self.assertEqual(errors, [])
        forged = _load(EXAMPLES[0])
        forged["spec"]["calibration"][1]["value"] = 0.05
        self.assertTrue(list(validator.iter_errors(forged)), "a non-integer calibration value must be refused")


if __name__ == "__main__":
    unittest.main()
