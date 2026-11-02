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


def _admission_model():
    """tools/tests/core/serving_admission_model.py, loaded by path (the test runs from any working directory)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("serving_admission_model",
                                                  Path(__file__).with_name("serving_admission_model.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    def test_admission_rule_is_specified(self) -> None:
        """§5.2 as amended (§11, amendment 5): stable facts by equality, live terms never compared, three admission
        checks on the live split, with kv_required from the selected backing floor of the total KV ceiling."""
        spec = (CONTRACT / "spec.md").read_text(encoding="utf-8")
        for text in ("The plan's measured `kv` is never an admission input. The live resolved KV must be at least "
                     "`kv_required`.",
                     "`live.vram.experts >= expected.vram.expert_floor`",
                     "`live.vram.claimable >= expected.vram.statics + live.vram.experts + live.vram.stage + "
                     "kv_required`",
                     "`live.vram.kv_resolved >= kv_required`",
                     "a fraction of the **total** `kv_ceiling`",
                     "the **common KV budget**",
                     "The expected plan freezes the **resolved** placement",
                     "Each card's physical capacity is the one minimum-bound exception",
                     "5. **Admission uses what the plan requires, read from the live split**"):
            self.assertIn(text, spec)
        self.assertNotIn("stage bytes, expert footprint", spec, "stage and residency are live terms (amendment 5)")

    def test_admission_model_reproduces_the_examples(self) -> None:
        """The model of the engine's split (serving_admission_model.resolve) gives each example's measured VRAM plan
        byte for byte, kv_resolved included, and each example is admissible against its own plan."""
        model = _admission_model()
        for name in EXAMPLES:
            with self.subTest(name=name):
                manifest = _load(name)
                expected = manifest["spec"]["plan"]["engine_memory"]["expected"]
                vram = {k: v for k, v in expected["vram"].items() if k not in ("from", "cards")}
                self.assertEqual(model.resolve(claimable=vram["claimable"], **model.PROD), vram)
                ppm = next((c["value"] for c in manifest["spec"]["calibration"]
                            if c["key"] == "kv_backing_floor.fraction_ppm"), 0)
                verdict = model.admit(expected, expected, ppm)
                self.assertTrue(verdict["admit"], verdict)
                self.assertEqual(verdict["kv_required"], 18907078656)
                # the measured split leaves no margin, which is why the measured kv is not an input
                self.assertEqual(vram["statics"] + vram["kv"] + vram["experts"] + vram["stage"], vram["claimable"])

    def test_admission_vectors_are_generated_and_hold(self) -> None:
        """admission-vectors.json is exactly what the model generates (so it cannot drift from the rule), every
        verdict is the rule's, and the cases the amendment names are all present. rad0011's check test runs the same
        file against the C++ check."""
        model = _admission_model()
        text = (CONTRACT / "admission-vectors.json").read_text(encoding="utf-8")
        self.assertEqual(text, model.render(), "regenerate: python3 tools/tests/core/serving_admission_model.py")
        vectors = json.loads(text)["vectors"]
        for v in vectors:
            with self.subTest(vector=v["name"]):
                self.assertEqual(model.admit(v["expected"], v["live"], v["ppm"]), v["verdict"])
        by = {v["name"]: v["verdict"] for v in vectors}
        self.assertEqual(len(by), len(vectors), "vector names are unique")
        for name in ("prod-a3-reading-1", "prod-a3-reading-3", "prod-minus-24k", "prod-kv-truncation-edge",
                     "lab-stage-payer-switch-600000", "lab-experts-minus-32k", "lab-no-lender-minus-8k",
                     "capacity-equal", "below-old-measured-split", "kv-resolved-equal", "ceiling-at-schema-max",
                     "live-stage-grew", "experts-at-floor", "ppm-max", "lab-no-lender-stage-from-experts",
                     "lab-kv-ceiling-roundup"):
            self.assertTrue(by[name]["admit"], name)
        refused = {"prod-kv-truncation-below": "admission: resolved KV: live 18906873856 bytes fall short of the "
                                               "required 18907078656 bytes",
                   "lab-stage-payer-switch-950000": "admission: claimable VRAM:",
                   "lab-experts-below-floor": "admission: expert residency:",
                   "capacity-one-short": "admission: claimable VRAM: live 7399 bytes fall short of the required 7400",
                   "kv-resolved-one-short": "admission: resolved KV: live 4199 bytes fall short of the required 4200",
                   "floor-rounds-up": "the required 4201 bytes",
                   "ceiling-at-schema-max-short": "the required 9007190247541737 bytes",
                   "ppm-negative": "kv_backing_floor_ppm must be in [0, 1000000)",
                   "ppm-million": "kv_backing_floor_ppm must be in [0, 1000000)",
                   "ceiling-past-schema-max": "byte counts in [0, 2^53 - 1]",
                   "kv-resolved-missing": "byte counts in [0, 2^53 - 1]",
                   "statics-differ": "VRAM statics: live 3001 bytes differ from the planned 3000 bytes",
                   "experts-below-floor": "admission: expert residency: live 99 bytes fall short of the required 100",
                   "card-plane-differs": "card 0 engine_expert_plane",
                   "lab-no-lender-stage-weights-roundup": "admission: resolved KV: live 535822336 bytes fall short of "
                                                          "the required 536870912 bytes",
                   "lab-stage-payer-switch-rounding-924805": "admission: resolved KV: live 7944011776 bytes fall short "
                                                             "of the required 7944014461 bytes"}
        for name, needle in refused.items():
            self.assertFalse(by[name]["admit"], name)
            self.assertIn(needle, by[name]["needle"], name)
        # resolver-shaped cases show what the review found: residency and staging move with 8 KiB of claimable
        moved = [v for v in vectors if v["name"].startswith("lab-experts-minus-")]
        self.assertTrue(any(v["live"]["vram"]["stage"] > v["expected"]["vram"]["stage"] for v in moved))
        switch = next(v for v in vectors if v["name"] == "lab-stage-payer-switch-600000")
        self.assertLess(switch["live"]["vram"]["kv"], switch["expected"]["vram"]["kv"])

    def test_schema_is_the_generated_one(self) -> None:
        """The specification's schema is the contract generator's output, byte for byte (contract/serving-v1.toml; the
        C++ bc_contract_check regenerates it in CI), and the registry the generator reads is integer-only."""
        generated = ROOT / "engines/radiance/components/serving/generated/serving-v1.schema.json"
        self.assertEqual(SCHEMA.read_bytes(), generated.read_bytes(),
                         "regenerate: bc_contract_gen contract/serving-v1.toml generated/, then copy the schema")
        import tomllib
        registry = tomllib.loads((ROOT / "engines/radiance/components/serving/contract/serving-v1.toml")
                                 .read_text(encoding="utf-8"))
        floats = [path for path, value in _walk(registry) if isinstance(value, float)]
        self.assertEqual(floats, [], "the registry is integer-only (spec §11, amendment 3)")
        fingerprint = (ROOT / "engines/radiance/components/serving/generated/contract.fingerprint").read_text()
        self.assertRegex(fingerprint, r"^sha256:[0-9a-f]{64}\n$")

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
