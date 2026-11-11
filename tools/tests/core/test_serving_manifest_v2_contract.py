"""Inactive v2 contract checks. Production remains pinned to the untouched v1 contract."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SERVING = ROOT / "engines/radiance/components/serving"
GENERATED = SERVING / "generated/v2"
DOCS = ROOT / "docs/design/serving-manifest-v2"
EXAMPLES = ("example.flashnext-4gpu.json", "example.lab-experiment.json")

spec = importlib.util.spec_from_file_location("bc_serving_v2_codec", GENERATED / "serving_codec.py")
assert spec and spec.loader
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)


class ServingManifestV2ContractTest(unittest.TestCase):
    def fixtures(self):
        return [json.loads((DOCS / name).read_text()) for name in EXAMPLES]

    def test_generated_schema_matches_documentation_copy(self):
        self.assertEqual((GENERATED / "serving-v2.schema.json").read_bytes(),
                         (DOCS / "serving-manifest.v2.schema.json").read_bytes())
        self.assertEqual(codec.SCHEMA["properties"]["apiVersion"]["const"], "bigcherry/v2")
        self.assertEqual(codec.SCHEMA["properties"]["metadata"]["properties"]
                         ["contract_version"]["const"], 2)
        self.assertEqual(codec.SCHEMA["$defs"]["powerSettings"]["additionalProperties"], False)
        self.assertEqual(codec.SCHEMA["properties"]["spec"]["properties"]["plan"]
                         ["properties"]["power"]["properties"]["cpu"]["properties"]
                         ["settings"]["additionalProperties"], False)

    def test_roundtrip_and_independent_ascii_jcs_digest(self):
        for name in EXAMPLES:
            with self.subTest(name=name):
                raw = (DOCS / name).read_text()
                doc = codec.decode(raw)
                self.assertEqual(codec.decode(codec.encode(doc)), doc)
                mutable = copy.deepcopy(doc)
                digest = mutable["metadata"].pop("content_digest")
                canonical = json.dumps(mutable, sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":"))
                self.assertTrue(canonical.isascii())
                self.assertEqual(digest, "sha256:" + hashlib.sha256(canonical.encode()).hexdigest())
                self.assertTrue(doc["spec"]["plan"]["power"]["devices"])
                self.assertEqual(doc["spec"]["plan"]["power"]["mode"], "observe")
                self.assertTrue(all(row["settings"] == {}
                                    for row in doc["spec"]["plan"]["power"]["devices"]))

    def test_other_api_version_is_refused(self):
        for name in EXAMPLES:
            with self.subTest(name=name):
                doc = json.loads((DOCS / name).read_text())
                doc["apiVersion"] = "bigcherry/v1"
                with self.assertRaisesRegex(codec.ManifestV2Error, "unsupported_version"):
                    codec.decode(json.dumps(doc))

    def test_reject_duplicate_json_members_and_non_integer_numbers(self):
        raw = (DOCS / EXAMPLES[0]).read_text()
        bad = raw.replace('"apiVersion": "bigcherry/v2",',
                          '"apiVersion": "bigcherry/v2", "apiVersion": "bigcherry/v2",', 1)
        with self.assertRaisesRegex(codec.ManifestV2Error, "duplicate JSON member"):
            codec.decode(bad)
        with self.assertRaisesRegex(codec.ManifestV2Error, "non-integer"):
            codec.decode(raw.replace('"histogram_period_s": 60', '"histogram_period_s": 60.5'))

    def test_unknown_members_and_type_faults_closed(self):
        example = self.fixtures()[0]
        paths = [
            ("power intent", example["spec"]["intent"]["power"]),
            ("power plan", example["spec"]["plan"]["power"]),
            ("resolved settings", example["spec"]["plan"]["power"]["devices"][0]["settings"]),
            ("measured resource ledger", example["spec"]["plan"]["resource_inputs"]),
            ("RHP18 startup state", example["spec"]["handoff"]["placement"]["live_policy_state"]),
        ]
        for description, mapping in paths:
            with self.subTest(description=description):
                mapping["unknown_s9_member"] = "not allowed"
                with self.assertRaises(codec.ManifestV2Error):
                    codec.decode(json.dumps(example))
                mapping.pop("unknown_s9_member")
        example["spec"]["plan"]["power"]["devices"][0]["settings"]["vddgfx_offset_mv"] = "wrong-type"
        with self.assertRaises(codec.ManifestV2Error):
            codec.decode(json.dumps(example))

    def test_power_schema_only_enforces_inherent_integer_bounds(self):
        document = self.fixtures()[0]
        settings = document["spec"]["plan"]["power"]["devices"][0]["settings"]
        settings["mclk_max_mhz"] = 0
        with self.assertRaises(codec.ManifestV2Error):
            codec.decode(json.dumps(document))
        settings.pop("mclk_max_mhz")
        settings["power_cap_w"] = 0
        with self.assertRaises(codec.ManifestV2Error):
            codec.decode(json.dumps(document))
        settings["power_cap_w"] = 1000000
        # No device-specific upper cap is invented by the manifest schema.
        # Observe-mode empty-settings semantics must still reject this.
        with self.assertRaisesRegex(codec.ManifestV2Error, "observe mode"):
            codec.decode(json.dumps(document))

    def test_role_bdf_resolution_fails_closed(self):
        doc = self.fixtures()[0]
        devs = doc["spec"]["plan"]["power"]["devices"]
        devs[1]["bdf"] = devs[0]["bdf"]
        with self.assertRaisesRegex(codec.ManifestV2Error, "role/BDF|duplicate BDF"):
            codec.decode(json.dumps(doc))
        doc = self.fixtures()[0]
        doc["spec"]["intent"]["power"]["role_overrides"]["helper99"] = {"power_cap_w": 240}
        with self.assertRaisesRegex(codec.ManifestV2Error, "unknown resolved role"):
            codec.decode(json.dumps(doc))

    def test_registry_records_provenance_without_origin_allowlists(self):
        import tomllib
        fields = tomllib.loads((SERVING / "contract/serving-v2.toml").read_text())["field"]
        self.assertFalse([f["path"] for f in fields if "origins" in f or "confidences" in f])
        self.assertFalse(hasattr(codec, "RESTRICTED"))


if __name__ == "__main__":
    unittest.main()
