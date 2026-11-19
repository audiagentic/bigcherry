"""MRC04 contract: narrow keys, provenance and lock-safe store (no engine needed)."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from bigcherry.__main__ import main
from bigcherry.core import calibration_records as cr


class CalibrationRecordsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.inputs = {
            "policy_key_version": 2,
            "model": "m1",
            "layout": "flashnext-4gpu",
            "objective": "balanced",
            "power_profile": "stock",
            "engine_cards": [
                {"bdf": "0000:01:00.0", "arch": "gfx1201", "vram_total_mib": 32768}
            ],
            "helper_cards": [],
            "histogram_period_s": 1,
            "helper_runtime": {"combine": True},
            "remote_service": {"max_rows": 1024},
        }
        self.absent = {
            k: "not observed"
            for k in cr.key_declaration("radiance.live_policy")["quantity_fields"]
            if k != "txn_p50_ns"
        }
        self.record = cr.new_record(
            "radiance.live_policy",
            self.inputs,
            {"measurements": {"txn_p50_ns": 100}, "absent": self.absent},
            {
                "build_id": "a" * 64,
                "plan_key": "b" * 64,
                "source_revision": "rev-main",
                "run_id": "run-a",
                "host_facts_digest": "c" * 64,
                "recorded_at": "2026-11-20",
            },
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_narrow_key_across_builds_and_nonkey_inputs(self):
        other = copy.deepcopy(self.record)
        other["provenance"].update(
            {"build_id": "d" * 64, "plan_key": "e" * 64, "run_id": "run-b"}
        )
        self.assertEqual(cr.validate_record(other), other)
        self.assertEqual(self.record["key"]["sha256"], other["key"]["sha256"])
        changed = cr.new_record(
            "radiance.live_policy",
            self.inputs | {"engine_build": "x"},
            other["quantities"],
            other["provenance"],
        )
        self.assertEqual(changed["key"], self.record["key"])
        inputs = dict(self.inputs, histogram_period_s=2)
        self.assertNotEqual(
            cr.new_record(
                "radiance.live_policy",
                inputs,
                self.record["quantities"],
                other["provenance"],
            )["key"],
            self.record["key"],
        )
        for forbidden in ("engine_build", "host_pool_mib", "helper_budgets_mib"):
            invalid = copy.deepcopy(self.record)
            invalid["key"]["inputs"][forbidden] = "x"
            with self.assertRaises(cr.CalibrationError):
                cr.validate_record(invalid)

    def test_atomic_put_get_and_explicit_replace(self):
        store = cr.CalibrationStore(self.root)
        store.put(self.record)
        self.assertEqual(
            store.get("radiance.live_policy", self.record["key"]["sha256"]), self.record
        )
        with self.assertRaisesRegex(cr.CalibrationError, "already exists"):
            store.put(self.record)
        updated = copy.deepcopy(self.record)
        updated["quantities"]["measurements"]["txn_p50_ns"] = 200
        store.put(updated, replace=True)
        self.assertEqual(
            store.get("radiance.live_policy", self.record["key"]["sha256"]), updated
        )
        self.assertEqual(len(store.list()), 1)
        self.assertTrue((self.root / ".lock").exists())

    def test_quantity_partition_is_strict(self):
        missing = copy.deepcopy(self.record)
        missing["quantities"]["absent"].pop("hot_set_k")
        with self.assertRaisesRegex(cr.CalibrationError, "hot_set_k"):
            cr.validate_record(missing)
        invalid = copy.deepcopy(self.record)
        invalid["quantities"]["measurements"]["txn_p50_ns"] = -1
        with self.assertRaises(cr.CalibrationError):
            cr.validate_record(invalid)

    def test_native_key_declaration_matches_cpp(self):
        import re

        source = (
            Path(__file__).resolve().parents[3]
            / "engines/radiance/components/serving/plan_cache.cpp"
        ).read_text()
        match = re.search(
            r"Json policy_calibration_key_fields\(\) \{\s*"
            r"return Json::array\(\{(.*?)\}\);",
            source,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        native = re.findall(r'"([^"]+)"', match.group(1))
        self.assertEqual(
            native, cr.key_declaration("radiance.live_policy")["key_fields"]
        )

    def test_locked_racing_writers_publish_one_complete_record(self):
        from concurrent.futures import ThreadPoolExecutor

        store = cr.CalibrationStore(self.root)
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs = [
                pool.submit(store.put, copy.deepcopy(self.record)) for _ in range(8)
            ]
        outcomes = []
        for result in jobs:
            try:
                result.result()
                outcomes.append("written")
            except cr.CalibrationError as exc:
                self.assertIn("already exists", str(exc))
                outcomes.append("duplicate")
        self.assertEqual(outcomes.count("written"), 1)
        self.assertEqual(
            store.get("radiance.live_policy", self.record["key"]["sha256"]), self.record
        )

    def test_bad_hash_and_traversal_are_refused(self):
        record = copy.deepcopy(self.record)
        record["key"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(cr.CalibrationError, "digest"):
            cr.validate_record(record)
        with self.assertRaises(cr.CalibrationError):
            cr.CalibrationStore(self.root).get("../../other", "a" * 64)
        bad = copy.deepcopy(self.record)
        bad["provenance"].pop("host_facts_digest")
        with self.assertRaises(cr.CalibrationError):
            cr.validate_record(bad)

    def test_native_toml_v1_and_cli(self):
        toml = self.root / "calibration.toml"
        scope = json.dumps(self.inputs, sort_keys=True, separators=(",", ":"))
        columns = json.dumps(
            list(cr.key_declaration("radiance.live_policy")["key_fields"])
        )
        toml.write_text(
            "schema_version = 1\n[[live_policy_record]]\n"
            + 'plan_cache_key = "'
            + self.record["key"]["sha256"]
            + '"\n'
            + 'record_schema_version = 1\ncalibration_kind = "radiance.live_policy"\n'
            + "calibration_key_fields = "
            + columns
            + "\n"
            + "calibration_inputs = "
            + json.dumps(scope)
            + "\n"
            + 'source_plan_key = "'
            + "b" * 64
            + '"\n'
            + 'source_engine_build = "'
            + "a" * 64
            + '"\n'
            + 'source_revision = "rev-main"\nrun_id = "run-a"\n'
            + 'host_facts_digest = "'
            + "c" * 64
            + '"\n'
            + 'layout = "flashnext-4gpu"\nobjective = "balanced"\n'
            + 'seeded = "2026-11-20"\nevidence = "fixture"\n'
            + "txn_p50_ns = 100\nabsent = {"
            + ", ".join(f'{name} = "not observed"' for name in self.absent)
            + "}\n"
        )
        rows = cr.read_native_toml(toml)
        self.assertEqual(rows[0]["status"], "current")
        self.assertEqual(rows[0]["record"]["key"], self.record["key"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["calibration", "validate", "--store", str(toml)]), 0)
        self.assertEqual(json.loads(output.getvalue())["valid"], 1)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["calibration", "list", "--store", str(toml)]), 0)
        self.assertEqual(json.loads(output.getvalue())[0]["status"], "current")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                main(
                    [
                        "calibration",
                        "show",
                        "--store",
                        str(toml),
                        "--kind",
                        "radiance.live_policy",
                        "--key",
                        self.record["key"]["sha256"],
                    ]
                ),
                0,
            )
        self.assertEqual(json.loads(output.getvalue())["record"], self.record)
        # Genuine hash-only history, not scoped post-#1169 evidence.
        old = toml.read_text()
        for field in (
            "record_schema_version",
            "calibration_kind",
            "calibration_key_fields",
            "calibration_inputs",
            "source_plan_key",
            "source_engine_build",
            "source_revision",
            "run_id",
            "host_facts_digest",
        ):
            old = "\n".join(
                line for line in old.split("\n") if not line.startswith(field + " = ")
            )
        toml.write_text(old)
        # A row without calibration_inputs is a format this reader does not know: refused, never classified.
        with self.assertRaisesRegex(cr.CalibrationError, "unsupported record format"):
            cr.read_native_toml(toml)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["calibration", "validate", "--store", str(toml)]), 1)

    def test_store_with_hash_only_rows_is_refused(self):
        fixture = Path(__file__).resolve().parents[1] / (
            "fixtures/calibration/production-2026-11-19-with-hash-only.toml"
        )
        with self.assertRaisesRegex(cr.CalibrationError, "unsupported record format"):
            cr.read_native_toml(fixture)

    def test_production_post_1169_record_stays_current_without_store_rewrite(self):
        # The 2026-11-19 production calibration store with its hash-only rows pruned (bc-serve calibration-prune).
        fixture = Path(__file__).resolve().parents[1] / (
            "fixtures/calibration/production-2026-11-19.toml"
        )
        before = fixture.read_bytes()
        key = "c2a1aa0eabcb2968a42876e364e0ac18a072f8fe0e229c8aaea838de27c5de6b"
        entries = cr.read_native_toml(fixture)
        row = next(item for item in entries if item["key"] == key)
        self.assertEqual(row["status"], "current")
        self.assertIsNone(row["reason"])
        self.assertEqual(row["record"]["key"]["sha256"], key)
        self.assertEqual(
            row["record"]["provenance"]["build_id"],
            "3ebb390d5e1a5ba6ca4d05c7118cf7cc809935cbf962ba55fa7b9d78d15e63c8",
        )
        # Missing newer provenance is not invented by the compatibility reader.
        self.assertNotIn("host_facts_digest", row["record"]["provenance"])
        self.assertEqual([item["status"] for item in entries], ["current"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["calibration", "list", "--store", str(fixture)]), 0)
        listing = json.loads(output.getvalue())
        self.assertEqual(
            next(item for item in listing if item["key"] == key)["status"], "current"
        )
        self.assertEqual(fixture.read_bytes(), before)

        # A carried scope must still hash to the stored key.
        corrupted = self.root / "corrupted.toml"
        corrupted.write_bytes(before.replace(key.encode(), ("f" * 64).encode()))
        bad = next(
            item for item in cr.read_native_toml(corrupted) if item["key"] == "f" * 64
        )
        self.assertEqual(bad["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
