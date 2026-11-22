"""A campaign job file describes its radiance serving arms as native arms: a config name plus its own YAML, never layout settings."""
import importlib.util
import json
import unittest
from pathlib import Path

from bigcherry.core import paths

LAB = paths.REPO_ROOT / "tools/lab/radiance"
CONFIGS = paths.REPO_ROOT / "engines/radiance/configs"
JOBS = sorted(LAB.glob("*.json"))
RETIRED_KEYS = {"layout", "set", "heat", "histogram", "experiment", "calibration", "live_policy", "calibration_run"}


def native_arms():
    spec = importlib.util.spec_from_file_location("native_arms", LAB / "native_arms.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def serving_jobs():
    for job in JOBS:
        doc = json.loads(job.read_text(encoding="utf-8"))
        if any("serving" in arm for arm in doc.get("arms", [])):
            yield job, doc


class NativeJobFileTests(unittest.TestCase):
    def test_serving_jobs_are_native_quality_jobs_without_shared_topology(self):
        seen = 0
        for job, doc in serving_jobs():
            with self.subTest(job=job.name):
                self.assertTrue(doc["quality_only"], "native arms are scored by bc-serve quality")
                self.assertEqual(doc["execution_evidence"], "observe")
                for key in ("model", "environment", "expected_execution"):
                    self.assertNotIn(key, doc, "the arm's YAML decides the model and the cards")
                # a reference is recorded by the job itself (native owner, its own YAML) or already stored with its provenance
                quality = doc["quality"]
                if "record" in quality:
                    self.assertEqual(set(quality["record"]["serving"]), {"config", "native_config"})
                    self.assertTrue((paths.REPO_ROOT / quality["record"]["serving"]["native_config"]).is_file())
                    self.assertIn("corpus", quality)
                    self.assertNotIn("reference_metadata", quality)
                for arm in doc["arms"]:
                    serving = arm["serving"]
                    self.assertEqual(set(serving), {"config", "native_config"})
                    self.assertFalse(RETIRED_KEYS & set(arm))
                    self.assertTrue((CONFIGS / f"{serving['config']}.yaml").is_file(), serving["config"])
                    self.assertTrue((paths.REPO_ROOT / serving["native_config"]).is_file(), serving["native_config"])
                    seen += 1
        self.assertGreaterEqual(seen, 6)

    def test_every_recipe_arm_file_is_what_the_recipe_writes(self):
        self.assertEqual(native_arms().main(["--check"]), 0)

    def test_every_job_arm_file_is_recipe_owned_or_a_27b_arm(self):
        recipes = native_arms().recipes()
        owned = {f"tools/lab/radiance/native-arms/{name}.yaml": recipe["base"] for name, recipe in recipes.items()}
        for job, doc in serving_jobs():
            for arm in doc["arms"]:
                serving = arm["serving"]
                if serving["native_config"] in owned:
                    with self.subTest(job=job.name, arm=arm["name"]):
                        self.assertEqual(owned[serving["native_config"]], serving["config"])
                else:
                    self.assertTrue(serving["native_config"].startswith("tools/lab/radiance/27b-native-kld/"), serving)

    def test_no_job_file_uses_the_retired_layout_serving_shape(self):
        for job in [*JOBS, *LAB.rglob("*.template.json")]:
            text = job.read_text(encoding="utf-8")
            with self.subTest(job=job.name):
                self.assertNotRegex(text, r'"layout"\s*:')


if __name__ == "__main__":
    unittest.main()
