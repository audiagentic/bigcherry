"""The 27B native-attention KL job names one native YAML per arm, and those files are what arm_yaml.py writes."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LAB = REPO / "tools/lab/radiance"
BASE = REPO / "engines/radiance/configs/qwen27b-2xtx.yaml"


def arm_yaml():
    spec = importlib.util.spec_from_file_location("arm_yaml", LAB / "arm_yaml.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeKldJobTests(unittest.TestCase):
    def test_each_arm_file_is_the_base_config_with_its_changes(self):
        job = json.loads((LAB / "27b-native-kld.json").read_text())
        module = arm_yaml()
        self.assertEqual([a["name"] for a in job["arms"]], ["compat", "attn-native"])
        for arm in job["arms"]:
            self.assertEqual(set(arm["serving"]), {"config", "native_config"})
            self.assertEqual(arm["serving"]["config"], BASE.stem)
            committed = REPO / arm["serving"]["native_config"]
            native = "true" if arm["name"] == "attn-native" else "false"
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "arm.yaml"
                module.write_arm(BASE, out, ["models.qwen27b.tp-wire=exact", "models.qwen27b.max-num-seqs=1",
                                             f"x-bigcherry.kernels.attn-native={native}",
                                             "x-bigcherry.kernels.moe-native=false"])
                self.assertEqual(committed.read_text(), out.read_text())
        record = job["quality"]["record"]["serving"]
        self.assertEqual(set(record), {"config", "native_config"})
        self.assertNotEqual(record["config"], BASE.stem, "the reference is recorded on a different layout than the control")
        self.assertTrue((REPO / record["native_config"]).is_file())
        self.assertNotIn("reference_metadata", job["quality"])


if __name__ == "__main__":
    unittest.main()
