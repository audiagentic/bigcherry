"""RSC11: the Python serving surface consumes C++ KFD/PCI structured output."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bigcherry.core.environment import Device, DevicePower
from bigcherry.engine_adapters import radiance_inventory as ri
from bigcherry.engine_adapters.radiance import RadianceAdapter

BDFS = ("0000:03:00.0", "0000:06:00.0", "0000:09:00.0", "0000:17:00.0")
# Non-sorted ordinals are deliberate: static host ordinals must not influence roles.
FACTS = {"gpus": [
    {"bdf": BDFS[0], "hip_ordinal": 8, "arch": "gfx1100", "vram_total_mib": 24560},
    {"bdf": BDFS[1], "hip_ordinal": 4, "arch": "gfx1100", "vram_total_mib": 24560},
    {"bdf": BDFS[2], "hip_ordinal": 2, "arch": "gfx1201", "vram_total_mib": 32624},
    {"bdf": BDFS[3], "hip_ordinal": 6, "arch": "gfx1030", "vram_total_mib": 16368},
]}
HELPER = {
    "bdf": BDFS[2], "hip_ordinal": 2, "arch": "gfx1201",
    "budget_mib": 30000, "plugin_file": "libr3.so",
    "formats": ["w4nl64", "fp8_e4m3_block"], "rows_kernel": True,
    "placement": "head", "share": 1,
}


def explained(*, engines=(BDFS[0], BDFS[1]), helpers=(BDFS[2],), facts=None):
    facts = FACTS if facts is None else facts
    from bigcherry.engine_adapters.radiance_layout import load_archs
    arch = load_archs()["gfx1201"]
    rows = []
    for i, bdf in enumerate(helpers):
        gpu = next((g for g in facts["gpus"] if g["bdf"] == bdf), {})
        rows.append({
            "bdf": bdf, "hip_ordinal": gpu.get("hip_ordinal"),
            "arch": gpu.get("arch"), "budget_mib": 30000,
            "plugin_file": arch.plugin, "formats": list(arch.formats),
            "rows_kernel": arch.rows_kernel, "placement": "head", "share": 1,
        })
    return {
        "facts": facts,
        "values": {"layout": {"name": "flashnext-tp2xtx-r9700",
                              "engine_cards": list(engines), "helper_cards": list(helpers)},
                   "engine": {"tp_wire": "exact"}},
        "process_helpers": rows,
    }


def local_host(*, devices=()):
    return SimpleNamespace(model_root="/models", work_root="/work", devices=tuple(devices))


class InventoryContractTests(unittest.TestCase):
    def test_detector_json_and_live_role_bdfs_are_authoritative(self):
        with patch.object(ri, "_run", return_value=explained()) as run:
            selection = ri.resolve("flashnext-tp2xtx-r9700", host=local_host(
                devices=(Device(0, "gfx1030", "wrong", 1, BDFS[0]),)))
        self.assertIn("--explain", run.call_args.args[1])
        self.assertIn("--json", run.call_args.args[1])
        self.assertEqual(selection.engine_bdfs, BDFS[:2])
        self.assertEqual(selection.resolved.engine_indices, (8, 4))
        self.assertEqual(selection.resolved.helpers[0].index, 2)
        self.assertEqual(selection.helper_bdfs, (BDFS[2],))
        self.assertEqual(selection.resolved.engine_arch, "gfx1100")
        self.assertEqual(selection.resolved.tp_wire, "exact")

    def test_malformed_and_duplicate_detector_bdfs_and_ordinals_fail_closed(self):
        for key, value, message in (
            ("bdf", BDFS[0], "duplicate detector BDF"),
            ("bdf", "bad", "invalid PCI BDF"),
            ("hip_ordinal", 8, "duplicate/invalid detector HIP ordinal"),
            ("vram_total_mib", 0, "invalid VRAM"),
        ):
            with self.subTest(key=key, value=value):
                facts = json.loads(json.dumps(FACTS))
                facts["gpus"][1][key] = value
                with self.assertRaisesRegex(ri.InventoryError, message):
                    ri.Inventory.from_facts(facts)

    def test_stale_role_and_helper_conflicts_rejected(self):
        cases = [
            (explained(engines=(BDFS[0], "0000:ee:00.0")), "stale or unavailable"),
            (explained(engines=(BDFS[0], BDFS[0])), "duplicate engine/helper"),
            (explained(helpers=(BDFS[0],)), "duplicate engine/helper"),
        ]
        for doc, reason in cases:
            with self.subTest(reason=reason), patch.object(ri, "_run", return_value=doc), self.assertRaisesRegex(ri.InventoryError, reason):
                ri.resolve("flashnext-tp2xtx-r9700")
        doc = explained()
        doc["process_helpers"][0]["hip_ordinal"] = 3
        with patch.object(ri, "_run", return_value=doc), self.assertRaisesRegex(ri.InventoryError, "disagrees with detected"):
            ri.resolve("flashnext-tp2xtx-r9700")

    def test_bdf_scoped_power_override_must_match_detector(self):
        inventory = ri.Inventory.from_facts(FACTS)
        inventory.check_power_overrides(local_host(devices=(
            Device(19, "gfx1201", "arbitrary", 1, BDFS[2], DevicePower(power_cap_w=270, evidence="run")),
        )))
        with self.assertRaisesRegex(ri.InventoryError, "stale or unavailable"):
            inventory.check_power_overrides(local_host(devices=(
                Device(0, "gfx1100", "stale", 1, "0000:ff:00.0", DevicePower(power_cap_w=100, evidence="run")),
            )))
        with self.assertRaisesRegex(ri.InventoryError, "requires its PCI BDF"):
            inventory.check_power_overrides(local_host(devices=(
                Device(0, "gfx1100", "unbound", 1, power=DevicePower(power_cap_w=100, evidence="run")),
            )))

    def test_no_explain_fallback_and_no_missing_fields(self):
        with patch.object(ri, "_run", return_value={"facts": FACTS}), self.assertRaisesRegex(ri.InventoryError, "lacks required structured"):
            ri.resolve("flashnext-tp2xtx-r9700")
        with patch.object(ri, "_run", side_effect=ri.InventoryError("offline")), self.assertRaisesRegex(ri.InventoryError, "offline"):
            ri.resolve("flashnext-tp2xtx-r9700")

    def test_fake_bc_serve_binary_detect_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "bc-serve"
            fake.write_text("#!" + sys.executable + "\n"
                            "import json,sys\n"
                            "assert sys.argv[1:3] == ['--detect','--json']\n"
                            "print(json.dumps(" + repr(FACTS) + "))\n", encoding="utf-8")
            fake.chmod(0o755)
            import subprocess
            original_run = subprocess.run
            def fake_run(command, **kwargs):
                return original_run([sys.executable, str(fake), *command[1:]], **kwargs)
            with patch.object(ri.subprocess, "run", side_effect=fake_run):
                result = ri.detect(binary=fake, sys_root=root)
            self.assertEqual([(g.ordinal, g.bdf) for g in result.gpus], [
                (8, BDFS[0]), (4, BDFS[1]), (2, BDFS[2]), (6, BDFS[3])])

    def test_sidecars_use_live_detector_not_host_ordinals(self):
        from bigcherry.engine_adapters.radiance import RadianceAdapter
        from test_engine_adapters import write_manifest
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_manifest(directory, helpers=(BDFS[1], BDFS[0]))
            with patch.object(ri, "detect", return_value=ri.Inventory.from_facts(FACTS)), patch(
                    "bigcherry.engine_adapters.radiance._host", return_value=local_host(
                        devices=(Device(0, "gfx1100", "stale", 1, BDFS[0]),))):
                sidecars = RadianceAdapter().sidecars(
                    {"name": "a", "manifest": manifest}, {}, Path(directory), Path("build/bin/radiance"))
            self.assertEqual([s.env["ROCR_VISIBLE_DEVICES"] for s in sidecars], ["4", "8"])

    def test_adapter_serving_layout_consumes_the_same_roles(self):
        with patch.object(ri, "_run", return_value=explained()), patch(
                "bigcherry.engine_adapters.radiance._host", return_value=local_host()), patch.dict(
                    os.environ, {"BC_MODEL_ROOT": "/models"}, clear=False):
            result = RadianceAdapter().serving_layout("flashnext-tp2xtx-r9700")
        self.assertEqual(result.environment, {"ROCR_VISIBLE_DEVICES": "8,4", "HIP_VISIBLE_DEVICES": "0,1"})
        self.assertEqual(result.expected_execution["locators"], list(BDFS[:2]))


if __name__ == "__main__":
    unittest.main()
