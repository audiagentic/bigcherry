from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from admin.install_bigcherry_slurm import _validate_preconditions, action_plan
from bigcherry.hardware.inventory import InventoryCatalog
from bigcherry.hardware.model import DeviceRecord, HardwareInventory


class SlurmInstallerTests(unittest.TestCase):
    def test_preflight_renders_accepted_inventory_and_safe_v1_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog = InventoryCatalog(root / "hardware")
            inventory = HardwareInventory(
                host_id="brutus",
                platform_family="linux-rocm",
                platform_environment_hash="env",
                devices=(
                    DeviceRecord(
                        "gpu-a",
                        "amd_uuid",
                        "gfx1100",
                        "7900XTX",
                        24 * 1024**3,
                        render_node="/dev/dri/renderD128",
                    ),
                ),
            )
            catalog.record_observed("brutus", inventory)
            catalog.accept_observed(
                "brutus", expected_material_hash=inventory.material_hash
            )
            project_root = Path(__file__).resolve().parents[3]
            files, accepted_hash = _validate_preconditions(
                project_root=project_root,
                hardware_root=root / "hardware",
                executor_id="brutus",
                node_name="brutus",
                cpus=16,
                real_memory_mib=32768,
            )
            self.assertEqual(accepted_hash, inventory.material_hash)
            active = [
                line.strip()
                for line in files["slurm.conf"].splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            self.assertFalse(any(line.startswith("RequeueExit") for line in active))
            self.assertIn("AccountingStorageType=accounting_storage/none", active)
            self.assertIn("ConstrainDevices=no", files["cgroup.conf"])
            self.assertIn("Name=gpu Type=gfx1100", files["gres.conf"])

    def test_action_plan_is_minimal_no_db_stack(self):
        argv = [action.argv for action in action_plan(node_name="brutus")]
        flattened = "\n".join(" ".join(row) for row in argv)
        self.assertIn("apt-get install -y slurm-wlm munge jq", flattened)
        self.assertIn("slurmd -G", flattened)
        self.assertIn("scontrol ping", flattened)
        self.assertNotIn("slurmdbd", flattened)
        self.assertNotIn("mysql", flattened)
        self.assertNotIn("mariadb", flattened)
        self.assertNotIn("sacctmgr", flattened)


if __name__ == "__main__":
    unittest.main()
