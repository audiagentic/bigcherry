from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from admin.render_bigcherry_slurm import install_rendered, render_files
from bigcherry.hardware.inventory import InventoryCatalog
from bigcherry.hardware.model import DeviceRecord, HardwareInventory


class SlurmRenderInstallTests(unittest.TestCase):
    def test_render_from_exact_accepted_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hardware_root = root / "hardware"
            catalog = InventoryCatalog(hardware_root)
            inv = HardwareInventory(
                host_id="brutus",
                platform_family="linux-rocm",
                platform_environment_hash="env",
                devices=(
                    DeviceRecord(
                        "uuid-a", "amd_uuid", "gfx1100", "7900XTX", 24 * 1024**3,
                        render_node="/dev/dri/renderD128",
                    ),
                    DeviceRecord(
                        "uuid-b", "amd_uuid", "gfx1201", "R9700", 32 * 1024**3,
                        render_node="/dev/dri/renderD129",
                    ),
                ),
            )
            catalog.record_observed("brutus", inv)
            catalog.accept_observed(
                "brutus", expected_material_hash=inv.material_hash
            )
            project_root = Path(__file__).resolve().parents[3]
            files = render_files(
                project_root=project_root,
                hardware_root=hardware_root,
                executor_id="brutus",
                node_name="brutus",
                cpus=32,
                real_memory_mib=64000,
            )
            self.assertIn("Gres=gpu:gfx1100:1,gpu:gfx1201:1", files["slurm.conf"])
            self.assertIn("CPUs=32", files["slurm.conf"])
            self.assertIn("RealMemory=64000", files["slurm.conf"])
            active = [
                line.strip()
                for line in files["slurm.conf"].splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            self.assertFalse(
                any(line.startswith("RequeueExit") for line in active),
                "monolithic validation jobs must not configure native RequeueExit",
            )
            self.assertIn("Type=gfx1100", files["gres.conf"])
            self.assertIn("Type=gfx1201", files["gres.conf"])
            self.assertNotIn("uuid-a", "\n".join(files.values()))
            self.assertIn("ConstrainDevices=no", files["cgroup.conf"])
            for text in files.values():
                self.assertIn(f"accepted_inventory_hash={inv.material_hash}", text)
            dest = root / "etc" / "slurm"
            written = install_rendered(files, dest)
            self.assertEqual(len(written), 3)
            self.assertEqual((dest / "gres.conf").read_text(), files["gres.conf"])

    def test_render_rejects_host_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalog = InventoryCatalog(root / "hardware")
            inv = HardwareInventory(
                host_id="other",
                platform_family="linux-rocm",
                devices=(
                    DeviceRecord(
                        "id", "fixture", "gfx1", "gpu", 1024,
                        render_node="/dev/dri/renderD128",
                    ),
                ),
            )
            catalog.record_observed("brutus", inv)
            catalog.accept_observed("brutus", expected_material_hash=inv.material_hash)
            with self.assertRaises(ValueError):
                render_files(
                    project_root=Path(__file__).resolve().parents[3],
                    hardware_root=root / "hardware",
                    executor_id="brutus",
                    node_name="brutus",
                    cpus=1,
                    real_memory_mib=1024,
                )


if __name__ == "__main__":
    unittest.main()
