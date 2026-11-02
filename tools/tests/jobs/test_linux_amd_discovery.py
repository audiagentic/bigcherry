from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from bigcherry.hardware.linux_amd import AmdDiscoveryError, inventory_from_amdsmi_json

LIST = {
    "gpu": {
        "0": {
            "BDF": "0000:03:00.0",
            "UUID": "GPU-AAA",
            "HIP_ID": 0,
            "RENDER": "renderD128",
        },
        "1": {
            "BDF": "0000:04:00.0",
            "UUID": "GPU-BBB",
            "HIP_ID": 1,
            "RENDER": "/dev/dri/renderD129",
        },
    }
}
STATIC = {
    "gpu": {
        "0": {
            "ASIC": {"MARKET_NAME": "Radeon XTX", "TARGET_GRAPHICS_VERSION": "gfx1100"},
            "BUS": {"BDF": "0000:03:00.0"},
            "VRAM": {"SIZE": "24 GiB"},
            "DRIVER": {"DRIVER_VERSION": "6.14.14"},
        },
        "1": {
            "ASIC": {"MARKET_NAME": "Radeon XTX", "TARGET_GRAPHICS_VERSION": "gfx1100"},
            "BUS": {"BDF": "0000:04:00.0"},
            "VRAM": {"SIZE": "24 GiB"},
            "DRIVER": {"DRIVER_VERSION": "6.14.14"},
        },
    }
}


class AmdSmiDiscoveryTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("linux"), "builds a fake sysfs tree whose PCI device names contain ':' (invalid on Windows)")
    def test_uuid_inventory_captures_locators_but_keys_identity_by_uuid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for bdf, numa in (("0000:03:00.0", "0"), ("0000:04:00.0", "1")):
                path = root / "bus" / "pci" / "devices" / bdf
                path.mkdir(parents=True)
                (path / "numa_node").write_text(numa, encoding="ascii")
            inv = inventory_from_amdsmi_json(
                LIST,
                STATIC,
                host_id="dev",
                platform_environment_hash="env",
                sysfs_root=root,
            )
        self.assertEqual([d.device_id for d in inv.devices], ["GPU-AAA", "GPU-BBB"])
        first = inv.devices[0]
        self.assertEqual(first.identity_source, "amd-smi-uuid")
        self.assertEqual(first.architecture, "gfx1100")
        self.assertEqual(first.vram_bytes, 24 * 1024**3)
        self.assertEqual(first.render_node, "/dev/dri/renderD128")
        self.assertEqual(first.launch_ordinal, 0)
        self.assertEqual(first.numa_node, 0)

    def test_amdsmi_26_gpu_data_static_shape(self):
        """amd-smi 26.x wraps static output as {"gpu_data": [...]}; the list output stays a bare list."""
        listed = [
            {"gpu": 0, "bdf": "0000:03:00.0", "uuid": "GPU-AAA", "hip_id": 0, "render": "renderD128"},
            {"gpu": 1, "bdf": "0000:09:00.0", "uuid": "GPU-CCC", "hip_id": 1, "render": "renderD129"},
        ]
        static = {"gpu_data": [
            {"gpu": 0, "asic": {"market_name": "AMD Radeon RX 7900 XTX", "target_graphics_version": "gfx1100"},
             "bus": {"bdf": "0000:03:00.0"}, "vram": {"size": {"value": 24560, "unit": "MB"}}},
            {"gpu": 1, "asic": {"market_name": "AMD Radeon AI PRO R9700", "target_graphics_version": "gfx1201"},
             "bus": {"bdf": "0000:09:00.0"}, "vram": {"size": {"value": 32624, "unit": "MB"}}},
        ]}
        with tempfile.TemporaryDirectory() as temp:
            inv = inventory_from_amdsmi_json(
                listed, static, host_id="dev", platform_environment_hash="env", sysfs_root=Path(temp),
            )
        self.assertEqual([d.architecture for d in inv.devices], ["gfx1100", "gfx1201"])
        self.assertEqual([d.model for d in inv.devices], ["AMD Radeon RX 7900 XTX", "AMD Radeon AI PRO R9700"])
        self.assertEqual([d.launch_ordinal for d in inv.devices], [0, 1])
        self.assertEqual([d.vram_bytes for d in inv.devices], [24560 * 1024**2, 32624 * 1024**2])

    @unittest.skipUnless(sys.platform.startswith("linux"), "fake sysfs PCI names contain ':'")
    def test_sysfs_vram_total_is_authoritative(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for bdf, total in (("0000:03:00.0", 25753026560), ("0000:04:00.0", 25753026560)):
                path = root / "bus" / "pci" / "devices" / bdf
                path.mkdir(parents=True)
                (path / "mem_info_vram_total").write_text(f"{total}\n", encoding="ascii")
            inv = inventory_from_amdsmi_json(LIST, STATIC, host_id="dev", platform_environment_hash="env",
                                             sysfs_root=root)
        self.assertEqual([d.vram_bytes for d in inv.devices], [25753026560, 25753026560])

    def test_missing_stable_id_requires_explicit_weak_epoch(self):
        listed = {
            "gpu": {
                "0": {
                    "BDF": "0000:03:00.0",
                    "HIP_ID": 0,
                    "RENDER": "renderD128",
                }
            }
        }
        static = {
            "gpu": {
                "0": {
                    "ASIC": {"MARKET_NAME": "R9700", "TARGET_GRAPHICS_VERSION": "gfx1201"},
                    "BUS": {"BDF": "0000:03:00.0"},
                    "VRAM": {"SIZE": "32 GiB"},
                }
            }
        }
        with self.assertRaises(AmdDiscoveryError):
            inventory_from_amdsmi_json(
                listed,
                static,
                host_id="dev",
                platform_environment_hash="env",
            )
        inv = inventory_from_amdsmi_json(
            listed,
            static,
            host_id="dev",
            platform_environment_hash="env",
            hardware_epoch="epoch-1",
        )
        self.assertEqual(inv.devices[0].identity_source, "weak-hardware-epoch")
        self.assertEqual(inv.devices[0].device_id, "weak:epoch-1:0000:03:00.0")

    def test_missing_architecture_fails_closed(self):
        broken = {
            "gpu": {
                "0": {
                    "BUS": {"BDF": "0000:03:00.0"},
                    "ASIC": {"MARKET_NAME": "Unknown"},
                    "VRAM": {"SIZE": "24 GiB"},
                }
            }
        }
        with self.assertRaises(AmdDiscoveryError):
            inventory_from_amdsmi_json(
                LIST,
                broken,
                host_id="dev",
                platform_environment_hash="env",
            )


if __name__ == "__main__":
    unittest.main()
