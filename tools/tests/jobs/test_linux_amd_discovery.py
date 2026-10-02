from __future__ import annotations

import tempfile
import sys
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
                host_id="brutus",
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
                host_id="brutus",
                platform_environment_hash="env",
            )
        inv = inventory_from_amdsmi_json(
            listed,
            static,
            host_id="brutus",
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
                host_id="brutus",
                platform_environment_hash="env",
            )


if __name__ == "__main__":
    unittest.main()
