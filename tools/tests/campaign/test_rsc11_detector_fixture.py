"""Card-free integration fixture for the C++ bc-serve KFD/PCI detector.

Set BC_SERVE_BINARY to a built bc-serve to exercise actual C++ discovery.
The fixture never reads or opens a physical GPU.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bigcherry.engine_adapters import radiance_inventory as ri


def fake_sys_root(root: Path) -> None:
    def write(name: str, value: str) -> None:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    write("sys/class/kfd/kfd/topology/nodes/0/properties", "gfx_target_version 0\n")
    for i, (bus, gfx, vram) in enumerate(((3, 110000, 24560), (6, 110000, 24560), (9, 120001, 32624)), start=1):
        write(f"sys/class/kfd/kfd/topology/nodes/{i}/properties",
              f"gfx_target_version {gfx}\nsimd_count 128\nlocation_id {bus*256}\ndomain 0\ndrm_render_minor {127+i}\n")
        pci = f"sys/bus/pci/devices/0000:{bus:02x}:00.0"
        write(f"{pci}/mem_info_vram_total", str(vram * 1048576))
        write(f"{pci}/mem_info_vram_used", str(128 * 1048576))
        write(f"{pci}/current_link_speed", "16.0 GT/s PCIe\n")
        write(f"{pci}/max_link_speed", "16.0 GT/s PCIe\n")
        write(f"{pci}/current_link_width", "16\n")
        write(f"{pci}/max_link_width", "16\n")
        write(f"{pci}/numa_node", "0\n")
    write("proc/meminfo", "MemTotal: 134217728 kB\nMemAvailable: 94371840 kB\nSwapTotal: 8388608 kB\n")
    (root / "dev/shm").mkdir(parents=True, exist_ok=True)
    write("proc/self/cgroup", "0::/agents/serve\n")
    write("sys/fs/cgroup/agents/serve/memory.max", str(64 << 30) + "\n")
    write("sys/fs/cgroup/agents/serve/memory.current", str(10 << 30) + "\n")
    write("sys/fs/cgroup/agents/serve/memory.stat", "anon 1\nfile 4294967296\n")
    write("sys/fs/cgroup/agents/memory.max", "max\n")
    write("sys/fs/cgroup/memory.max", "max\n")
    write("proc/self/limits", "Limit                     Soft Limit           Hard Limit           Units\nMax locked memory         8388608              8388608              bytes\n")
    write("proc/self/status", "Name:\tbc-serve\nCapEff:\t00000000a80425fb\n")
    write("sys/module/amdgpu/parameters/no_system_mem_limit", "N\n")


class RealDetectorFixtureTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("BC_SERVE_BINARY"), "set BC_SERVE_BINARY to test the built C++ detector")
    def test_kfd_pci_fixture_and_duplicate_bdf_refusal(self):
        binary = Path(os.environ["BC_SERVE_BINARY"])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fake_sys_root(root)
            detector = ri.detect(binary=binary, sys_root=root)
            self.assertEqual([(g.ordinal, g.bdf, g.arch) for g in detector.gpus], [
                (0, "0000:03:00.0", "gfx1100"),
                (1, "0000:06:00.0", "gfx1100"),
                (2, "0000:09:00.0", "gfx1201"),
            ])
            with patch.dict(os.environ, {"RADIANCE_HOME": "/runtime", "BC_MODEL_ROOT": "/models"}):
                layout = ri.resolve("flashnext-2xtx", binary=binary, sys_root=root,
                                    host=SimpleNamespace(model_root="/models", devices=()))
            self.assertEqual(layout.engine_bdfs, ("0000:09:00.0",))
            self.assertEqual(layout.helper_bdfs, ("0000:03:00.0", "0000:06:00.0"))
            self.assertEqual(layout.resolved.engine_indices, (2,))
            self.assertEqual([h.index for h in layout.resolved.helpers], [0, 1])
            properties = root / "sys/class/kfd/kfd/topology/nodes/2/properties"
            properties.write_text(properties.read_text().replace("location_id 1536", "location_id 768"))
            failed = subprocess.run([str(binary), "--detect", "--json", "--sys-root", str(root)],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(failed.returncode, 2)
            self.assertIn("duplicate KFD PCI locator", failed.stderr)


if __name__ == "__main__":
    unittest.main()
