"""Card-free v1 launcher input adapter tests: no GPU and no model payload."""
from __future__ import annotations

import argparse
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bigcherry.engine_adapters import radiance_inventory as inv
from bigcherry.engine_adapters import radiance_launch as launch
from bigcherry.engine_adapters import radiance_layout as layouts


def synthetic_rad(path: Path) -> None:
    """Minimal ABI v2 directory, with two routed experts and one embedding."""
    names = b"\x00gate\x00down\x00embed\x00ngram\x00"
    entries = [(1, 1, 0, 2), (6, 1, 1, 3), (11, -1, -1, 5), (17, -1, -1, 7)]
    header = bytearray(104)
    base = 104 + len(names)
    body = bytearray(len(entries) * launch._RAD_ENTRY)
    for i, (name, layer, expert, size) in enumerate(entries):
        off = i * launch._RAD_ENTRY
        struct.pack_into("<Q", body, off, name)
        struct.pack_into("<iii", body, off + 80, layer, expert, 0)
        struct.pack_into("<Q", body, off + 112, size)
    struct.pack_into("<IIQ", header, 0, launch._RAD_MAGIC, launch._RAD_VERSION,
                     base + len(body))
    struct.pack_into("<QQ", header, 56, 104, len(names))
    struct.pack_into("<QQ", header, 88, base, len(entries))
    path.write_bytes(header + names + body)


class LaunchDerivationTests(unittest.TestCase):
    def test_rad_directory_reads_expert_and_other_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.rad"
            synthetic_rad(path)
            sizes, embedding, ngram = launch._rad_sizes(path)
            self.assertEqual(set(sizes), {(1, 0), (1, 1)})
            self.assertEqual(embedding, 256)
            self.assertEqual(ngram, 256)
            path.write_bytes(b"not a model")
            with self.assertRaisesRegex(launch.LaunchResourceError, "RAD v2"):
                launch._rad_sizes(path)

    def test_rederives_from_cpp_owner_map_inventory_and_host_headroom(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model.rad"
            synthetic_rad(model)
            engine = inv.Gpu("0000:03:00.0", 0, "gfx1201", 32624)
            helper = inv.Gpu("0000:04:00.0", 1, "gfx1100", 24560)
            arch = layouts.load_archs()[helper.arch]
            resolved = layouts.Resolved(
                "flashnext-2xtx", (0,), engine.arch, "libr4d.so", "model.rad",
                None, (), (layouts.HelperCard(1, arch, helper.vram_mib, 20000, "head", 1),),
            )
            selected = inv.Layout(resolved, inv.Inventory((engine, helper)),
                                  (engine.bdf,), (helper.bdf,))
            doc = {
                "facts": {"host": {
                    "mem_available_mib": 65536,
                    "mem_total_mib": 131072,
                    "memory_cgroups": [{"room_mib": 60000}],
                    "amdgpu_pin": {"loaded": False, "lifted": True},
                }},
                "values": {"memory": {"host_reserve_mib": 5000}},
            }
            args = argparse.Namespace(
                layout="flashnext-2xtx", model=model, bc_serve=root/"bc-serve",
                radiance_home=root, heat=root/"heat", histogram="off",
                sessions=8, context=262144, ngram="disk",
            )
            def fake_run(cmd, **kwargs):
                out = Path(cmd[cmd.index("--write-manifest") + 1])
                out.write_text(json.dumps({"spec": {"plan": {"owner_map": {
                    "owner_map_sha256": "a"*64,
                    "helpers": [{"experts": [[1, 0]]}],
                }}}}))
                return argparse.Namespace(returncode=0, stderr="")
            with (patch.object(inv, "resolve", return_value=selected),
                  patch.object(inv, "_run", return_value=doc),
                  patch.object(launch.subprocess, "run", side_effect=fake_run)):
                plan = launch._collect(args)
                self.assertEqual(plan.helpers[0].bdf, helper.bdf)
                self.assertEqual(plan.helpers[0].expert_budget_mib, 20000)
                self.assertGreater(plan.host_pool_mib, 0)
                self.assertEqual(plan.owner_map_digest, "a"*64)
                self.assertEqual(plan.host_available_bytes, 60000 * launch._MIB)
                smaller = dict(doc["facts"]["host"])
                smaller["mem_available_mib"] = 200
                doc["facts"]["host"] = smaller
                with self.assertRaisesRegex(ValueError, "host RAM admission refused"):
                    launch._collect(args)


if __name__ == "__main__":
    unittest.main()
