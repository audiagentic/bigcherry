from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bigcherry.cli.hardware import main as hardware_main
from bigcherry.hardware.drift import classify_drift
from bigcherry.hardware.inventory import (
    HardwareBindingError,
    InventoryCatalog,
    bind_gpu_requirement,
    verify_series_allocation,
)
from bigcherry.hardware.model import DeviceRecord, HardwareInventory
from bigcherry.hardware.slurm import GresRenderError, render_gres_conf, render_node_gres
from bigcherry.jobs.executor import Allocation
from bigcherry.jobs.model import GpuRequirement


def inventory(*, moved=False, peer=True, second_id="gpu-b") -> HardwareInventory:
    return HardwareInventory(
        host_id="brutus",
        platform_family="linux-rocm",
        platform_environment_hash="env1",
        devices=(
            DeviceRecord(
                "gpu-a",
                "amd_uuid",
                "gfx1100",
                "7900XTX",
                24 * 1024**3,
                pci_bdf="0000:01:00.0" if not moved else "0000:03:00.0",
                render_node="/dev/dri/renderD128" if not moved else "/dev/dri/renderD130",
                numa_node=0,
                launch_ordinal=0,
            ),
            DeviceRecord(
                second_id,
                "amd_uuid",
                "gfx1100",
                "7900XTX",
                24 * 1024**3,
                pci_bdf="0000:02:00.0",
                render_node="/dev/dri/renderD129",
                numa_node=0,
                launch_ordinal=1,
            ),
        ),
        peer_access=(("gpu-a", second_id),) if peer else (),
    )


class InventoryVerificationTests(unittest.TestCase):
    def test_exact_series_allocation_verification(self):
        inv = inventory()
        binding = bind_gpu_requirement(GpuRequirement("gfx1100", 1), inv)
        self.assertTrue(binding.reserve_all_of_arch)
        selected = verify_series_allocation(
            binding,
            Allocation(
                native_gpu_ids=("0", "1"),
                stable_gpu_ids=("gpu-a", "gpu-b"),
            ),
            inv,
        )
        self.assertEqual(selected, ("gpu-a",))

    def test_native_ids_without_stable_attestation_fail(self):
        inv = inventory()
        binding = bind_gpu_requirement(GpuRequirement("gfx1100", 1), inv)
        with self.assertRaises(HardwareBindingError):
            verify_series_allocation(binding, Allocation(("0", "1")), inv)

    def test_allocation_missing_bound_card_fails(self):
        inv = inventory()
        binding = bind_gpu_requirement(GpuRequirement("gfx1100", 1), inv)
        with self.assertRaises(HardwareBindingError):
            verify_series_allocation(
                binding,
                Allocation(native_gpu_ids=("1",), stable_gpu_ids=("gpu-b",)),
                inv,
            )

    def test_inventory_drift_fails_existing_binding(self):
        inv = inventory()
        binding = bind_gpu_requirement(GpuRequirement("gfx1100", 1), inv)
        with self.assertRaises(HardwareBindingError):
            verify_series_allocation(
                binding,
                Allocation(
                    native_gpu_ids=("0", "1"),
                    stable_gpu_ids=("gpu-a", "gpu-b"),
                ),
                inventory(moved=True),
            )


class InventoryCatalogTests(unittest.TestCase):
    def test_observed_accept_requires_reviewed_hash_and_explicit_material_change(self):
        with tempfile.TemporaryDirectory() as temp:
            catalog = InventoryCatalog(Path(temp))
            initial = inventory()
            catalog.record_observed("brutus", initial)
            accepted = catalog.accept_observed(
                "brutus", expected_material_hash=initial.material_hash
            )
            self.assertEqual(accepted.material_hash, initial.material_hash)
            self.assertEqual(catalog.drift("brutus").kind, "same")

            moved = inventory(moved=True)
            catalog.record_observed("brutus", moved)
            with self.assertRaises(HardwareBindingError):
                catalog.accept_observed(
                    "brutus", expected_material_hash=moved.material_hash
                )
            with self.assertRaises(HardwareBindingError):
                catalog.accept_observed(
                    "brutus",
                    expected_material_hash="not-the-observed-hash",
                    allow_material_change=True,
                )
            accepted2 = catalog.accept_observed(
                "brutus",
                expected_material_hash=moved.material_hash,
                allow_material_change=True,
            )
            self.assertEqual(accepted2.material_hash, moved.material_hash)

    def test_hardware_cli_records_accepts_and_renders(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "inventory.json"
            source.write_text(
                json.dumps(dataclasses.asdict(inventory())), encoding="utf-8"
            )
            env = {"BIGCHERRY_HARDWARE_ROOT": str(root / "state")}
            output = io.StringIO()
            with mock.patch.dict(os.environ, env, clear=False), contextlib.redirect_stdout(output):
                self.assertEqual(
                    hardware_main(["record-observed", "brutus", str(source)]), 0
                )
            record = json.loads(output.getvalue())
            observed_hash = record["material_hash"]

            output = io.StringIO()
            with mock.patch.dict(os.environ, env, clear=False), contextlib.redirect_stdout(output):
                self.assertEqual(
                    hardware_main(
                        [
                            "accept",
                            "brutus",
                            "--expected-hash",
                            observed_hash,
                        ]
                    ),
                    0,
                )
            self.assertTrue(json.loads(output.getvalue())["accepted"])

            output = io.StringIO()
            with mock.patch.dict(os.environ, env, clear=False), contextlib.redirect_stdout(output):
                self.assertEqual(hardware_main(["render-gres", "brutus"]), 0)
            rendered = json.loads(output.getvalue())
            self.assertEqual(rendered["node_gres"], "gpu:gfx1100:2")
            self.assertIn("Type=gfx1100", rendered["gres_conf"])


class GresTests(unittest.TestCase):
    def test_gres_is_architecture_typed_and_deterministic(self):
        inv = inventory()
        rendered = render_gres_conf(inv)
        self.assertIn("Type=gfx1100", rendered)
        self.assertNotIn("gpu-a", rendered)
        self.assertEqual(render_node_gres(inv), "gpu:gfx1100:2")
        reordered = HardwareInventory(
            host_id=inv.host_id,
            platform_family=inv.platform_family,
            platform_environment_hash=inv.platform_environment_hash,
            devices=tuple(reversed(inv.devices)),
            peer_access=inv.peer_access,
        )
        self.assertEqual(render_gres_conf(inv), render_gres_conf(reordered))

    def test_missing_render_node_fails_closed(self):
        inv = HardwareInventory(
            host_id="x",
            platform_family="linux-rocm",
            devices=(DeviceRecord("id", "weak", "gfx1", "gpu", 1),),
        )
        with self.assertRaises(GresRenderError):
            render_gres_conf(inv)


class DriftTests(unittest.TestCase):
    def test_locator_move_is_material_not_new_cohort_when_topology_same(self):
        report = classify_drift(inventory(), inventory(moved=True))
        self.assertEqual(report.kind, "locator-only")
        self.assertTrue(report.material)
        self.assertFalse(report.requires_new_cohort)

    def test_peer_change_requires_new_cohort(self):
        report = classify_drift(inventory(), inventory(peer=False))
        self.assertEqual(report.kind, "topology-change")
        self.assertTrue(report.requires_new_cohort)

    def test_replacement_requires_new_cohort(self):
        report = classify_drift(inventory(), inventory(second_id="gpu-c"))
        self.assertEqual(report.kind, "device-replacement")
        self.assertTrue(report.requires_new_cohort)


if __name__ == "__main__":
    unittest.main()
