from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bigcherry.hardware.inventory import (
    HardwareBindingError,
    bind_gpu_requirement,
    verify_series_allocation,
)
from bigcherry.hardware.model import DeviceRecord, HardwareInventory
from bigcherry.hardware.runtime import (
    allocation_from_visible_tokens,
    selected_launch_ordinals,
    selected_local_positions,
)
from bigcherry.jobs.model import GpuRequirement
from bigcherry.jobs.runner import campaign_argv


def inventory() -> HardwareInventory:
    return HardwareInventory(
        host_id="brutus",
        platform_family="linux-rocm",
        platform_environment_hash="env",
        devices=(
            DeviceRecord(
                "uuid-a", "amd_uuid", "gfx1100", "7900XTX", 24 * 1024**3,
                render_node="/dev/dri/renderD128", launch_ordinal=0,
            ),
            DeviceRecord(
                "uuid-b", "amd_uuid", "gfx1100", "7900XTX", 24 * 1024**3,
                render_node="/dev/dri/renderD129", launch_ordinal=1,
            ),
        ),
    )


class RuntimeGpuMappingTests(unittest.TestCase):
    def test_proper_subset_uses_actual_allocation_local_position(self):
        inv = inventory()
        binding = bind_gpu_requirement(
            GpuRequirement("gfx1100", count=1, exact_device_ids=("uuid-b",)), inv
        )
        self.assertTrue(binding.reserve_all_of_arch)
        allocation = allocation_from_visible_tokens("0,1", inv)
        verify_series_allocation(binding, allocation, inv)
        self.assertEqual(allocation.stable_gpu_ids, ("uuid-a", "uuid-b"))
        self.assertEqual(selected_local_positions(binding, allocation), (1,))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = {
                "patch": "p",
                "architecture": "gfx1100",
                "model": "/m.gguf",
                "baseline_source": "bigcherry-tuning",
                "gpu": {"count": 1},
                "producer_inputs": [],
                "common_patches": [],
            }
            argv = campaign_argv(
                job,
                attempt_root=root / "attempt",
                shared_root=root / "shared",
                local_device_indices=(1,),
            )
        self.assertEqual(argv[argv.index("--device-map") + 1], "gfx1100=1")

    def test_uuid_visibility_order_is_preserved(self):
        inv = inventory()
        binding = bind_gpu_requirement(
            GpuRequirement("gfx1100", count=1, exact_device_ids=("uuid-b",)), inv
        )
        allocation = allocation_from_visible_tokens("uuid-b,uuid-a", inv)
        self.assertEqual(allocation.stable_gpu_ids, ("uuid-b", "uuid-a"))
        self.assertEqual(selected_local_positions(binding, allocation), (0,))

    def test_unknown_visible_token_fails_closed(self):
        with self.assertRaises(HardwareBindingError):
            allocation_from_visible_tokens("7", inventory())

    def test_local_frozen_cohort_maps_to_current_accepted_ordinal(self):
        inv = inventory()
        binding = bind_gpu_requirement(
            GpuRequirement("gfx1100", count=1, exact_device_ids=("uuid-b",)), inv
        )
        self.assertEqual(selected_launch_ordinals(binding, inv), (1,))


if __name__ == "__main__":
    unittest.main()
