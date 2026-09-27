from __future__ import annotations

import unittest

from bigcherry.hardware.model import DeviceRecord, HardwareInventory
from bigcherry.jobs.acceptance import acceptance_document, acceptance_matrix
from bigcherry.jobs.migration import LegacyMigrationError, migrate_run_campaign_command
from bigcherry.jobs.model import BatchSpec, TargetPolicy


def inventory() -> HardwareInventory:
    return HardwareInventory(
        host_id="brutus",
        platform_family="linux-rocm",
        platform_environment_hash="env",
        devices=(
            DeviceRecord("gpu-a", "amd-smi-uuid", "gfx1100", "XTX", 24 << 30),
            DeviceRecord("gpu-b", "amd-smi-uuid", "gfx1100", "XTX", 24 << 30),
            DeviceRecord("gpu-c", "amd-smi-uuid", "gfx1201", "R9700", 32 << 30),
        ),
        peer_access=(("gpu-a", "gpu-b"),),
    )


class AcceptanceMatrixTests(unittest.TestCase):
    def test_matrix_is_capability_derived_and_keeps_unsupported(self):
        batches = (
            BatchSpec(
                patch="p",
                architectures=("gfx1100",),
                model="/models/a.gguf",
                gpu_count=2,
                require_peer_access=True,
                target=TargetPolicy("brutus", "brutus", "linux-rocm"),
            ),
            BatchSpec(
                patch="q",
                architectures=("gfx1030",),
                model="/models/b.gguf",
                target=TargetPolicy("brutus", "brutus", "linux-rocm"),
            ),
        )
        cases = acceptance_matrix(batches, inventories={"brutus": inventory()})
        self.assertEqual(len(cases), 2)
        supported = [case for case in cases if case.supported]
        unsupported = [case for case in cases if not case.supported]
        self.assertEqual(supported[0].selected_device_ids, ("gpu-a", "gpu-b"))
        self.assertIn("no homogeneous GPU cohort", unsupported[0].reason)
        doc = acceptance_document(cases)
        self.assertEqual(doc["supported"], 1)
        self.assertEqual(doc["unsupported"], 1)
        self.assertFalse(doc["ready"])

    def test_duplicate_capability_case_is_deduplicated(self):
        batch = BatchSpec(
            patch="p",
            architectures=("gfx1201",),
            model="/models/a.gguf",
            target=TargetPolicy("brutus", "brutus", "linux-rocm"),
        )
        cases = acceptance_matrix((batch, batch), inventories={"brutus": inventory()})
        self.assertEqual(len(cases), 1)
        self.assertTrue(cases[0].supported)
        self.assertEqual(cases[0].selected_device_ids, ("gpu-c",))


class LegacyMigrationTests(unittest.TestCase):
    def test_wrapper_migration_drops_slot_and_preserves_scientific_args(self):
        result = migrate_run_campaign_command(
            "tools/lab/plan-qualification/run_campaign.sh p p/producer gfx1100 1 old-run "
            "--common-patches dep1,dep2 --producer-input corpus=/x/corpus.json "
            "--producer-corpus /x/text.txt --production-lane",
            environment={"BC_MODEL": "/models/a.gguf", "BC_HIP_PATH": "/opt/rocm"},
            planned_sessions=4,
            target=TargetPolicy("brutus", "brutus", "linux-rocm"),
        )
        self.assertEqual(result.legacy_device, "1")
        self.assertEqual(result.legacy_run_name, "old-run")
        self.assertEqual(result.batch.architectures, ("gfx1100",))
        self.assertEqual(result.batch.common_patches, ("dep1", "dep2"))
        self.assertEqual(result.batch.producer_inputs, (("corpus", "/x/corpus.json"),))
        self.assertTrue(result.batch.production_lane)
        self.assertEqual(result.batch.planned_sessions, 4)

    def test_unknown_legacy_argument_fails_instead_of_guessing(self):
        with self.assertRaises(LegacyMigrationError):
            migrate_run_campaign_command(
                "run_campaign.sh p - gfx1100 0 old --allow-rejected",
                environment={"BC_MODEL": "/models/a.gguf", "BC_HIP_PATH": "/opt/rocm"},
                planned_sessions=4,
            )

    def test_planned_n_is_never_inferred(self):
        with self.assertRaises(LegacyMigrationError):
            migrate_run_campaign_command(
                "run_campaign.sh p - gfx1100 0 old",
                environment={"BC_MODEL": "/models/a.gguf", "BC_HIP_PATH": "/opt/rocm"},
                planned_sessions=0,
            )


if __name__ == "__main__":
    unittest.main()
