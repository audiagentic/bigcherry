from __future__ import annotations

import tempfile
import unittest
from pathlib import Path, PurePosixPath

from admin.install_bigcherry_jobs import unit_texts
from bigcherry.hardware.inventory import HardwareBindingError, bind_gpu_requirement
from bigcherry.hardware.model import DeviceRecord, HardwareInventory
from bigcherry.jobs.fake import FakeExecutor
from bigcherry.jobs.identity import StaticScientificIdentityResolver
from bigcherry.jobs.model import BatchSpec, GpuRequirement, TargetPolicy
from bigcherry.jobs.remote import FakeRemoteTransport, RemoteExecutor
from bigcherry.jobs.runner import campaign_argv
from bigcherry.jobs.service import JobService, JobServiceError
from bigcherry.jobs.store import IdempotencyConflict, RunStore
from bigcherry.jobs.workspace import PassthroughWorkspaceManager


def inventory(*, models=("7900XTX", "7900XTX")) -> HardwareInventory:
    devices = tuple(
        DeviceRecord(
            device_id=f"gpu-{index}",
            identity_source="fixture",
            architecture="gfx1100",
            model=model,
            vram_bytes=24 * 1024**3,
            launch_ordinal=index,
        )
        for index, model in enumerate(models)
    )
    return HardwareInventory(
        host_id="brutus",
        platform_family="linux-rocm",
        devices=devices,
        peer_access=(("gpu-0", "gpu-1"),),
        platform_environment_hash="env-linux-rocm",
    )


class HardwareBindingTests(unittest.TestCase):
    def test_binding_is_deterministic_and_subset_reserves_arch_pool(self):
        inv = inventory()
        req = GpuRequirement("gfx1100", count=1)
        first = bind_gpu_requirement(req, inv)
        reordered = HardwareInventory(
            host_id=inv.host_id,
            platform_family=inv.platform_family,
            devices=tuple(reversed(inv.devices)),
            peer_access=inv.peer_access,
            platform_environment_hash=inv.platform_environment_hash,
        )
        second = bind_gpu_requirement(req, reordered)
        self.assertEqual(first.selected_device_ids, ("gpu-0",))
        self.assertEqual(first.selected_device_ids, second.selected_device_ids)
        self.assertTrue(first.reserve_all_of_arch)
        self.assertEqual(first.scheduler_count, 2)

    def test_mixed_model_is_ambiguous(self):
        with self.assertRaises(HardwareBindingError):
            bind_gpu_requirement(
                GpuRequirement("gfx1100", 1), inventory(models=("A", "B"))
            )

    def test_peer_pair(self):
        binding = bind_gpu_requirement(
            GpuRequirement("gfx1100", 2, require_peer_access=True), inventory()
        )
        self.assertEqual(binding.selected_device_ids, ("gpu-0", "gpu-1"))
        self.assertFalse(binding.reserve_all_of_arch)


class StoreAndServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        (self.project / "tools").mkdir(parents=True)
        self.fake = FakeExecutor()
        self.store = RunStore(self.root / "jobs")
        self.identity = StaticScientificIdentityResolver()
        self.service = JobService(
            store=self.store,
            executors={"brutus": self.fake},
            workspace_manager=PassthroughWorkspaceManager(
                self.project, commit="a" * 40
            ),
            inventory_loader=lambda executor_id: inventory(),
            identity_resolver=self.identity,
            attempt_identity_resolver_factory=lambda root: self.identity,
            shared_root=self.root / "shared",
        )

    def tearDown(self):
        self.temp.cleanup()

    def batch(self, sessions=2):
        return BatchSpec(
            patch="1265_example",
            architectures=("gfx1100",),
            model="/models/model.gguf",
            planned_sessions=sessions,
            producer="1265_example/producer",
            target=TargetPolicy(
                executor_id="brutus",
                host_id="brutus",
                platform_family="linux-rocm",
            ),
        )

    def test_plan_submit_ingest_complete_review(self):
        batch = self.batch()
        self.assertEqual(self.service.plan_batch(batch), self.service.plan_batch(batch))
        result = self.service.submit_batch(
            batch, idempotency_key="wave-1", actor="agent"
        )
        pending_before = self.store.pending()
        events_before = self.store.read_events()
        replay = self.service.submit_batch(
            batch, idempotency_key="wave-1", actor="different-client"
        )
        self.assertEqual(result, replay)
        self.assertEqual(self.store.pending(), pending_before)
        self.assertEqual(self.store.read_events(), events_before)
        self.assertEqual(len(self.store.list_runs()), 2)
        ingest = self.service.ingest_once()
        self.assertEqual(ingest["accepted"], 2)
        started = self.fake.start_ready()
        self.assertEqual(len(started), 2)
        for handle in started:
            self.fake.complete(handle)
        statuses = self.service.list_status()
        self.assertEqual({item["state"] for item in statuses}, {"completed"})
        series_id = self.store.list_series()[0]["series_id"]
        review = self.service.review(series_id)
        self.assertTrue(review["execution_complete"])
        self.assertFalse(review["review_ready"])
        self.assertEqual(review["completed_sessions"], 2)

    def test_idempotency_key_conflict(self):
        self.service.submit_batch(self.batch(), idempotency_key="same")
        with self.assertRaises(IdempotencyConflict):
            self.service.submit_batch(
                self.batch(sessions=3), idempotency_key="same"
            )

    def test_disable_defers_then_enable_runs(self):
        self.service.submit_batch(
            self.batch(sessions=1), idempotency_key="disable"
        )
        run_id = self.store.list_runs()[0]["run_id"]
        self.service.control_run(run_id, "disable")
        self.assertEqual(self.service.ingest_once()["deferred"], 1)
        self.service.control_run(run_id, "enable")
        self.assertEqual(self.service.ingest_once()["accepted"], 1)

    def test_processing_receipt_survives_service_crash_without_new_attempt(self):
        self.service.submit_batch(self.batch(sessions=1), idempotency_key="crash")
        run_id = self.store.list_runs()[0]["run_id"]
        processing = self.store.claim_pending(self.store.pending()[0])
        first_handle = self.service._start_or_recover(run_id)
        self.assertTrue(processing.is_file())
        self.assertEqual(self.store.latest_attempt(run_id), 1)
        restarted = JobService(
            store=self.store,
            executors={"brutus": self.fake},
            workspace_manager=PassthroughWorkspaceManager(
                self.project, commit="a" * 40
            ),
            inventory_loader=lambda executor_id: inventory(),
            identity_resolver=self.identity,
            attempt_identity_resolver_factory=lambda root: self.identity,
            shared_root=self.root / "shared",
        )
        result = restarted.ingest_once()
        self.assertEqual(result["accepted"], 1)
        self.assertEqual(self.store.latest_attempt(run_id), 1)
        self.assertEqual(self.fake.correlate(first_handle.execution_id), (first_handle,))

    def test_retry_same_commit_creates_new_attempt(self):
        self.service.submit_batch(self.batch(sessions=1), idempotency_key="retry")
        run_id = self.store.list_runs()[0]["run_id"]
        self.service.ingest_once()
        handle = self.fake.start_ready()[0]
        self.fake.complete(handle, failed=True)
        self.service.retry(run_id, latest=False)
        self.service.ingest_once()
        self.assertEqual(self.store.latest_attempt(run_id), 2)
        self.assertEqual(
            self.store.attempt(run_id, 1)["commit"],
            self.store.attempt(run_id, 2)["commit"],
        )

    def test_scientific_identity_drift_blocks_new_attempt(self):
        self.service.submit_batch(self.batch(sessions=1), idempotency_key="drift")
        run_id = self.store.list_runs()[0]["run_id"]
        drifted = StaticScientificIdentityResolver(
            {"schema": "mock", "identity_hash": "different"}
        )
        service = JobService(
            store=self.store,
            executors={"brutus": self.fake},
            workspace_manager=PassthroughWorkspaceManager(
                self.project, commit="a" * 40
            ),
            inventory_loader=lambda executor_id: inventory(),
            identity_resolver=self.identity,
            attempt_identity_resolver_factory=lambda root: drifted,
            shared_root=self.root / "shared",
        )
        self.assertEqual(service.ingest_once()["rejected"], 1)
        events = self.store.read_events()
        self.assertTrue(any("scientific identity drifted" in str(e.data) for e in events))

    def test_events_recover_torn_tail(self):
        self.store.append_event(kind="one")
        with self.store.event_path.open("ab") as handle:
            handle.write(b'{"broken":')
        self.store.append_event(kind="two")
        self.assertEqual(
            [event.seq for event in self.store.read_events()], [1, 2]
        )


class RunnerTests(unittest.TestCase):
    def test_device_map_is_allocation_local_not_physical_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = {
                "patch": "p",
                "architecture": "gfx1100",
                "model": "/m.gguf",
                "baseline_source": "bigcherry-tuning",
                "gpu": {"count": 2},
                "producer_inputs": [],
                "common_patches": [],
            }
            argv = campaign_argv(
                job, attempt_root=root / "attempt", shared_root=root / "shared"
            )
            self.assertEqual(
                argv[argv.index("--device-map") + 1], "gfx1100=0,1"
            )
            self.assertNotIn("gpu-0", " ".join(argv))


class RemoteAndInstallTests(unittest.TestCase):
    def test_remote_executor_protocol_against_fake_worker(self):
        fake = FakeExecutor()
        remote = RemoteExecutor("windows-rdna3", FakeRemoteTransport(fake))
        from bigcherry.jobs.executor import ExecutionRequest, ResourceRequest

        request = ExecutionRequest(
            execution_id="remote-1",
            command=("python", "-V"),
            cwd=".",
            env=(),
            stdout_path="out",
            stderr_path="err",
            resources=ResourceRequest(1, None, "build", None, 60),
        )
        handle = remote.submit(request)
        self.assertEqual(remote.correlate("remote-1"), (handle,))
        fake.start_ready()
        self.assertEqual(remote.status(handle).state.value, "running")
        fake.complete(fake.correlate("remote-1")[0])
        self.assertEqual(remote.status(handle).state.value, "completed")

    def test_systemd_render_is_host_parameterized(self):
        units = unit_texts(
            project_root=PurePosixPath("/srv/bigcherry"),
            work_root=PurePosixPath("/mnt/data/bc"),
            python=PurePosixPath("/srv/venv/bin/python"),
            user="bigcherry",
        )
        service = units["bigcherry-jobs-ingest.service"]
        path = units["bigcherry-jobs-ingest.path"]
        self.assertIn("User=bigcherry", service)
        self.assertIn("-m bigcherry jobs ingest --once", service)
        self.assertIn("/mnt/data/bc/jobs/inbox/pending", path)


if __name__ == "__main__":
    unittest.main()
