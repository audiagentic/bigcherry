from __future__ import annotations

import json
import unittest

from bigcherry.jobs.executor import (
    Allocation,
    ExecutionRequest,
    ExecutionState,
    ExecutorControl,
    ResourceRequest,
    SchedulerGpuRequest,
)
from bigcherry.jobs.fake import FakeExecutor
from bigcherry.jobs.slurm import (
    CommandResult,
    SlurmExecutor,
    SlurmPolicy,
    build_sbatch_argv,
    normalize_slurm_state,
    parse_sacct_pipe,
    parse_squeue_json,
)


class QueueRunner:
    def __init__(self, results: list[CommandResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[tuple[str, ...], str | None, dict[str, str] | None]] = []

    def run(self, argv, *, cwd=None, env=None):
        self.calls.append((tuple(argv), cwd, None if env is None else dict(env)))
        if not self.results:
            raise AssertionError(f"unexpected runner call: {argv}")
        return self.results.pop(0)


def request(*, activity="build", gpu=None, dependencies=(), env=()):
    return ExecutionRequest(
        execution_id="series:s1/a1",
        command=("/tmp/attempt/launch.sh",),
        cwd="/tmp/attempt",
        env=tuple(env),
        stdout_path="/tmp/attempt/slurm-%j.out",
        stderr_path="/tmp/attempt/slurm-%j.err",
        resources=ResourceRequest(
            cpu_slots=4,
            gpu=gpu,
            activity_class=activity,
            memory_bytes=3 * 1024 * 1024 + 1,
            timeout_seconds=901,
        ),
        dependencies=tuple(dependencies),
    )


class SlurmRenderingTests(unittest.TestCase):
    def test_build_argv_is_resolved_scheduler_request(self):
        argv = build_sbatch_argv(request(), SlurmPolicy())
        self.assertEqual(argv[:5], ("sbatch", "--parsable", "--account", "bigcherry", "--job-name"))
        self.assertIn("bc-build", argv)
        self.assertIn("build_slot:1,host_activity:1", argv)
        self.assertIn("00:15:01", argv)
        self.assertIn("4", argv)
        self.assertIn("--mem", argv)
        self.assertIn("4", argv)  # 3 MiB + 1 byte -> 4 MiB
        self.assertNotIn("--gres", argv)
        self.assertEqual(argv[-1], "/tmp/attempt/launch.sh")

    def test_timed_gpu_argv_has_architecture_count_only(self):
        argv = build_sbatch_argv(
            request(activity="timed-measure", gpu=SchedulerGpuRequest("gfx1100", 2)),
            SlurmPolicy(),
        )
        i = argv.index("--gres")
        self.assertEqual(argv[i + 1], "gpu:gfx1100:2")
        self.assertNotIn("gfx1100_0", " ".join(argv))
        self.assertIn("host_activity:2", argv)
        self.assertIn("bc-measure", argv)

    def test_dependency_uses_native_ids_only(self):
        argv = build_sbatch_argv(request(), SlurmPolicy(), native_dependency_ids=("17", "18"))
        i = argv.index("--dependency")
        self.assertEqual(argv[i + 1], "afterok:17:18")
        with self.assertRaises(ValueError):
            build_sbatch_argv(request(), SlurmPolicy(), native_dependency_ids=("scientific-id",))


class SlurmParsingTests(unittest.TestCase):
    def test_squeue_state_variants_normalize(self):
        payload = json.dumps({
            "jobs": [
                {"job_id": 41, "job_state": ["PENDING"], "state_reason": "JobHeldUser"}
            ]
        })
        status = parse_squeue_json(payload, "41")
        self.assertIsNotNone(status)
        self.assertEqual(status.state, ExecutionState.HELD)

        payload2 = json.dumps({
            "jobs": [{"job_id": 42, "job_state": {"current": "RUNNING"}, "state_reason": "None"}]
        })
        self.assertEqual(parse_squeue_json(payload2, "42").state, ExecutionState.RUNNING)

    def test_unknown_state_does_not_guess(self):
        self.assertEqual(normalize_slurm_state("FUTURE_STATE").state, ExecutionState.UNKNOWN)

    def test_sacct_completed_fallback(self):
        self.assertEqual(parse_sacct_pipe("77|COMPLETED\n", "77").state, ExecutionState.COMPLETED)
        self.assertEqual(parse_sacct_pipe("77|CANCELLED by 1000\n", "77").state, ExecutionState.CANCELLED)


class SlurmExecutorTests(unittest.TestCase):
    def test_submit_parses_parsable_job_id_and_exports_environment(self):
        runner = QueueRunner([CommandResult(0, "123;bigcherry\n", "")])
        executor = SlurmExecutor(runner=runner)
        handle = executor.submit(request(env=(("BIGCHERRY_RUN_ID", "r1"),)))
        self.assertEqual(handle.native_id, "123")
        argv, cwd, env = runner.calls[0]
        self.assertEqual(argv[0], "sbatch")
        self.assertEqual(cwd, "/tmp/attempt")
        self.assertEqual(env["BIGCHERRY_RUN_ID"], "r1")

    def test_dependency_resolver_is_adapter_boundary(self):
        runner = QueueRunner([CommandResult(0, "124\n", "")])
        executor = SlurmExecutor(runner=runner, dependency_resolver=lambda execution_id: {"prepare": "8"}[execution_id])
        executor.submit(request(dependencies=("prepare",)))
        argv = runner.calls[0][0]
        self.assertEqual(argv[argv.index("--dependency") + 1], "afterok:8")

    def test_status_uses_squeue_then_sacct(self):
        runner = QueueRunner([
            CommandResult(0, json.dumps({"jobs": []}), ""),
            CommandResult(0, "123|COMPLETED\n", ""),
        ])
        executor = SlurmExecutor(runner=runner)
        handle = executor.submit.__annotations__  # keep static tools from treating handle as magic
        from bigcherry.jobs.executor import ExecutionHandle
        status = executor.status(ExecutionHandle("slurm", "123", "x"))
        self.assertEqual(status.state, ExecutionState.COMPLETED)
        self.assertEqual(runner.calls[0][0][:2], ("squeue", "--json"))
        self.assertEqual(runner.calls[1][0][0], "sacct")

    def test_hold_release_cancel_commands(self):
        from bigcherry.jobs.executor import ExecutionHandle
        runner = QueueRunner([
            CommandResult(0, "", ""), CommandResult(0, "", ""), CommandResult(0, "", "")
        ])
        executor = SlurmExecutor(runner=runner)
        handle = ExecutionHandle("slurm", "99", "x")
        executor.control(handle, ExecutorControl.HOLD)
        executor.control(handle, ExecutorControl.RELEASE)
        executor.cancel(handle)
        self.assertEqual(runner.calls[0][0], ("scontrol", "hold", "99"))
        self.assertEqual(runner.calls[1][0], ("scontrol", "release", "99"))
        self.assertEqual(runner.calls[2][0], ("scancel", "99"))


class FakeExecutorTests(unittest.TestCase):
    def test_dependency_hold_release_cancel_and_events(self):
        fake = FakeExecutor()
        first = fake.submit(request())
        second = fake.submit(ExecutionRequest(
            execution_id="second", command=("true",), cwd="/tmp", env=(),
            stdout_path="/tmp/o", stderr_path="/tmp/e",
            resources=ResourceRequest(1, None, "build", None, 60),
            dependencies=("series:s1/a1",),
        ))
        self.assertEqual(fake.start_ready(), (first,))
        self.assertEqual(fake.status(second).state, ExecutionState.QUEUED)
        fake.complete(first)
        self.assertEqual(fake.start_ready(), (second,))
        fake.control(second, ExecutorControl.HOLD)  # running hold is a no-op by contract
        self.assertEqual(fake.status(second).state, ExecutionState.RUNNING)
        fake.complete(second)
        events = tuple(fake.events(second))
        self.assertEqual([event.kind for event in events], ["submitted", "started", "completed"])

        third = fake.submit(ExecutionRequest(
            execution_id="third", command=("true",), cwd="/tmp", env=(),
            stdout_path="/tmp/o3", stderr_path="/tmp/e3",
            resources=ResourceRequest(1, None, "build", None, 60),
        ))
        fake.control(third, ExecutorControl.HOLD)
        self.assertEqual(fake.status(third).state, ExecutionState.HELD)
        fake.control(third, ExecutorControl.RELEASE)
        self.assertEqual(fake.status(third).state, ExecutionState.QUEUED)
        fake.set_allocation(third, Allocation(("native0",)))
        self.assertEqual(fake.allocation(third).native_gpu_ids, ("native0",))
        fake.cancel(third)
        self.assertEqual(fake.status(third).state, ExecutionState.CANCELLED)


if __name__ == "__main__":
    unittest.main()
