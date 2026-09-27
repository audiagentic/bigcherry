from __future__ import annotations

import json
import unittest

from bigcherry.jobs.executor import (
    Allocation,
    ExecutionHandle,
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
    parse_correlation,
    parse_sacct_pipe,
    parse_squeue_json,
)


class QueueRunner:
    def __init__(self, results: list[CommandResult]) -> None:
        self.results = list(results)
        self.calls = []

    def run(self, argv, *, cwd=None, env=None, input_text=None):
        self.calls.append((tuple(argv), cwd, None if env is None else dict(env), input_text))
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
    def test_build_argv_uses_no_account_by_default(self):
        argv = build_sbatch_argv(request(), SlurmPolicy())
        self.assertEqual(argv[:2], ("sbatch", "--parsable"))
        self.assertNotIn("--account", argv)
        self.assertIn("bc-build", argv)
        self.assertIn("build_slot:1,host_activity:1", argv)
        self.assertIn("00:15:01", argv)
        self.assertIn("--mem", argv)
        self.assertNotIn("--gres", argv)
        self.assertEqual(argv[-1], "/tmp/attempt/launch.sh")

    def test_explicit_account_is_opt_in(self):
        argv = build_sbatch_argv(request(), SlurmPolicy(account="research"))
        self.assertEqual(argv[argv.index("--account") + 1], "research")

    def test_timed_gpu_argv_has_architecture_count_only(self):
        argv = build_sbatch_argv(request(activity="timed-measure", gpu=SchedulerGpuRequest("gfx1100", 2)), SlurmPolicy())
        self.assertEqual(argv[argv.index("--gres") + 1], "gpu:gfx1100:2")
        self.assertNotIn("gfx1100_0", " ".join(argv))
        self.assertIn("host_activity:2", argv)
        self.assertIn("bc-measure", argv)

    def test_dependency_uses_native_ids_only(self):
        argv = build_sbatch_argv(request(), SlurmPolicy(), native_dependency_ids=("17", "18"))
        self.assertEqual(argv[argv.index("--dependency") + 1], "afterok:17:18")
        with self.assertRaises(ValueError):
            build_sbatch_argv(request(), SlurmPolicy(), native_dependency_ids=("scientific-id",))


class SlurmParsingTests(unittest.TestCase):
    def test_squeue_state_variants_normalize(self):
        payload = json.dumps({"jobs": [{"job_id": 41, "job_state": ["PENDING"], "state_reason": "JobHeldUser"}]})
        self.assertEqual(parse_squeue_json(payload, "41").state, ExecutionState.HELD)
        payload2 = json.dumps({"jobs": [{"job_id": 42, "job_state": {"current": "RUNNING"}, "state_reason": "None"}]})
        self.assertEqual(parse_squeue_json(payload2, "42").state, ExecutionState.RUNNING)

    def test_unknown_state_does_not_guess(self):
        self.assertEqual(normalize_slurm_state("FUTURE_STATE").state, ExecutionState.UNKNOWN)

    def test_sacct_parser_remains_compat_only(self):
        self.assertEqual(parse_sacct_pipe("77|COMPLETED\n", "77").state, ExecutionState.COMPLETED)

    def test_correlation_uses_exact_comment(self):
        payload = json.dumps({"jobs": [
            {"job_id": 12, "comment": "bigcherry:exec-1"},
            {"job_id": 13, "comment": "bigcherry:exec-10"},
        ]})
        self.assertEqual(parse_correlation(payload, "exec-1"), ("12",))


class SlurmExecutorTests(unittest.TestCase):
    def test_submit_parses_parsable_job_id_and_exports_environment(self):
        runner = QueueRunner([CommandResult(0, "123;bigcherry\n", "")])
        executor = SlurmExecutor(runner=runner)
        handle = executor.submit(request(env=(("BIGCHERRY_RUN_ID", "r1"),)))
        self.assertEqual(handle.native_id, "123")
        argv, cwd, env, _ = runner.calls[0]
        self.assertEqual(argv[0], "sbatch")
        self.assertEqual(cwd, "/tmp/attempt")
        self.assertEqual(env["BIGCHERRY_RUN_ID"], "r1")

    def test_dependency_resolver_is_adapter_boundary(self):
        runner = QueueRunner([CommandResult(0, "124\n", "")])
        executor = SlurmExecutor(runner=runner, dependency_resolver=lambda execution_id: {"prepare": "8"}[execution_id])
        executor.submit(request(dependencies=("prepare",)))
        argv = runner.calls[0][0]
        self.assertEqual(argv[argv.index("--dependency") + 1], "afterok:8")

    def test_status_does_not_depend_on_sacct(self):
        runner = QueueRunner([CommandResult(0, json.dumps({"jobs": []}), "")])
        executor = SlurmExecutor(runner=runner)
        status = executor.status(ExecutionHandle("slurm", "123", "x"))
        self.assertEqual(status.state, ExecutionState.UNKNOWN)
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(runner.calls[0][0][:2], ("squeue", "--json"))

    def test_correlate_queries_active_queue(self):
        runner = QueueRunner([CommandResult(0, json.dumps({"jobs": [{"job_id": 99, "comment": "bigcherry:x"}]}), "")])
        handles = SlurmExecutor(runner=runner).correlate("x")
        self.assertEqual(handles, (ExecutionHandle("slurm", "99", "x"),))

    def test_hold_release_cancel_commands(self):
        runner = QueueRunner([CommandResult(0, "", ""), CommandResult(0, "", ""), CommandResult(0, "", "")])
        executor = SlurmExecutor(runner=runner)
        handle = ExecutionHandle("slurm", "99", "x")
        executor.control(handle, ExecutorControl.HOLD)
        executor.control(handle, ExecutorControl.RELEASE)
        executor.cancel(handle)
        self.assertEqual(runner.calls[0][0], ("scontrol", "hold", "99"))
        self.assertEqual(runner.calls[1][0], ("scontrol", "release", "99"))
        self.assertEqual(runner.calls[2][0], ("scancel", "99"))


class FakeExecutorTests(unittest.TestCase):
    def test_dependency_hold_release_cancel_correlation_and_events(self):
        fake = FakeExecutor()
        first = fake.submit(request())
        self.assertEqual(fake.correlate("series:s1/a1"), (first,))
        second = fake.submit(ExecutionRequest(
            execution_id="second", command=("true",), cwd="/tmp", env=(),
            stdout_path="/tmp/o", stderr_path="/tmp/e",
            resources=ResourceRequest(1, None, "build", None, 60),
            dependencies=("series:s1/a1",),
        ))
        self.assertEqual(fake.start_ready(), (first,))
        fake.complete(first)
        self.assertEqual(fake.start_ready(), (second,))
        fake.complete(second)
        self.assertEqual([event.kind for event in fake.events(second)], ["submitted", "started", "completed"])
        third = fake.submit(ExecutionRequest(
            execution_id="third", command=("true",), cwd="/tmp", env=(),
            stdout_path="/tmp/o3", stderr_path="/tmp/e3",
            resources=ResourceRequest(1, None, "build", None, 60),
        ))
        fake.control(third, ExecutorControl.HOLD)
        self.assertEqual(fake.status(third).state, ExecutionState.HELD)
        fake.control(third, ExecutorControl.RELEASE)
        fake.set_allocation(third, Allocation(("native0",)))
        self.assertEqual(fake.allocation(third).native_gpu_ids, ("native0",))
        fake.cancel(third)
        self.assertEqual(fake.status(third).state, ExecutionState.CANCELLED)


if __name__ == "__main__":
    unittest.main()
