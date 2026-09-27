from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
LAB = REPO / "tools" / "lab" / "plan-qualification"
HELPER = LAB / "work-root.sh"


def _wait(path: Path, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path}")


def _proc(script: str, work: Path) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["WORK"] = str(work)
    env["HELPER"] = str(HELPER)
    return subprocess.Popen(
        ["bash", "-c", script],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


class PlanQualificationActivityLockTests(unittest.TestCase):
    def test_shell_scripts_parse(self) -> None:
        for name in ("work-root.sh", "profile_run.sh", "run_campaign.sh", "queue.sh"):
            completed = subprocess.run(
                ["bash", "-n", str(LAB / name)],
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_work_root_entrypoint_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            completed = subprocess.run(
                ["bash", str(HELPER), str(REPO)],
                env={**os.environ, "BIGCHERRY_WORK_ROOT": temp},
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.strip(), temp)

    def test_live_queue_scripts_wire_expected_resource_policy(self) -> None:
        helper = HELPER.read_text(encoding="utf-8")
        profile = (LAB / "profile_run.sh").read_text(encoding="utf-8")
        campaign = (LAB / "run_campaign.sh").read_text(encoding="utf-8")
        queue = (LAB / "queue.sh").read_text(encoding="utf-8")

        self.assertIn("gpu_lock_acquire()", helper)
        self.assertIn("activity_lock_shared_acquire()", helper)
        self.assertIn("activity_lock_exclusive_acquire()", helper)
        self.assertIn('build-locks/$arch-$toolchain.lock', profile)
        self.assertIn('flock -x "$build_fd"', profile)
        self.assertIn('flock -s "$build_fd"', profile)
        self.assertIn('--prepare-only --prepared-manifest "$prepared"', profile)
        self.assertIn('--prepared-manifest "$prepared"', profile)
        self.assertLess(
            profile.index('flock -x "$build_fd"'),
            profile.index('activity_lock_shared_acquire "$work"'),
        )
        run_fn = profile.index("run_profile()")
        self.assertLess(
            profile.index('flock -s "$build_fd"', run_fn),
            profile.index('activity_lock_shared_acquire "$work"', run_fn),
        )
        self.assertLess(
            profile.index('activity_lock_shared_acquire "$work"', run_fn),
            profile.index('gpu_lock_acquire "$work" "$dev"', run_fn),
        )
        self.assertIn('echo "PROFILE_EXIT=$rc"', profile)
        self.assertIn('run_profile "$@"\nexit $?', profile)

        campaign_activity = 'activity_lock_exclusive_acquire "$work"'
        campaign_gpu = 'gpu_lock_acquire "$work" "$dev"'
        self.assertLess(campaign.index(campaign_activity), campaign.index(campaign_gpu))
        self.assertIn('echo "CAMPAIGN_EXIT=$rc"\nexit "$rc"', campaign)

        self.assertIn('BC_PROFILE_PHASE=prepare', queue)
        self.assertIn('BC_PROFILE_PHASE=run', queue)
        self.assertIn('ready_profiles+=("${profiles[$i]}")', queue)
        self.assertIn('if ! wait "$pid"; then failures=$((failures + 1)); fi', queue)
        self.assertIn('return "$rc"', queue)
        self.assertIn('if ((failures)); then', queue)
        self.assertLess(queue.index("profile prepare phase:"), queue.index("profile trace phase:"))
        self.assertLess(queue.index("profile trace phase:"), queue.index("campaign phase:"))

    def test_shared_profile_holders_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            release = work / "release"
            first = work / "first"
            second = work / "second"
            p1 = _proc(
                'source "$HELPER"; activity_lock_shared_acquire "$WORK"; '
                'touch "$WORK/first"; while [ ! -e "$WORK/release" ]; do sleep 0.02; done',
                work,
            )
            try:
                _wait(first)
                p2 = _proc(
                    'source "$HELPER"; activity_lock_shared_acquire "$WORK"; touch "$WORK/second"',
                    work,
                )
                try:
                    _wait(second, 1.0)
                    self.assertIsNone(p1.poll(), "first shared holder should still own the gate")
                    self.assertEqual(p2.wait(timeout=2), 0)
                finally:
                    if p2.poll() is None:
                        p2.kill()
                release.touch()
                self.assertEqual(p1.wait(timeout=2), 0)
            finally:
                if p1.poll() is None:
                    p1.kill()

    def test_pending_and_active_exclusive_blocks_new_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            p1 = _proc(
                'source "$HELPER"; activity_lock_shared_acquire "$WORK"; '
                'touch "$WORK/shared1"; while [ ! -e "$WORK/release1" ]; do sleep 0.02; done',
                work,
            )
            exclusive = None
            p2 = None
            try:
                _wait(work / "shared1")
                exclusive = _proc(
                    'source "$HELPER"; activity_lock_exclusive_acquire "$WORK"; '
                    'touch "$WORK/exclusive"; while [ ! -e "$WORK/release-exclusive" ]; do sleep 0.02; done',
                    work,
                )
                _wait(work / "queue" / "activity" / "performance.intent")
                p2 = _proc(
                    'source "$HELPER"; activity_lock_shared_acquire "$WORK"; touch "$WORK/shared2"',
                    work,
                )
                time.sleep(0.15)
                self.assertFalse((work / "shared2").exists(), "pending writer must stop reader barging")

                (work / "release1").touch()
                _wait(work / "exclusive")
                time.sleep(0.15)
                self.assertFalse((work / "shared2").exists(), "active exclusive holder must block profiles")

                (work / "release-exclusive").touch()
                self.assertEqual(exclusive.wait(timeout=2), 0)
                _wait(work / "shared2")
                self.assertEqual(p2.wait(timeout=2), 0)
                self.assertEqual(p1.wait(timeout=2), 0)
            finally:
                for proc in (p1, exclusive, p2):
                    if proc is not None and proc.poll() is None:
                        proc.kill()

    def test_stale_writer_intent_is_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            activity = work / "queue" / "activity"
            activity.mkdir(parents=True)
            intent = activity / "performance.intent"
            intent.write_text("pid=dead\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$HELPER"; activity_lock_shared_acquire "$WORK"; touch "$WORK/acquired"',
                ],
                env={**os.environ, "WORK": str(work), "HELPER": str(HELPER)},
                text=True,
                capture_output=True,
                timeout=3,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((work / "acquired").exists())
            self.assertFalse(intent.exists())


if __name__ == "__main__":
    unittest.main()
