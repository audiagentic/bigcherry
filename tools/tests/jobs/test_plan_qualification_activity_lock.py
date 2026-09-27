from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
LAB = REPO / "tools" / "lab" / "plan-qualification"
HELPER = LAB / "activity-lock.sh"


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
        for name in (
            "activity-lock.sh",
            "gpu-lock.sh",
            "profile_run.sh",
            "run_campaign.sh",
            "queue.sh",
        ):
            completed = subprocess.run(
                ["bash", "-n", str(LAB / name)],
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_live_queue_scripts_wire_expected_resource_policy(self) -> None:
        profile = (LAB / "profile_run.sh").read_text(encoding="utf-8")
        campaign = (LAB / "run_campaign.sh").read_text(encoding="utf-8")
        queue = (LAB / "queue.sh").read_text(encoding="utf-8")
        build_lock = 'flock "$build_fd"'
        profile_activity = 'activity_lock_shared_acquire "$work"'
        profile_gpu = 'gpu_lock_acquire "$work" "$dev"'
        campaign_activity = 'activity_lock_exclusive_acquire "$work"'
        campaign_gpu = 'gpu_lock_acquire "$work" "$dev"'
        self.assertIn('build-locks/$arch-$toolchain.lock', profile)
        self.assertLess(profile.index(build_lock), profile.index(profile_activity))
        self.assertLess(profile.index(profile_activity), profile.index(profile_gpu))
        self.assertLess(campaign.index(campaign_activity), campaign.index(campaign_gpu))
        self.assertIn('run_line "$line" &', queue)
        self.assertIn('for pid in "${pids[@]}"; do wait "$pid"', queue)
        self.assertLess(queue.index("profile phase:"), queue.index("campaign phase:"))

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
