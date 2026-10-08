"""Regression tests for the changed-patch offline checker."""

from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from bigcherry.patch import offline_check


class OfflineCheckAuditTests(unittest.TestCase):
    def test_experiment_scope_is_changed_patch_only(self):
        raw = {
            "experiment": {
                "alpha": {"patches": ["1200_alpha"]},
                "beta": {"patches": ["1201_beta", "1202_shared"]},
                "gamma": {"patches": ["1202_shared"]},
            }
        }
        self.assertEqual(
            offline_check._experiment_names_for_audit(
                raw, ["1201_beta"], full_audit=False
            ),
            ("beta",),
        )
        self.assertEqual(
            offline_check._experiment_names_for_audit(
                raw, ["unused"], full_audit=True
            ),
            ("alpha", "beta", "gamma"),
        )

    def test_referenced_experiment_uses_declared_source(self):
        raw = {
            "source": {
                "bigcherry": {"patch-sets": []},
                "bigcherry-tuning": {"patch-sets": []},
                "bigcherry-qualification": {"patch-sets": []},
                "llama-native": {"patch-sets": []},
            },
            "patch-set": {},
            "experiment": {
                "rd19-only": {"patches": ["1200_rd19"]},
                "hi134-meta-stage-trace": {"patches": ["1325_hi134"]},
                "native-plus-1333": {"patches": ["1333_native"]},
            },
        }
        usage = {
            "rd19-only": {"bigcherry-tuning"},
            "hi134-meta-stage-trace": {"bigcherry-qualification"},
            "native-plus-1333": {"bigcherry"},
        }
        self.assertEqual(
            offline_check._experiment_sources(raw, usage, "rd19-only"),
            ("bigcherry-tuning",),
        )
        self.assertEqual(
            offline_check._experiment_sources(raw, usage, "hi134-meta-stage-trace"),
            ("bigcherry-qualification",),
        )
        self.assertEqual(
            offline_check._experiment_sources(raw, usage, "native-plus-1333"),
            ("llama-native",),
        )

    def test_run_times_out_instead_of_hanging(self):
        checks = []
        with mock.patch.object(
            offline_check.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(["cmd"], timeout=7),
        ):
            rc = offline_check._run(
                "bounded check",
                ["cmd"],
                checks,
                timeout_seconds=7,
            )
        self.assertEqual(rc, 124)
        self.assertEqual(checks[0]["returncode"], 124)
        self.assertIn("timed out after 7s", checks[0]["detail"])

    def test_parallel_audit_collects_all_failures(self):
        raw = {
            "source": {
                "bigcherry": {"patch-sets": []},
                "bigcherry-tuning": {"patch-sets": []},
            },
            "patch-set": {},
            "experiment": {
                "alpha": {"patches": ["1200_alpha"]},
                "beta": {"patches": ["1201_beta"]},
            },
        }
        completed = []

        def fake_run(command, **kwargs):
            name = command[-1]
            completed.append(name)
            return subprocess.CompletedProcess(
                command,
                1 if name == "alpha" else 2,
                stdout=f"{name} failed\n",
            )

        checks = []
        with (
            mock.patch.object(
                offline_check,
                "_experiment_usage",
                return_value={
                    "alpha": {"bigcherry-tuning"},
                    "beta": {"bigcherry-tuning"},
                },
            ),
            mock.patch.object(offline_check.subprocess, "run", side_effect=fake_run),
        ):
            failed = offline_check._run_experiment_audit(
                raw,
                ("alpha", "beta"),
                checks,
                workers=2,
                timeout_seconds=30,
            )

        self.assertTrue(failed)
        self.assertCountEqual(completed, ["alpha", "beta"])
        failures = [
            item for item in checks if str(item["name"]).startswith("experiment ")
        ]
        self.assertEqual(len(failures), 2)
        self.assertCountEqual([item["returncode"] for item in failures], [1, 2])


if __name__ == "__main__":
    unittest.main()
