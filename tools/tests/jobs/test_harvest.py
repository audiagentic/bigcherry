from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from bigcherry.jobs.gitops import GitOpsError
from bigcherry.jobs.harvest import HarvestError, _verify_record_for_run, harvest_series
from bigcherry.jobs.store import RunStore
from bigcherry.patch import evidence as patch_evidence


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def evidence_record(*, digest_char: str) -> dict[str, object]:
    record: dict[str, object] = {
        "patch_id": "p",
        "campaign_identity_digest": digest_char * 64,
        "patch_implementation_digest": "a" * 64,
        "gpu_architectures": ["gfx1100"],
        "lane_effects": [],
    }
    record["record_digest"] = patch_evidence._record_digest(record)  # type: ignore[attr-defined]
    return record


class HarvestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        git(self.repo, "init")
        git(self.repo, "config", "user.email", "jobs@example.invalid")
        git(self.repo, "config", "user.name", "BigCherry Jobs Test")
        (self.repo / "engines" / "llamacpp" / "patches" / "p").mkdir(parents=True)
        (self.repo / "engines" / "llamacpp" / "patches" / "p" / "patch.py").write_text("STATE='untested'\n")
        git(self.repo, "add", "--", "engines/llamacpp/patches/p/patch.py")
        git(self.repo, "commit", "-m", "base")
        self.commit = git(self.repo, "rev-parse", "HEAD")
        self.store = RunStore(self.root / "jobs")
        self.series_id = "s-test"
        self.store.create_series(
            self.series_id,
            {
                "series_id": self.series_id,
                "series_material": {
                    "planned_sessions": 2,
                    "scientific_identity_hash": "science",
                    "hardware_cohort_hash": "hardware",
                },
            },
        )
        self.worktrees: list[Path] = []
        for session, char in ((1, "b"), (2, "c")):
            worktree = self.root / f"worktree-{session}"
            git(self.repo, "worktree", "add", "--detach", str(worktree), self.commit)
            patch_evidence.write_record(
                evidence_record(digest_char=char), root=worktree / "engines" / "llamacpp" / "patches"
            )
            self.worktrees.append(worktree)
            run_id = f"r-{session}"
            job = {"patch": "p", "architecture": "gfx1100"}
            self.store.create_run(
                run_id,
                {
                    "run_id": run_id,
                    "series_id": self.series_id,
                    "batch_id": "b",
                    "session": session,
                    "job": job,
                    "scientific_identity": {
                        "identity_hash": "science",
                        "focal": {"implementation_digest": "a" * 64},
                    },
                    "binding": {"hardware_cohort_hash": "hardware"},
                    "executor_id": "fake",
                },
            )
            attempt = self.store.create_attempt(
                run_id,
                1,
                {
                    "run_id": run_id,
                    "attempt": 1,
                    "execution_id": f"{run_id}:a1",
                    "commit": self.commit,
                    "project_root": str(worktree),
                    "shared_root": str(self.root / "shared"),
                    "job": job,
                },
            )
            (attempt / "executor-result.json").write_text(
                '{"returncode":0}\n', encoding="ascii"
            )

    def tearDown(self):
        for worktree in self.worktrees:
            subprocess.run(
                ("git", "-C", str(self.repo), "worktree", "remove", "--force", str(worktree)),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        self.temp.cleanup()

    def test_commit_harvest_is_exact_and_idempotent(self):
        before = self.commit
        result = harvest_series(
            store=self.store,
            series_id=self.series_id,
            project_root=self.repo,
            work_root=self.root / "work",
            commit=True,
        )
        self.assertTrue(result["verified"])
        self.assertNotEqual(result["commit"], before)
        records = patch_evidence.load_records("p", root=self.repo / "engines" / "llamacpp" / "patches")
        self.assertEqual(len(records), 2)
        self.assertEqual(
            {row["campaign_identity_digest"] for row in records},
            {"b" * 64, "c" * 64},
        )
        expected = patch_evidence.evidence_path(
            "p", root=self.repo / "engines" / "llamacpp" / "patches"
        ).relative_to(self.repo).as_posix()
        changed = git(self.repo, "show", "--pretty=format:", "--name-only", result["commit"])
        self.assertEqual(changed.strip(), expected)
        replay = harvest_series(
            store=self.store,
            series_id=self.series_id,
            project_root=self.repo,
            work_root=self.root / "work",
            commit=True,
        )
        self.assertTrue(replay["already_harvested"])
        self.assertEqual(git(self.repo, "rev-parse", "HEAD"), result["commit"])

    def test_unrelated_staged_change_is_refused(self):
        unrelated = self.repo / "unrelated.txt"
        unrelated.write_text("do not commit\n")
        git(self.repo, "add", "--", "unrelated.txt")
        with self.assertRaises(GitOpsError):
            harvest_series(
                store=self.store,
                series_id=self.series_id,
                project_root=self.repo,
                work_root=self.root / "work",
                commit=True,
            )
        self.assertEqual(git(self.repo, "diff", "--cached", "--name-only"), "unrelated.txt")
        self.assertFalse(
            patch_evidence.evidence_path("p", root=self.repo / "engines" / "llamacpp" / "patches").exists()
        )

    def test_dirty_destination_is_refused_before_merge(self):
        destination = patch_evidence.evidence_path("p", root=self.repo / "engines" / "llamacpp" / "patches")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text('{"operator":"uncommitted"}\n', encoding="utf-8")
        with self.assertRaises(GitOpsError):
            harvest_series(
                store=self.store,
                series_id=self.series_id,
                project_root=self.repo,
                work_root=self.root / "work",
                commit=True,
            )
        self.assertEqual(destination.read_text(encoding="utf-8"), '{"operator":"uncommitted"}\n')

    def test_v2_contract_hash_mismatch_fails_closed(self):
        record = evidence_record(digest_char="d")
        record.update(
            validation_implementation_digest="b" * 64,
            contracts=[{"id": "C", "hash": "c" * 64}],
        )
        record["record_digest"] = patch_evidence._record_digest(record)  # type: ignore[attr-defined]
        run = {
            "job": {"patch": "p", "architecture": "gfx1100"},
            "scientific_identity": {
                "schema": "bigcherry.scientific-identity.v2",
                "focal": {
                    "implementation_digest": "a" * 64,
                    "validation_digest": "b" * 64,
                    "contract_bindings": [{"id": "C", "hash": "e" * 64}],
                },
            },
        }
        with self.assertRaises(HarvestError):
            _verify_record_for_run(record, run)


if __name__ == "__main__":
    unittest.main()
