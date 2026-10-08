from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.cli import slice as slice_cli  # noqa: E402


class FakeGit:
    def __init__(self):
        self.calls: list[tuple[tuple[str, ...], bool]] = []

    def __call__(self, args, *, check=True):
        argv = tuple(args)
        self.calls.append((argv, check))
        if argv[:1] == ("for-each-ref",):
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=(
                    "main\tmainsha\t100\tOwner\n"
                    "release-please--branches--main--components--bigcherry\trpsha\t100\tBot\n"
                    "merged-one\tmergedsha\t100\tAlice\n"
                    "old-open\toldsha\t100\tBob\n"
                    "fresh-open\tfreshsha\t1999000\tCarol\n"
                ),
                stderr="",
            )
        if argv[:2] == ("merge-base", "--is-ancestor"):
            return subprocess.CompletedProcess(
                argv, 0 if argv[2] == "mergedsha" else 1, stdout="", stderr=""
            )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

 
class SlicePruneTests(unittest.TestCase):
    def test_default_lists_merged_without_deleting(self):
        git = FakeGit()
        result = slice_cli.prune_remote_branches(
            now=2_000_000,
            stale_days=14,
            runner=git,
        )

        self.assertEqual(result["merged"], ["merged-one"])
        self.assertEqual(result["deleted"], [])
        self.assertEqual(result["stale_unmerged"], ["old-open"])
        self.assertFalse(any(call[:1] == ("push",) for call, _ in git.calls))

    def test_apply_deletes_only_fully_merged(self):
        git = FakeGit()
        result = slice_cli.prune_remote_branches(
            now=2_000_000,
            stale_days=14,
            apply=True,
            runner=git,
        )
        self.assertEqual(result["deleted"], ["merged-one"])
        deletes = [
            call for call, _ in git.calls
            if call[:3] == ("push", "origin", "--delete")
        ]
        self.assertEqual(deletes, [("push", "origin", "--delete", "merged-one")])
        self.assertNotIn(
            ("push", "origin", "--delete", "old-open"),
            [call for call, _ in git.calls],
        )

    def test_release_and_base_branches_are_never_considered_for_delete(self):
        git = FakeGit()
        slice_cli.prune_remote_branches(now=2_000_000, apply=True, runner=git)
        checked_shas = [
            call[2]
            for call, _ in git.calls
            if call[:2] == ("merge-base", "--is-ancestor")
        ]
        self.assertNotIn("mainsha", checked_shas)
        self.assertNotIn("rpsha", checked_shas)

    def test_negative_stale_days_fails_before_branch_deletion(self):
        with self.assertRaises(ValueError):
            slice_cli.prune_remote_branches(stale_days=-1, runner=FakeGit())


if __name__ == "__main__":
    unittest.main()
