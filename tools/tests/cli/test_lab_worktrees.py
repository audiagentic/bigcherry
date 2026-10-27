"""WSL07: lab worktree lint and the lab worktree classifier."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry import lab_worktree_lint  # noqa: E402
from bigcherry.cli import slice_lab_worktrees as lab  # noqa: E402

GOOD = """\
MAIN_WT="$BC_WORK_ROOT/worktrees/rad20-ab-main-$(basename $OUT)"
git worktree add -q --detach "$MAIN_WT" origin/main || exit 1
trap 'git worktree remove --force "$MAIN_WT" 2>/dev/null' EXIT
"""


class LintTests(unittest.TestCase):
    def test_per_run_worktree_with_trap_passes(self):
        self.assertEqual(lab_worktree_lint.lint_script(GOOD, "a.sh"), [])

    def test_literal_work_root_target_passes(self):
        text = 'git worktree add "${BC_WORK_ROOT}/worktrees/x" HEAD\ntrap "git worktree remove x" EXIT\n'
        self.assertEqual(lab_worktree_lint.lint_script(text, "a.sh"), [])

    def test_target_outside_work_root_fails(self):
        findings = lab_worktree_lint.lint_script("git worktree add ../wt-main origin/main\n", "a.sh")
        self.assertEqual(len(findings), 1)
        self.assertTrue(findings[0].startswith("a.sh:1:"))

    def test_missing_trap_fails(self):
        text = GOOD.splitlines()[0] + "\n" + GOOD.splitlines()[1] + "\n"
        findings = lab_worktree_lint.lint_script(text, "a.sh")
        self.assertEqual(len(findings), 1)
        self.assertIn("trap", findings[0])

    def test_comments_and_unrelated_scripts_pass(self):
        self.assertEqual(lab_worktree_lint.lint_script("# git worktree add ../x\necho hi\n", "a.sh"), [])

    def test_tree_scan_reports_file_and_line(self):
        with tempfile.TemporaryDirectory() as raw:
            lab_dir = Path(raw) / "tools" / "lab" / "x"
            lab_dir.mkdir(parents=True)
            (lab_dir / "bad.sh").write_text("echo\ngit worktree add /tmp/w main\n", encoding="utf-8")
            (lab_dir / "ok.sh").write_text(GOOD, encoding="utf-8")
            findings = lab_worktree_lint.lint_tree(Path(raw))
        self.assertEqual(len(findings), 1)
        self.assertTrue(findings[0].startswith("tools/lab/x/bad.sh:2:"))

    def test_current_tree_passes(self):
        self.assertEqual(lab_worktree_lint.lint_tree(Path(__file__).resolve().parents[3]), [])


PORCELAIN = """\
worktree /lab/main
HEAD aaaaaaaaaaaa
branch refs/heads/main

worktree /lab/work/worktrees/clean
HEAD bbbbbbbbbbbb
detached

worktree /lab/work/worktrees/dirty
HEAD cccccccccccc
branch refs/heads/feat/dirty-one

worktree /lab/work/worktrees/unique
HEAD dddddddddddd
detached
"""


class ClassifierTests(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(lab.classify(dirty=False, on_origin=True), lab.REMOVABLE)
        self.assertEqual(lab.classify(dirty=True, on_origin=True), lab.DIRTY)
        self.assertEqual(lab.classify(dirty=False, on_origin=False), lab.UNIQUE)
        self.assertEqual(lab.classify(dirty=True, on_origin=False), lab.DIRTY)

    def test_in_use_beats_removable(self):
        self.assertEqual(lab.classify(dirty=False, on_origin=True, in_use=True), lab.IN_USE)
        self.assertEqual(lab.classify(dirty=True, on_origin=False, in_use=True), lab.IN_USE)

    def test_inspection_marks_in_use(self):
        out = (
            "STATE\t/lab/main\t0\t1\t0\nSTATE\t/lab/work/worktrees/clean\t0\t1\t1\n"
            "LIST-BEGIN\n" + PORCELAIN
        )
        states = [state for _tree, state in lab.parse_inspection(out)]
        self.assertEqual(states[:2], [lab.REMOVABLE, lab.IN_USE])

    def test_parse_porcelain(self):
        trees = lab.parse_porcelain(PORCELAIN)
        self.assertEqual([t.primary for t in trees], [True, False, False, False])
        self.assertEqual(trees[2].branch, "feat/dirty-one")
        self.assertIsNone(trees[1].branch)

    def test_inspection_joins_flags(self):
        out = (
            "STATE\t/lab/main\t0\t1\nSTATE\t/lab/work/worktrees/clean\t0\t1\n"
            "STATE\t/lab/work/worktrees/dirty\t1\t1\nSTATE\t/lab/work/worktrees/unique\t0\t0\n"
            "LIST-BEGIN\n" + PORCELAIN
        )
        states = [state for _tree, state in lab.parse_inspection(out)]
        self.assertEqual(states, [lab.REMOVABLE, lab.REMOVABLE, lab.DIRTY, lab.UNIQUE])

    def test_missing_state_is_kept(self):
        states = [state for _tree, state in lab.parse_inspection("LIST-BEGIN\n" + PORCELAIN)]
        self.assertEqual(set(states), {lab.IN_USE})

    def test_failed_fetch_refuses(self):
        with self.assertRaises(ValueError):
            lab.parse_inspection("FETCH-FAILED\nLIST-BEGIN\n" + PORCELAIN)


if __name__ == "__main__":
    unittest.main()
