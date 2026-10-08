"""PA47 remote command is deterministic and preserves queue ownership."""

from __future__ import annotations

import subprocess
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from bigcherry.cli import main as cli_main
from bigcherry.cli.slice_lab import build_lab_command
from bigcherry.core import environment


ROOT = Path(__file__).resolve().parents[3]


class LabCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.host = replace(
            environment.load(ROOT / "config" / "environment.example.toml").host(),
            name="brutus",
            hostname="lab-user@lab-host",
            repo="/lab/workspaces/main",
            cache_root="/lab/cache",
        )

    def command(self, script: str = "tools/lab/flash-next/flash-prefill-env-ab.sh", *args: str):
        return build_lab_command(
            branch="feat/pa47-example",
            host=self.host,
            script=script,
            arguments=args,
            run_id="abcdef012345",
        )

    def test_shell_script_is_syntactically_valid(self) -> None:
        cmd = self.command("tools/lab/flash-next/queue-env-ab.sh", "arm", "run", "8192")
        syntax = subprocess.run(
            ["bash", "-n"], input=cmd.script, text=True,
            capture_output=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_queue_wrapper_avoids_nested_gpu_lock(self) -> None:
        for script in ("queue-meta-mem.sh", "queue-env-ab.sh", "queue.sh"):
            with self.subTest(script=script):
                cmd = self.command("tools/lab/plan-qualification/queue.sh" if script == "queue.sh" else "tools/lab/flash-next/" + script, "arm")
                self.assertIn("QUEUE pa47-", cmd.script)
                self.assertIn("worktree add --detach", cmd.script)
                self.assertIn('worktree remove "$stage"', cmd.script)
                self.assertNotIn("worktree remove --force", cmd.script)

    def test_regular_script_uses_locking_script_row(self) -> None:
        cmd = self.command()
        self.assertIn("SCRIPT pa47-", cmd.script)
        self.assertEqual(cmd.ssh[:4], ("ssh", "-o", "BatchMode=yes", "lab-user@lab-host"))
        self.assertIn("BC_PRIMARY_ROOT", cmd.script)
        self.assertIn("BIGCHERRY_WORK_ROOT", cmd.script)
        self.assertIn("lab primary must stay on main", cmd.script)
        self.assertIn("trap cleanup EXIT", cmd.script)
        self.assertIn("refs/bigcherry-lab/abcdef012345", cmd.script)
        self.assertIn("update-ref -d", cmd.script)
        self.assertNotIn("FETCH_HEAD", cmd.script)

    def test_arguments_are_shell_quoted(self) -> None:
        cmd = self.command("tools/lab/flash-next/queue-env-ab.sh", "x; touch /tmp/injected", "space value")
        self.assertIn("'x; touch /tmp/injected'", cmd.script)
        self.assertIn("'space value'", cmd.script)

    def test_rejects_unsafe_paths_and_branch_names(self) -> None:
        for script in ("/tmp/evil.sh", "../evil.sh", "tools/lab/../evil.sh"):
            with self.subTest(script=script):
                with self.assertRaises(ValueError):
                    build_lab_command(
                        branch="feat/pa47-example", host=self.host,
                        script=script, run_id="abcdef012345",
                    )
        with self.assertRaises(ValueError):
            build_lab_command(
                branch="main", host=self.host,
                script="tools/lab/good.sh", run_id="abcdef012345",
            )

    def test_cli_parses_requested_host_position(self) -> None:
        parser = cli_main.build_parser()
        args = parser.parse_args([
            "slice", "lab", "feat/pa47-example", "--host", "brutus",
            "--dry-run", "--", "tools/lab/flash-next/queue-env-ab.sh", "a",
        ])
        self.assertEqual(args.host, "brutus")
        self.assertTrue(args.dry_run)
        self.assertEqual(args.script[-2:], ["tools/lab/flash-next/queue-env-ab.sh", "a"])

    def test_cli_defaults_to_environment_default_host(self) -> None:
        parser = cli_main.build_parser()
        args = parser.parse_args([
            "slice", "lab", "feat/pa47-example", "--dry-run", "--",
            "tools/lab/flash-next/queue-env-ab.sh", "a",
        ])
        self.assertIsNone(args.host)


    def test_nested_queue_is_dispatched_and_script_exit_is_preserved(self) -> None:
        queue_path = ROOT / "tools" / "lab" / "plan-qualification" / "queue.sh"
        code = queue_path.read_text(encoding="utf-8")
        self.assertIn('if [ "$1" = QUEUE ]; then queue_line "$@"; return $?; fi', code)
        self.assertIn('rc=${PIPESTATUS[0]}', code)
        self.assertIn('BC_QUEUE_STREAM', code)
        self.assertIn('PARSED_MODEL=${BC_MODEL:-}', code)
        self.assertIn('PARSED_HIP=${BC_HIP_PATH:-}', code)
        self.assertNotIn('"$root"/work/builds/', code)
        self.assertIn('"$builds"/', code)
        self.assertIn('ProjectContext.resolve().work_root / "builds"', code)
        syntax = subprocess.run(
            ["bash", "-n", str(queue_path)], capture_output=True, text=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_lab_enables_live_stream_and_uses_primary_host_config(self) -> None:
        cmd = self.command()
        self.assertIn("BC_QUEUE_STREAM=1", cmd.script)
        from bigcherry.cli import slice_lab
        from inspect import getsource
        self.assertIn("repo_root=paths.primary_root()", getsource(slice_lab.cmd_slice_lab))


if __name__ == "__main__":
    unittest.main()
