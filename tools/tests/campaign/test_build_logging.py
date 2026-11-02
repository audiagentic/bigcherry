"""BRBC04: real card-free campaign failures and successful CMake execution."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from bigcherry.build.logs import BuildLogs, stage
from bigcherry.cli.build import cmd_build_new
from bigcherry.cli.main import build_parser
from bigcherry.core import config, retention
from bigcherry.core.context import ProjectContext
from bigcherry.core.process import run_logged


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def campaign(tmp_path):
    project = tmp_path / "project"
    upstream = tmp_path / "upstream"
    project.mkdir()
    upstream.mkdir()
    for repo in (project, upstream):
        git(repo, "init")
        git(repo, "config", "user.email", "test@example.invalid")
        git(repo, "config", "user.name", "Test")
    patches = project / "patches"
    patches.mkdir()
    (project / "fixture.txt").write_text("card-free fixture\n")
    context = ProjectContext(
        engine="llamacpp", project_root=project, config_path=project / "recipes.toml",
        work_root=tmp_path / "work", artifacts_root=project / "artifacts",
        upstream_repo=upstream, overlay_root=project / "src", patches_root=patches,
    )

    def run(mode, run_id=None):
        if mode == "materialize":
            package = patches / "0001_forbidden"
            package.mkdir()
            (package / "patch.toml").write_text(
                'schema = 1\nid = "0001_forbidden"\norder = 1\n'
                'state = "validated"\nkind = "framework"\n')
            (package / "patch.py").write_text(
                "from bigcherry.patch.apply import FilePatch\nPATCHES = []\n")
        cmake = ('cmake_minimum_required(VERSION 3.20)\n'
                 'project(card_free NONE)\n'
                 'message(STATUS "configure-stdout")\n')
        if mode == "configure":
            cmake += 'message(FATAL_ERROR "forced configure failure")\n'
        else:
            cmake += ('file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin")\n'
                      'file(WRITE "${CMAKE_BINARY_DIR}/bin/llama-bench" "card-free")\n'
                      'add_custom_target(llama-bench COMMAND "${CMAKE_COMMAND}" -E echo "compile-stdout")\n')
        (upstream / "CMakeLists.txt").write_text(cmake)
        git(upstream, "add", ".")
        if git(upstream, "status", "--porcelain"):
            git(upstream, "commit", "-m", "fixture")
        revision = git(upstream, "rev-parse", "HEAD")
        git(project, "add", ".")
        if git(project, "status", "--porcelain"):
            git(project, "commit", "-m", "fixture")
        cfg = config.Config(
            engine="llamacpp", pinned=revision,
            patch_sets={"framework": config.PatchSet(name="framework", required_state="validated", patches=("0001_forbidden",))} if mode == "materialize" else {},
            sources={"fixture": config.Source(
                name="fixture", ref=revision, overlay=False,
                patch_sets=("framework",) if mode == "materialize" else ())},
            builds={"stock": config.Build(name="stock", options=(), variant_set=None, needs=frozenset())},
            platforms={"local": config.Platform(name="local", targets=(), options=())},
            experiments={}, campaigns={}, path=context.config_path,
        )
        argv = ["build", "--lane", "fixture:stock:local"]
        if run_id:
            argv += ["--run-id", run_id]
        args = build_parser().parse_args(argv)
        with patch("bigcherry.core.context.resolve_work_root", return_value=context.work_root), \
             patch("bigcherry.core.context.ProjectContext.resolve", return_value=context), \
             patch("bigcherry.core.config.engine_defining", return_value="llamacpp"), \
             patch("bigcherry.core.config.load", return_value=cfg), \
             patch("bigcherry.release.pin_status.campaign_preflight", return_value=([], False)):
            code = cmd_build_new(args)
        roots = sorted((context.work_root / "build-logs").iterdir())
        root = context.work_root / "build-logs" / run_id if run_id else roots[-1]
        return code, root, json.loads((root / "summary.json").read_text())
    return run, context


def lane(summary):
    return summary["lanes"]["fixture:stock:local"]


def test_materialize_forbidden_import_keeps_traceback(campaign, capsys):
    run, context = campaign
    code, root, summary = run("materialize", "bad-patch")
    assert code == 1
    record = lane(summary)["stages"]["materialize"]
    text = Path(record["log_path"]).read_text()
    assert "Traceback (most recent call last)" in text
    assert "PipelineError" in text and "bigcherry.patch.apply" in text
    assert record["status"] == "failed"
    assert "bigcherry.patch.apply" in record["first_error"]
    assert summary["status"] == "failed"
    assert (root / "build.log").is_file()
    assert (context.work_root / "build-advisories" / "bad-patch" / "advisories.json").is_file()
    console = capsys.readouterr().err
    assert record["log_path"] in console
    assert str(root / "build.log") in console
    assert console in (root / "build.log").read_text()


@pytest.mark.skipif(not shutil.which("cmake") or not shutil.which("ninja"), reason="requires CMake/Ninja")
def test_configure_failure_records_diagnostic(campaign, capsys):
    code, root, summary = campaign[0]("configure", "bad-configure")
    assert code == 1
    record = lane(summary)["stages"]["configure"]
    text = Path(record["log_path"]).read_text()
    assert "configure-stdout" in text and "forced configure failure" in text
    assert "CalledProcessError" in text and "Traceback" in text
    assert record["status"] == "failed"
    assert "Error" in record["first_error"]
    assert record["log_path"] in capsys.readouterr().err
    assert lane(summary)["build_plan_id"]
    assert lane(summary)["build_dir"]


@pytest.mark.skipif(not shutil.which("cmake") or not shutil.which("ninja"), reason="requires CMake/Ninja")
def test_success_and_untagged_runs_keep_independent_logs(campaign):
    first_code, first, first_summary = campaign[0]("success")
    second_code, second, summary = campaign[0]("success")
    assert first_code == second_code == 0
    assert first != second
    assert first_summary["run_id"] != summary["run_id"]
    assert "unspecified" not in summary["run_id"]
    record = lane(summary)
    assert summary["status"] == record["status"] == "succeeded"
    assert record["duration_seconds"] > 0
    assert all(s["duration_seconds"] > 0 for s in record["stages"].values())
    build = Path(record["build_dir"])
    for root in (first, second):
        copy = build / "build-logs" / root.name
        assert (copy / "build.log").read_bytes() == (root / "build.log").read_bytes()
        assert (copy / "summary.json").read_bytes() == (root / "summary.json").read_bytes()
        assert list(copy.glob("*/materialize.log"))
        assert list(copy.glob("*/build.log"))
    assert "compile-stdout" in (first / "build.log").read_text()
    assert "configure-stdout" in Path(lane(first_summary)["stages"]["configure"]["log_path"]).read_text()


def test_capture_full_streams_without_console_noise(tmp_path, capfd):
    logs = BuildLogs(tmp_path, None)
    with logs.invocation(), logs.lane_scope("one"):
        with stage("generate"):
            subprocess.run([sys.executable, "-c",
                            "import sys; print('visible-out'); print('visible-err', file=sys.stderr)"], check=True)
            run_logged([sys.executable, "-c",
                        "import sys; print('hidden-out'); print('hidden-err', file=sys.stderr)"],
                       check=True, capture_output=True, text=True)
        logs.summary["status"] = "succeeded"
    out, err = capfd.readouterr()
    assert "visible-out" in out and "visible-err" in err
    assert "hidden" not in out + err
    for path in (logs.root / "build.log", Path(logs.summary["lanes"]["one"]["stages"]["generate"]["log_path"])):
        text = path.read_text()
        for line in ("visible-out", "visible-err", "hidden-out", "hidden-err"):
            assert text.count(line) == 1


def test_explicit_run_ids_are_exclusive_and_safe(tmp_path):
    BuildLogs(tmp_path, "once")
    with pytest.raises(FileExistsError):
        BuildLogs(tmp_path, "once")
    for value in ("../outside", "unspecified", "."):
        with pytest.raises(ValueError):
            BuildLogs(tmp_path, value)


def test_cleaner_reports_and_protects_build_logs(tmp_path):
    logs = BuildLogs(tmp_path, "retained")
    with logs.invocation():
        logs.summary["status"] = "succeeded"
    inventory = retention.inspect_workspace(tmp_path)
    assert logs.root.parent not in inventory.unknown
    decision, = retention.plan_cleanup(inventory).decisions
    assert decision.path == logs.root
    assert decision.action == "protected"
    assert "retained indefinitely" in decision.reasons[0]


def test_captured_failure_keeps_full_output_and_traceback(tmp_path, capsys):
    logs = BuildLogs(tmp_path, "smoke-failure")
    with pytest.raises(subprocess.CalledProcessError):
        with logs.invocation(), logs.lane_scope("one"), stage("runtime-smoke"):
            run_logged(
                [sys.executable, "-c",
                 "import sys; print('result'); sys.stderr.write('begin\\n' + 'x'*5000 + '\\nend\\n'); sys.exit(1)"],
                capture_output=True, text=True, check=True,
            )
    record = logs.summary["lanes"]["one"]["stages"]["runtime-smoke"]
    text = Path(record["log_path"]).read_text()
    assert "begin\n" + "x"*5000 + "\nend" in text
    assert "CalledProcessError" in text and "Traceback" in text
    assert record["status"] == "failed"
    assert logs.summary["status"] == "failed"
    assert capsys.readouterr() == ("", "")
