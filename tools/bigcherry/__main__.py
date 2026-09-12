"""The ``bigcherry`` command line.

One command per stage of taking a new llama.cpp release into production:

    pull -> audit -> apply -> generate -> build

Stages are idempotent, and each refuses to run on a tree that has not passed
the stage before it. That ordering is the whole point: patches are only
meaningful against a tree whose shape has been verified, and a build is only
meaningful against a manifest generated from that same tree.
"""

from __future__ import annotations

import importlib
from typing import Any, cast

from .cli.diagnostics import cmd_check, cmd_doctor, cmd_status  # noqa: F401
from .release import records as _release_records

_source_cli = importlib.import_module("bigcherry.cli.source")
cmd_audit = _source_cli.cmd_audit
cmd_pull = _source_cli.cmd_pull

_patch_cli = importlib.import_module("bigcherry.cli.patch")
cmd_apply = _patch_cli.cmd_apply
cmd_patch_explain = _patch_cli.cmd_patch_explain
cmd_patch_graph = _patch_cli.cmd_patch_graph
cmd_patch_lint = _patch_cli.cmd_patch_lint
cmd_patch_status = _patch_cli.cmd_patch_status
cmd_patch_validate = _patch_cli.cmd_patch_validate
cmd_patch_verify_evidence = _patch_cli.cmd_patch_verify_evidence
cmd_patches = _patch_cli.cmd_patches
_tuning_cli = importlib.import_module("bigcherry.cli.tuning")
cmd_generate = _tuning_cli.cmd_generate
cmd_replay_inspect = _tuning_cli.cmd_replay_inspect
cmd_inventory = _tuning_cli.cmd_inventory

_build_cli = importlib.import_module("bigcherry.cli.build")
cmd_build_new = _build_cli.cmd_build_new

_experiment_cli = importlib.import_module("bigcherry.cli.experiment")
cmd_experiment_validate = _experiment_cli.cmd_experiment_validate
cmd_experiment_list = _experiment_cli.cmd_experiment_list
cmd_experiment_plan = _experiment_cli.cmd_experiment_plan
cmd_experiment_run = _experiment_cli.cmd_experiment_run
cmd_experiment_report = _experiment_cli.cmd_experiment_report

# TR09: build_parser/_legacy_main/cmd_repin/cmd_pin_status/_configure_output now
# live in bigcherry.cli.main -- the canonical parser-assembly/entrypoint home.
# Re-exported here (same pattern as the cmd_* handlers above) because existing
# tests resolve them as bigcherry.__main__.build_parser et al.
_main_cli = importlib.import_module("bigcherry.cli.main")
build_parser = _main_cli.build_parser
_legacy_main = _main_cli._legacy_main
cmd_repin = _main_cli.cmd_repin
cmd_pin_status = _main_cli.cmd_pin_status
_configure_output = _main_cli._configure_output

_copy_overlay = _patch_cli._copy_overlay
_restore_overlay = _patch_cli._restore_overlay
_apply_exact_selection = _patch_cli._apply_exact_selection
_record_for = _release_records.record_for_checkout

def main(argv: list[str] | None = None) -> int:
    """Delegate the package entrypoint to the canonical CLI bootstrap."""
    from importlib import import_module

    cli_main = cast(Any, import_module("bigcherry.cli.main").main)
    return int(cli_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
