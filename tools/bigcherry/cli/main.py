"""Canonical CLI bootstrap, argument parser, and entrypoint.

Owns argparse assembly (``build_parser``) and command dispatch
(``_legacy_main``) for every ``bigcherry`` subcommand. Each subcommand's
handler implementation lives in the sibling ``cli/<domain>.py`` module
(``source.py``, ``patch.py``, ``tuning.py``, ``build.py``, ``experiment.py``,
``diagnostics.py``); this module only wires them into the parser.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .. import __version__, patch_catalog, patchset, recipes
from ..source import sources
from ..release import pin as _release_pin
from .build import cmd_build_new
from .diagnostics import cmd_check, cmd_doctor, cmd_status
from .experiment import (
    cmd_experiment_list,
    cmd_experiment_plan,
    cmd_experiment_report,
    cmd_experiment_run,
    cmd_experiment_validate,
)
from .patch import (
    cmd_apply,
    cmd_patch_explain,
    cmd_patch_graph,
    cmd_patch_lint,
    cmd_patch_status,
    cmd_patch_validate,
    cmd_patch_verify_evidence,
    cmd_patches,
)
from .source import cmd_audit, cmd_pull
from .tuning import cmd_generate, cmd_inventory, cmd_replay_inspect

_CLI_DESCRIPTION = """The ``bigcherry`` command line.

One command per stage of taking a new llama.cpp release into production:

    pull -> audit -> apply -> generate -> build

Stages are idempotent, and each refuses to run on a tree that has not passed
the stage before it. That ordering is the whole point: patches are only
meaningful against a tree whose shape has been verified, and a build is only
meaningful against a manifest generated from that same tree.
"""


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    """The patch-selection flags, shared by every command that selects."""
    parser.add_argument(
        "--recipe",
        default=None,
        choices=recipes.names() or None,
        help="named build definition from config/recipes.toml (default: all patches)",
    )
    parser.add_argument(
        "--groups",
        default=None,
        help="comma-separated patch groups, overriding the recipe's "
        "(e.g. 'core'). Empty string selects none.",
    )
    parser.add_argument(
        "--states",
        default=None,
        help=f"comma-separated patch states, overriding the recipe's "
        f"({', '.join(patchset.STATES)}).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bigcherry", description=_CLI_DESCRIPTION)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--llama-root",
        default=None,
        help="llama.cpp checkout (default: vendor/llama.cpp)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pull = sub.add_parser("pull", help="clone or update the llama.cpp checkout")
    pull.add_argument(
        "--ref",
        default=None,
        help="tag, branch or sha to check out (e.g. b1234), or 'latest' for "
        "the newest upstream release. Overrides --recipe.",
    )
    pull.add_argument(
        "--recipe",
        default=None,
        choices=recipes.names() or None,
        help="take the ref from this recipe in config/recipes.toml",
    )
    pull.add_argument(
        "--full",
        action="store_true",
        help="full clone instead of depth-1 (needed to check out "
        "arbitrary older revisions)",
    )
    pull.set_defaults(func=cmd_pull)

    audit = sub.add_parser("audit", help="verify upstream invariants")
    audit.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="treat warnings as failures (default)",
    )
    audit.add_argument("--no-strict", dest="strict", action="store_false")
    audit.add_argument("-v", "--verbose", action="store_true")
    audit.set_defaults(func=cmd_audit)

    check_cmd = sub.add_parser("check", help="run deterministic local CI gates")
    check_tier = check_cmd.add_mutually_exclusive_group()
    check_tier.add_argument("--quick", action="store_const", const="quick", dest="tier")
    check_tier.add_argument(
        "--default", action="store_const", const="default", dest="tier"
    )
    check_tier.add_argument("--full", action="store_const", const="full", dest="tier")
    check_cmd.add_argument("--fail-fast", action="store_true")
    check_cmd.add_argument("--json", metavar="PATH", default=None)
    check_cmd.set_defaults(func=cmd_check, tier="default")

    apply_cmd = sub.add_parser("apply", help="apply the overlay and patches")
    apply_cmd.add_argument(
        "--dry-run", action="store_true", help="report what would change, write nothing"
    )
    apply_cmd.add_argument(
        "--force", action="store_true", help="patch even without a passing audit"
    )
    _add_selection_args(apply_cmd)
    apply_cmd.set_defaults(func=cmd_apply)

    patches_cmd = sub.add_parser(
        "patches",
        help="list patches with group, state and upstream status",
    )
    patches_cmd.add_argument("--llama-root", default=None)
    _add_selection_args(patches_cmd)
    patches_cmd.add_argument(
        "--kind",
        default=None,
        choices=patch_catalog.KINDS,
        help="filter to patches/catalog.toml's kind (framework|upstream-backport|"
        "enhancement) -- the metadata substitute for a physical folder split "
        "(RE41: browsability metadata instead of a directory split; "
        "patch-system PA02 keeps it metadata-first)",
    )
    patches_cmd.add_argument(
        "--backend",
        default=None,
        choices=patch_catalog.BACKENDS,
        help="filter to patches/catalog.toml's backend (hip|vulkan|agnostic)",
    )
    patches_cmd.add_argument(
        "--origin",
        default=None,
        choices=patch_catalog.ORIGINS,
        help="filter to patches/catalog.toml's origin (local|upstream-commit|"
        "upstream-pr|external-fork)",
    )
    patches_cmd.set_defaults(func=cmd_patches)

    patch_status_cmd = sub.add_parser(
        "patch-status",
        help="EC19: computed plan/patch/contract lifecycle status, not "
        "hand-maintained prose -- source-pinned/materialized/build-state/"
        "contracted per RD/EX/HI plan item",
    )
    patch_status_cmd.add_argument(
        "--item",
        default=None,
        help="show only this plan-item (e.g. RD21) instead of every item "
        "with any signal",
    )
    patch_status_cmd.set_defaults(func=cmd_patch_status)

    patch_explain_cmd = sub.add_parser(
        "patch-explain",
        help="RE43: everything known about one patch -- source, plan, "
        "requires/conflicts, which recipes/experiments select it, "
        "state, content hash, files touched",
    )
    patch_explain_cmd.add_argument(
        "patch_id", help="e.g. 1217_rd44_graph_opt_default_rdna35"
    )
    patch_explain_cmd.set_defaults(func=cmd_patch_explain)

    patch_graph_cmd = sub.add_parser(
        "patch-graph",
        help="RE43: textual REQUIRES/CONFLICTS dependency topology",
    )
    patch_graph_cmd.add_argument(
        "--roots",
        action="append",
        default=None,
        help="restrict to this patch's real dependency closure (repeatable); "
        "omit to show every patch with any requires/conflicts edge",
    )
    patch_graph_cmd.set_defaults(func=cmd_patch_graph)

    patch_verify_evidence_cmd = sub.add_parser(
        "patch-verify-evidence",
        help="HI83: verify that every STATE='validated' patch has current "
        "matching validation evidence (observational only -- not wired "
        "into apply/build)",
    )
    patch_verify_evidence_cmd.add_argument(
        "patch_id",
        nargs="?",
        default=None,
        help="check only this patch instead of every patch in the catalog",
    )
    patch_verify_evidence_cmd.add_argument("--json", action="store_true")
    patch_verify_evidence_cmd.add_argument(
        "--no-legacy-grandfather",
        action="store_true",
        help="require real HI83 evidence; do not accept the one-time legacy baseline",
    )
    patch_verify_evidence_cmd.set_defaults(func=cmd_patch_verify_evidence)

    patch_lint_cmd = sub.add_parser(
        "patch-lint", help="lint patch metadata without mutation"
    )
    patch_lint_cmd.add_argument("--json", action="store_true")
    patch_lint_cmd.set_defaults(func=cmd_patch_lint)

    patch_validate_cmd = sub.add_parser(
        "patch-validate", help="verify existing patch evidence"
    )
    patch_validate_cmd.add_argument("patch_id", nargs="?", default=None)
    patch_validate_cmd.add_argument("--json", action="store_true")
    patch_validate_cmd.add_argument("--no-legacy-grandfather", action="store_true")
    patch_validate_cmd.set_defaults(func=cmd_patch_validate)

    sources.register(sub)

    repin = sub.add_parser(
        "repin", help="move config/recipes.toml's pin to the newest upstream release"
    )
    repin.add_argument(
        "--ref",
        default=None,
        help="pin to this ref instead of querying for the newest release",
    )
    repin.set_defaults(func=cmd_repin)

    pin_status_cmd = sub.add_parser(
        "pin-status",
        help=(
            "name the pin state of this tree (and configured remotes): "
            "consistent / mid-rebase / drift; --strict = pipeline preflight, "
            "--complete = bump-completion gate"
        ),
    )
    mode = pin_status_cmd.add_mutually_exclusive_group()
    mode.add_argument(
        "--strict",
        action="store_true",
        help="pipeline preflight: fail on drift/unavailable/"
        "unresolvable-pin/uncommitted-transition of the "
        "gated tree",
    )
    mode.add_argument(
        "--complete",
        action="store_true",
        help="bump-completion gate: every required tree "
        "reachable, converged and consistent",
    )
    pin_status_cmd.add_argument(
        "--json", action="store_true", help="machine-readable report"
    )
    remotes = pin_status_cmd.add_mutually_exclusive_group()
    remotes.add_argument(
        "--remote", default=None, help="probe only this configured tree"
    )
    remotes.add_argument(
        "--all-remotes", action="store_true", help="probe every configured tree"
    )
    pin_status_cmd.set_defaults(func=cmd_pin_status)

    replay_inspect_cmd = sub.add_parser(
        "replay-inspect",
        help=(
            "HI15/HI16: inspect the registry and a replay cache through the "
            "real C++ loader; --manifest adds catalog<->registry agreement"
        ),
    )
    replay_inspect_cmd.add_argument(
        "cache",
        nargs="?",
        default=None,
        help="replay cache to inspect (omit for registry only)",
    )
    replay_inspect_cmd.add_argument(
        "--manifest",
        default=None,
        help="build manifest JSON; check catalog<->registry agreement",
    )
    replay_inspect_cmd.add_argument(
        "--tool",
        default=None,
        help="path to the hip-autotune-inspect binary "
        "(default: $BIGCHERRY_INSPECT_TOOL, then build dirs)",
    )
    replay_inspect_cmd.add_argument(
        "--tool-interpreter",
        nargs="*",
        default=None,
        metavar="ARG",
        help="command prefix for invoking the tool (e.g. a python script "
        "stand-in); the tool path is appended to it",
    )
    replay_inspect_cmd.add_argument(
        "--json", action="store_true", help="machine-readable report"
    )
    replay_inspect_cmd.set_defaults(func=cmd_replay_inspect)

    # RE21/RE23: `build` is the multi-lane planner/runner (RE18) and nothing
    # else -- a canonical-v2 interface only, never a translation layer for
    # the retired mutable-checkout recipe model (--recipe/--groups/--states/
    # --variant-set/--force/--target selected compat recipes and mutated
    # patch-selection axes on one shared checkout; canonical v2 sources
    # instead name exact patch sets directly, each build isolated by
    # content-addressed identity). Any such flag is simply not defined on
    # this parser, so argparse itself rejects it with exit 2 ("unrecognized
    # arguments") -- fail closed, never silent routing anywhere.
    # --inventory/--winners DO translate (they distribute to whichever
    # standard lanes declare that need, per Build.needs), since they are
    # canonical v2 concepts (CampaignLaneExecutionSpec.inputs), not legacy
    # ones. The `legacy-build` compatibility/diagnostic command that used
    # to exist alongside this was deleted in RE23 once the cutover's
    # objective compatibility gates (RE15's real-hardware acceptance chain,
    # both platform suites, RE24's adversarial matrix) were all satisfied.
    new_build_cmd = sub.add_parser(
        "build",
        help="build via the multi-lane campaign engine (canonical v2 identities only)",
    )
    new_build_cmd.add_argument("--llama-root", default=None)
    new_build_cmd.add_argument("--source", default="bigcherry")
    new_build_cmd.add_argument(
        "--profile",
        default=None,
        help="named campaign profile from config/recipes.toml's [campaign.<name>] "
        "(e.g. 'standard')",
    )
    new_build_cmd.add_argument(
        "--lane",
        action="append",
        default=None,
        metavar="SOURCE:BUILD:PLATFORM",
        help="explicit lane selector (repeatable); alternative to --profile, "
        "not combinable with it",
    )
    new_build_cmd.add_argument(
        "--all",
        action="store_true",
        help="build the canonical standard profile -- shorthand for --profile standard",
    )
    new_build_cmd.add_argument(
        "--arch",
        default=None,
        help="comma-separated architectures, overriding each lane's "
        "platform.targets (must be a non-empty subset)",
    )
    new_build_cmd.add_argument(
        "--inventory",
        default=None,
        help="signature inventory JSON, distributed to any planned lane "
        'whose build declares needs = ["inventory", ...]',
    )
    new_build_cmd.add_argument(
        "--winners",
        default=None,
        help="promoted-winners JSONL, distributed to any planned lane whose "
        'build declares needs including "promoted-winners"',
    )
    new_build_cmd.add_argument(
        "--model",
        default=None,
        help="gguf model path -- if given, every planned lane runs a real "
        "runtime-smoke validation against it",
    )
    new_build_cmd.add_argument(
        "--hip-visible-devices",
        default="0",
        help="only meaningful together with --model",
    )
    new_build_cmd.add_argument(
        "--binary-relative-path",
        default="bin/llama-bench",
        help="which binary each planned lane builds/publishes as its "
        "primary artifact, e.g. 'bin/llama-server' for a real "
        "production/deployment build -- matches campaign-build's own "
        "flag of the same name (re14_real_run.py); every lane in the "
        "request gets the same value, there is no per-lane override",
    )
    new_build_cmd.add_argument("--c-compiler", default=None)
    new_build_cmd.add_argument("--cxx-compiler", default=None)
    new_build_cmd.add_argument("--run-id", default=None)
    new_build_cmd.add_argument(
        "--experiment",
        default=None,
        help="name of a [experiment.<name>] entry in config/recipes.toml (an exact "
        "extra patch list) -- for benching one experimental patch in "
        "isolation against the source's normal patch-set, e.g. "
        "'--source bigcherry-native --experiment rd19-only'",
    )
    new_build_cmd.set_defaults(func=cmd_build_new)

    from .. import autotune_schema as _schema

    generate = sub.add_parser(
        "generate", help="generate the candidate catalog and its artifacts"
    )
    generate.add_argument(
        "--variant-set", default="inventory", choices=_schema.VARIANT_SETS
    )
    generate.add_argument(
        "--arch",
        default="all",
        help="comma-separated architectures or group names "
        f"({', '.join(sorted(_schema.ARCHITECTURE_GROUPS))})",
    )
    generate.add_argument(
        "--inventory",
        default=None,
        help="inventory JSON from a record-mode run (required for workload-max)",
    )
    generate.add_argument(
        "--winners",
        default=None,
        help="measurements JSONL from a tuning run (required for replay-slim)",
    )
    generate.add_argument(
        "--generated-root",
        default=None,
        help="build-local directory for generated compile inputs",
    )
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument(
        "--force", action="store_true", help="generate even against an unpatched tree"
    )
    generate.set_defaults(func=cmd_generate)

    status = sub.add_parser("status", help="show checkout and release status")
    status.set_defaults(func=cmd_status)

    experiment_cmd = sub.add_parser(
        "experiment-contract",
        help="Experiment Contract validate/list/plan/run/report (EC01-EC11) -- "
        "not to be confused with `experiment`, HI47's managed bundle runner",
    )
    experiment_sub = experiment_cmd.add_subparsers(
        dest="experiment_command", required=True
    )

    def _add_contracts_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--contracts",
            default=None,
            help="path to the contract registry TOML (default: config/experiment-contracts.toml)",
        )

    def _add_lane_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--config",
            default=None,
            help="path to recipes.toml (default: config/recipes.toml)",
        )
        p.add_argument(
            "--source",
            required=True,
            help="config.Source name to expand the contract against",
        )
        p.add_argument("--build", required=True, help="config.Build name")
        p.add_argument("--platform", required=True, help="config.Platform name")

    experiment_validate = experiment_sub.add_parser(
        "validate",
        help="schema-check a contract (or every contract) without running anything",
    )
    _add_contracts_arg(experiment_validate)
    experiment_validate.add_argument("contract_id", nargs="?", default=None)
    experiment_validate.set_defaults(func=cmd_experiment_validate)

    experiment_list = experiment_sub.add_parser(
        "list", help="list every registered contract"
    )
    _add_contracts_arg(experiment_list)
    experiment_list.set_defaults(func=cmd_experiment_list)

    experiment_plan = experiment_sub.add_parser(
        "plan", help="show the campaign lanes a contract would expand into (dry-run)"
    )
    _add_contracts_arg(experiment_plan)
    _add_lane_args(experiment_plan)
    experiment_plan.add_argument("contract_id")
    experiment_plan.set_defaults(func=cmd_experiment_plan)

    experiment_run = experiment_sub.add_parser(
        "run", help="execute a contract's full lane set through run_campaign()"
    )
    _add_contracts_arg(experiment_run)
    _add_lane_args(experiment_run)
    experiment_run.add_argument("contract_id")
    experiment_run.set_defaults(func=cmd_experiment_run)

    experiment_report = experiment_sub.add_parser(
        "report", help="render a contract's report from a stored evidence JSON file"
    )
    _add_contracts_arg(experiment_report)
    experiment_report.add_argument("contract_id")
    experiment_report.add_argument(
        "--evidence-file",
        required=True,
        help="JSON with correctness_gate/aggregated_effects/generalisation_result "
        "(and optionally promotion_gate) keys",
    )
    experiment_report.set_defaults(func=cmd_experiment_report)

    doctor_cmd = sub.add_parser(
        "doctor", help="audit migration assumptions and identity inputs"
    )
    doctor_cmd.add_argument("--json", action="store_true")
    doctor_cmd.set_defaults(func=cmd_doctor)

    tune_journal_cmd = sub.add_parser(
        "tune-journal", help="crash-safe tuning journal status/compaction (HI48)"
    )
    tune_journal_cmd.set_defaults(
        func=lambda args: _tune_journal_main(args.tune_journal_args)
    )
    tune_journal_cmd.add_argument("tune_journal_args", nargs=argparse.REMAINDER)

    tune_promote_cmd = sub.add_parser(
        "tune-promote",
        help="apply experiment-wide BH promotion to fresh-confirmation evidence (HI34)",
    )
    tune_promote_cmd.add_argument("measurements")
    tune_promote_cmd.add_argument("--output", required=True)
    tune_promote_cmd.add_argument(
        "--dispatch-db",
        required=True,
        help="dispatch database (schema 6+) this measurements file was ingested "
        "into, e.g. via `bigcherry inventory tuning` -- required so a "
        "non-native winner's CPU-reference correctness evidence (HI67) "
        "can be checked as a hard AND with the statistical promotion "
        "criteria",
    )
    tune_promote_cmd.add_argument("--q", type=float, default=0.05)
    tune_promote_cmd.add_argument("--threshold-pct", type=float, default=1.0)
    tune_promote_cmd.add_argument("--resamples", type=int, default=10_000)
    tune_promote_cmd.set_defaults(
        func=lambda args: _tune_promote_main(
            [
                args.measurements,
                "--output",
                args.output,
                "--dispatch-db",
                args.dispatch_db,
                "--q",
                str(args.q),
                "--threshold-pct",
                str(args.threshold_pct),
                "--resamples",
                str(args.resamples),
            ]
        )
    )

    tune_null_fdr_cmd = sub.add_parser(
        "tune-null-fdr",
        help="deterministic global-null BH simulation, for auditing the promotion gate",
    )
    tune_null_fdr_cmd.add_argument("--output", required=True)
    tune_null_fdr_cmd.add_argument("--experiments", type=int, default=5000)
    tune_null_fdr_cmd.add_argument("--hypotheses", type=int, required=True)
    tune_null_fdr_cmd.add_argument("--q", type=float, default=0.05)
    tune_null_fdr_cmd.add_argument("--seed", type=int, required=True)
    tune_null_fdr_cmd.set_defaults(
        func=lambda args: _tune_null_fdr_main(
            [
                "--output",
                args.output,
                "--experiments",
                str(args.experiments),
                "--hypotheses",
                str(args.hypotheses),
                "--q",
                str(args.q),
                "--seed",
                str(args.seed),
            ]
        )
    )

    experiment_cmd = sub.add_parser(
        "experiment", help="managed experiment bundle: run or validate (HI47)"
    )
    experiment_cmd.set_defaults(
        func=lambda args: _experiment_main(args.experiment_args)
    )
    experiment_cmd.add_argument("experiment_args", nargs=argparse.REMAINDER)

    # RE14: the new, content-addressed, isolated-worktree campaign path,
    # registered as a real subcommand rather than remaining only a
    # standalone script -- not yet the default execution path (that flip
    # is RE14 step 7, gated on further negative-case coverage). Legacy
    # `build` above is completely untouched by this and remains the normal
    # path until that flip happens.
    campaign_build_cmd = sub.add_parser(
        "campaign-build",
        help="RE14: build via the new isolated/content-addressed campaign "
        "path (not yet the default -- see `build` for the normal path)",
    )
    campaign_build_cmd.set_defaults(
        func=lambda args: _campaign_build_main(args.campaign_build_args)
    )
    campaign_build_cmd.add_argument("campaign_build_args", nargs=argparse.REMAINDER)

    from .. import compare_tunes as _compare_tunes

    compare = sub.add_parser(
        "compare-tunes", help="compare two current tuning runs by signature"
    )
    compare.add_argument("before")
    compare.add_argument("after")
    compare.add_argument(
        "--record", default=None, help="record JSONL for call-weighted impact"
    )
    compare.add_argument("--output", default=None, help="JSON report path")

    def _run_compare(args):
        try:
            result = _compare_tunes.compare(
                Path(args.before),
                Path(args.after),
                record=Path(args.record) if args.record else None,
            )
        except (OSError, ValueError, _compare_tunes.CompareError) as exc:
            print(f"invalid: {exc}", file=sys.stderr)
            return 1
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0

    compare.set_defaults(func=_run_compare)

    ab = sub.add_parser(
        "ab-benchmark",
        help="paired, interleaved native-versus-replay end-to-end benchmark",
    )
    ab.add_argument("--cache", required=True)
    ab.add_argument("--output", required=True)
    ab.add_argument("--pairs", type=int, default=3)
    ab.add_argument("--schedule-seed", type=int, default=0)
    ab.add_argument("--structured", action="store_true")
    ab.add_argument("--practical-threshold-pct", type=float, default=1.0)
    ab.add_argument("--decision-grade", action="store_true")
    ab.add_argument("--settle-seconds", type=float, default=20.0)
    ab.add_argument("--cwd", default=None)
    ab.add_argument("--metric", action="append", default=[])
    ab.add_argument("--lower-is-better", action="append", default=[])
    ab.add_argument("--stock-binary", default=None)
    ab.add_argument("--stock-cmake-cache", default=None)
    ab.add_argument("--patched-cmake-cache", default=None)
    ab.add_argument("command", nargs=argparse.REMAINDER)
    ab.set_defaults(
        func=lambda args: _ab_benchmark_main(
            [
                "--cache",
                args.cache,
                "--output",
                args.output,
                "--pairs",
                str(args.pairs),
                "--schedule-seed",
                str(args.schedule_seed),
                "--practical-threshold-pct",
                str(args.practical_threshold_pct),
                *(["--structured"] if args.structured else []),
                *(["--decision-grade"] if args.decision_grade else []),
                "--settle-seconds",
                str(args.settle_seconds),
                *(["--cwd", args.cwd] if args.cwd else []),
                *(item for spec in args.metric for item in ["--metric", spec]),
                *(
                    item
                    for name in args.lower_is_better
                    for item in ["--lower-is-better", name]
                ),
                *(["--stock-binary", args.stock_binary] if args.stock_binary else []),
                *(
                    ["--stock-cmake-cache", args.stock_cmake_cache]
                    if args.stock_cmake_cache
                    else []
                ),
                *(
                    ["--patched-cmake-cache", args.patched_cmake_cache]
                    if args.patched_cmake_cache
                    else []
                ),
                "--",
                *args.command,
            ]
        )
    )

    validate_release_cmd = sub.add_parser(
        "probe-release",
        help="probe patch compatibility against a ref in an isolated checkout (HI46)",
    )
    validate_release_cmd.add_argument("--run-id", required=True)
    validate_release_cmd.add_argument("--staging-root", default=None)
    validate_release_cmd.add_argument("--ref", default="master")
    validate_release_cmd.add_argument("--recipe", default="bigcherry")
    validate_release_cmd.add_argument("--inventory", default=None)
    validate_release_cmd.set_defaults(
        func=lambda args: _validate_release_main(
            [
                "--run-id",
                args.run_id,
                "--ref",
                args.ref,
                "--recipe",
                args.recipe,
                *(["--inventory", args.inventory] if args.inventory else []),
                *(["--staging-root", args.staging_root] if args.staging_root else []),
            ]
        )
    )

    validate_ref_cmd = sub.add_parser(
        "validate-ref",
        help="alias for the isolated patch/build compatibility probe (HI46)",
    )
    validate_ref_cmd.add_argument("--run-id", required=True)
    validate_ref_cmd.add_argument("--staging-root", default=None)
    validate_ref_cmd.add_argument("--ref", default="master")
    validate_ref_cmd.add_argument("--recipe", default="bigcherry")
    validate_ref_cmd.add_argument("--inventory", default=None)
    validate_ref_cmd.add_argument("--promoted-winners", default=None)
    validate_ref_cmd.set_defaults(
        func=lambda args: _validate_release_main(
            [
                "--run-id",
                args.run_id,
                "--ref",
                args.ref,
                "--recipe",
                args.recipe,
                *(["--inventory", args.inventory] if args.inventory else []),
                *(
                    ["--promoted-winners", args.promoted_winners]
                    if args.promoted_winners
                    else []
                ),
                *(["--staging-root", args.staging_root] if args.staging_root else []),
            ]
        )
    )

    rank_replay_cmd = sub.add_parser(
        "rank-replay",
        help="report/replay ranking-policy decisions recorded in a measurements file (HI50)",
    )
    rank_replay_cmd.add_argument("measurements")
    rank_replay_cmd.add_argument(
        "--dispatch", help="full per-policy candidate detail for one dispatch"
    )
    rank_replay_cmd.add_argument(
        "--verify-parity",
        action="store_true",
        help="assert the production policy's pick matches provisional_winner",
    )
    rank_replay_cmd.add_argument(
        "--policy-module",
        help="registry name, dotted module path, or .py file of a "
        "not-yet-installed policy to replay alongside the recorded ones",
    )
    rank_replay_cmd.add_argument("--output", help="write the JSON report here too")
    rank_replay_cmd.add_argument(
        "--json", action="store_true", help="print JSON instead of a text summary"
    )
    rank_replay_cmd.set_defaults(
        func=lambda args: _rank_replay_main(
            [
                args.measurements,
                *(["--dispatch", args.dispatch] if args.dispatch else []),
                *(["--verify-parity"] if args.verify_parity else []),
                *(
                    ["--policy-module", args.policy_module]
                    if args.policy_module
                    else []
                ),
                *(["--output", args.output] if args.output else []),
                *(["--json"] if args.json else []),
            ]
        )
    )

    resource = sub.add_parser(
        "resource-report", help="parse and policy-check a compiler resource stream"
    )
    resource.add_argument("raw")
    resource.add_argument("--symbol-map", required=True)
    resource.add_argument("--output", required=True)
    resource.add_argument("--compiler-family", default="clang")
    resource.add_argument("--compiler-major", type=int, required=True)
    resource.add_argument("--compiler-version", required=True)
    resource.add_argument("--architecture", required=True)
    resource.add_argument("--source-revision", required=True)
    resource.add_argument("--manifest-hash", required=True)
    resource.add_argument("--reject-lds-gt", type=int, default=None)
    resource.add_argument("--warn-occupancy-lt", type=float, default=None)
    resource.set_defaults(
        func=lambda args: _resource_report_main(
            [
                args.raw,
                "--symbol-map",
                args.symbol_map,
                "--output",
                args.output,
                "--compiler-family",
                args.compiler_family,
                "--compiler-major",
                str(args.compiler_major),
                "--compiler-version",
                args.compiler_version,
                "--architecture",
                args.architecture,
                "--source-revision",
                args.source_revision,
                "--manifest-hash",
                args.manifest_hash,
                *(
                    ["--reject-lds-gt", str(args.reject_lds_gt)]
                    if args.reject_lds_gt is not None
                    else []
                ),
                *(
                    ["--warn-occupancy-lt", str(args.warn_occupancy_lt)]
                    if args.warn_occupancy_lt is not None
                    else []
                ),
            ]
        )
    )

    binsize = sub.add_parser(
        "candidate-binary-size",
        help="per-candidate device .text size from a built HIP library",
    )
    binsize.add_argument("library")
    binsize.add_argument("--manifest", required=True)
    binsize.add_argument("--output", required=True)
    binsize.add_argument("--workdir", default=None)
    binsize.add_argument("--symbol-map-dir", default=None)
    binsize.add_argument("--objdump", default=None)
    binsize.add_argument("--readelf", default=None)
    binsize.add_argument("--allow-unresolved", action="store_true")
    binsize.set_defaults(
        func=lambda args: _candidate_binary_size_main(
            [
                args.library,
                "--manifest",
                args.manifest,
                "--output",
                args.output,
                *(["--workdir", args.workdir] if args.workdir else []),
                *(
                    ["--symbol-map-dir", args.symbol_map_dir]
                    if args.symbol_map_dir
                    else []
                ),
                *(["--objdump", args.objdump] if args.objdump else []),
                *(["--readelf", args.readelf] if args.readelf else []),
                *(["--allow-unresolved"] if args.allow_unresolved else []),
            ]
        )
    )

    from .. import report as _report

    _report.build_parser(sub)

    from .. import impact as _impact
    from .. import kernel_fraction as _kernel_fraction

    _impact.build_parser(sub)
    _kernel_fraction.build_parser(sub)

    # Inventory: convert record JSONL → SQLite + inventory JSON, or load tuning measurements.
    inventory = sub.add_parser(
        "inventory",
        help="Convert record JSONL to inventory/DB, or load tuning measurements",
    )
    inv_sub = inventory.add_subparsers(dest="inv_subcommand")

    # Record mode: JSONL → SQLite + inventory JSON (existing behavior)
    inv_record = inv_sub.add_parser(
        "record", help="Convert record-mode JSONL to inventory + DB"
    )
    inv_record.add_argument("record", help="JSONL written by GGML_HIP_DISPATCH_DB")
    inv_record.add_argument(
        "--inventory",
        default=None,
        help="inventory JSON to write (default: alongside)",
    )
    inv_record.add_argument(
        "--database",
        default=None,
        help="SQLite database to write (default: alongside)",
    )
    inv_record.set_defaults(func=lambda args: cmd_inventory(args, subcmd="record"))

    # Tuning mode: measurements JSONL → SQLite with winners/measurements/candidates
    inv_tuning = inv_sub.add_parser(
        "tuning", help="Load tuning measurements into SQLite"
    )
    inv_tuning.add_argument(
        "measurements",
        help="JSONL written by GGML_HIP_DISPATCH_DB (the .measurements.jsonl file)",
    )
    inv_tuning.add_argument(
        "--database",
        default=None,
        help="SQLite database path (default: alongside measurements, .sqlite extension)",
    )
    inv_tuning.add_argument(
        "--manifest",
        default=None,
        help="Manifest JSON for full candidate data (artifacts/<rev>/hip-autotune-manifest.json)",
    )
    inv_tuning.add_argument(
        "--signature-source",
        action="append",
        default=[],
        help="JSONL record/replay diagnostics file containing canonical shapes; may be repeated",
    )
    inv_tuning.set_defaults(func=lambda args: cmd_inventory(args, subcmd="tuning"))

    # Hot list: rank observed signatures by estimated time contribution
    # (HI24 steps 5-6), consumed by GGML_HIP_TUNE_HOT_SIGNATURES.
    inv_hot = inv_sub.add_parser(
        "hot-list", help="Rank observed signatures by estimated time contribution"
    )
    inv_hot.add_argument("record", help="JSONL written by GGML_HIP_DISPATCH_DB")
    inv_hot.add_argument(
        "--measurements",
        default=None,
        help="a previous tune's .measurements.jsonl; upgrades the ranking "
        "from calls x est_bytes to calls x native_median_us",
    )
    inv_hot.add_argument(
        "--output", default=None, help="hot list to write (default: alongside)"
    )
    inv_hot.set_defaults(func=lambda args: cmd_inventory(args, subcmd="hot-list"))

    # Workload overlap: how much of a record's workload has a tuned winner,
    # call-weighted (HI37 Part 2). The tuned signature set comes either from
    # a measurements JSONL's winners or (HI101) from a binary v5 replay cache
    # via replay_cache.read_cache() -- the union of entry signatures, which
    # is the same hardware-agnostic semantics as the measurements path
    # (v5 entries carry no separate hardware field; it is folded into the
    # portable dispatch digest).
    inv_workload = inv_sub.add_parser(
        "workload-check",
        help="Report call-weighted signature coverage of a measurements file "
        "or a v5 replay cache against a record",
    )
    inv_workload.add_argument("record", help="JSONL written by GGML_HIP_DISPATCH_DB")
    tuned_source = inv_workload.add_mutually_exclusive_group(required=True)
    tuned_source.add_argument(
        "--measurements",
        default=None,
        help="a .measurements.jsonl file whose winners define the tuned set",
    )
    tuned_source.add_argument(
        "--cache",
        default=None,
        help="a binary v5 replay cache whose entry signatures define the tuned set",
    )
    inv_workload.set_defaults(
        func=lambda args: cmd_inventory(args, subcmd="workload-check")
    )

    return parser


def _tune_journal_main(argv: list[str]) -> int:
    from .. import tune_journal

    return tune_journal.main(argv)


def _tune_promote_main(argv: list[str]) -> int:
    from .. import tune_promotion

    return tune_promotion.main(argv)


def _tune_null_fdr_main(argv: list[str]) -> int:
    from .. import tune_promotion

    return tune_promotion.null_fdr_main(argv)


def _experiment_main(argv: list[str]) -> int:
    from .. import experiment_bundle

    return experiment_bundle.main(argv)


def _ab_benchmark_main(argv: list[str]) -> int:
    from .. import ab_benchmark

    return ab_benchmark.main(argv)


def _campaign_build_main(argv: list[str]) -> int:
    from .. import re14_real_run

    # REMAINDER captures a leading "--" (needed so the outer parser doesn't
    # try to consume flags like --upstream-repo itself) literally as part
    # of argv -- strip it before forwarding, same as ab_benchmark.main()
    # already does for its own REMAINDER-captured command.
    if argv[:1] == ["--"]:
        argv = argv[1:]
    return re14_real_run.main(argv)


def _resource_report_main(argv: list[str]) -> int:
    from .. import resource_report

    return resource_report.main(argv)


def _validate_release_main(argv: list[str]) -> int:
    from .. import release_validate

    return release_validate.main(argv)


def _rank_replay_main(argv: list[str]) -> int:
    from .. import rank_replay

    return rank_replay.main(argv)


def _candidate_binary_size_main(argv: list[str]) -> int:
    from .. import candidate_binary_size

    return candidate_binary_size.main(argv)


# TR03 compatibility facades: the canonical pin handlers now live in
# bigcherry.release.pin while these names remain stable for existing callers.


def cmd_repin(args: argparse.Namespace) -> int:
    return _release_pin.cmd_repin(args)


def cmd_pin_status(args: argparse.Namespace) -> int:
    return _release_pin.cmd_pin_status(args)


def _legacy_main(argv: list[str] | None = None) -> int:
    _configure_output()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def _configure_output() -> None:
    """Keep diagnostic output printable on Windows' legacy code pages.

    Reports intentionally use a few Unicode layout glyphs.  Preserve the
    console's selected encoding, but replace characters it cannot represent
    instead of allowing a diagnostic command to fail while printing it.
    Captured/test streams may not implement ``reconfigure``; those are left
    untouched.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")


def main(argv: list[str] | None = None) -> int:
    """Run the existing parser/handler implementation through the new entrypoint."""
    return int(_legacy_main(argv))
