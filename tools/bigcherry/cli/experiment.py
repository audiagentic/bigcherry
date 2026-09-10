"""CLI presentation handlers for Experiment Contract workflows (EC11)."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

from ..core import paths


def _load_contract_registry(args: Namespace):
    from ..experiment import contract as ec

    override = bool(args.contracts)
    contracts_path = Path(args.contracts) if override else paths.EXPERIMENT_CONTRACTS
    # The registry cross-checks are applied ONLY to the repo's own registry.
    # Both are opt-in by design (see known_source_ids_from_external_sources'
    # docstring): a caller pointing --contracts at their own file is working
    # in isolation and is not forced to also maintain matching
    # external-sources.toml / models.toml fixtures. Loading the real registry
    # is the case where an unregistered source or model IS a defect, so this
    # is where the check belongs -- before this, neither cross-check ran
    # anywhere outside the unit tests, so a contract naming a model that had
    # never existed validated clean.
    if override:
        return ec, ec.load_contracts(contracts_path)
    return ec, ec.load_contracts(
        contracts_path,
        known_source_ids=ec.known_source_ids_from_external_sources(),
        known_model_ids=ec.known_model_ids_from_models_registry(),
    )


def cmd_experiment_validate(args: Namespace) -> int:
    """Schema-check every contract in the registry (or just --contract-id,
    if named) without running anything -- EC01/EC02's own validation, the
    cheap check before any real campaign work (same discipline as
    `patcher.apply_all(dry_run=True)` before a real build)."""
    from ..experiment import contract as ec

    try:
        _, registry = _load_contract_registry(args)
    except ec.ExperimentContractError as exc:
        print(f"experiment-contract validate: {exc}", file=sys.stderr)
        return 1
    ids = [args.contract_id] if args.contract_id else sorted(registry.contracts)
    if args.contract_id and args.contract_id not in registry.contracts:
        print(
            f"experiment-contract validate: no such contract {args.contract_id!r}",
            file=sys.stderr,
        )
        return 1
    for contract_id in ids:
        contract = registry[contract_id]
        print(
            f"  [ OK ] {contract_id}: {contract.title} (hash {contract.contract_hash})"
        )

    # VA24 registry lint: a contract declaring a gain threshold under the
    # weaker point_estimate_v1 policy must appear in the frozen legacy
    # manifest. Runs over the WHOLE registry even when --contract-id narrowed
    # the listing above, because the property being checked is a registry
    # invariant (is anything using the weak policy without a waiver), not a
    # property of one contract in isolation.
    from pathlib import Path as _Path

    waivers = ec.load_legacy_waivers(_Path(ec.LEGACY_MANIFEST_PATH))
    problems = ec.lint_effect_evidence_policy(registry, waivers)
    if problems:
        for problem in problems:
            print(f"  [FAIL] {problem}", file=sys.stderr)
        return 1
    return 0


def cmd_experiment_list(args: Namespace) -> int:
    from ..experiment import contract as ec

    try:
        _, registry = _load_contract_registry(args)
    except ec.ExperimentContractError as exc:
        print(f"experiment-contract list: {exc}", file=sys.stderr)
        return 1
    if not registry.contracts:
        print("  (no contracts registered)")
        return 0
    for contract_id in sorted(registry.contracts):
        contract = registry[contract_id]
        target_label = (
            f"{contract.target.kind}:{contract.target.family}"
            if contract.target.family is not None
            else contract.target.kind
        )
        print(
            f"  {contract_id:<20} target={target_label:<20} "
            f"source={contract.source.source_id} -- {contract.title}"
        )
    return 0


def cmd_experiment_plan(args: Namespace) -> int:
    """Show what campaign lanes a contract would expand into (EC03),
    without executing them -- a dry-run, matching this project's existing
    --dry-run conventions."""
    from ..core import config as campaign_config
    from ..experiment import contract as ec
    from ..campaign.planner import CampaignPlannerError, expand_contract

    try:
        _, registry = _load_contract_registry(args)
        contract = registry[args.contract_id]
    except (ec.ExperimentContractError, KeyError) as exc:
        print(f"experiment-contract plan: {exc}", file=sys.stderr)
        return 1
    context_config_path = Path(args.config) if args.config else paths.RECIPES
    try:
        cfg = campaign_config.load(context_config_path)
        lanes = expand_contract(
            contract,
            cfg=cfg,
            source_name=args.source,
            build_name=args.build,
            platform_name=args.platform,
        )
    except (campaign_config.ConfigError, CampaignPlannerError) as exc:
        print(f"experiment-contract plan: {exc}", file=sys.stderr)
        return 1
    print(f"  contract {contract.id}: {len(lanes)} lane(s)")
    for lane in lanes:
        detail = f"role={lane.role}"
        if lane.workload_tag:
            detail += f" workload={lane.workload_tag}"
        if lane.model_ref:
            detail += f" model={lane.model_ref}"
        if lane.boundary_dimension:
            detail += f" {lane.boundary_dimension}={lane.boundary_value}"
        print(
            f"    {lane.source_name}:{lane.build_name}:{lane.platform_name} ({detail})"
        )
    return 0


def cmd_experiment_run(args: Namespace) -> int:
    """Execute a contract's full lane set (EC03/EC04) through
    run_campaign() -- the real materialize/generate/build/smoke
    orchestration, identical to `bigcherry build`'s own engine. Does NOT
    yet run the correctness/aggregation/promotion chain (EC06-EC09) --
    those consume real benchmark measurements this command does not
    itself capture; wire a real comparison harness (see
    tools/bigcherry/re15_acceptance_run.py's stage-by-stage pattern) on
    top of this lane execution to close that gap."""
    from ..core import config as campaign_config
    from ..experiment import contract as ec
    from ..core.artifacts import ArtifactStore
    from ..campaign.planner import CampaignPlannerError, expand_contract, run_campaign
    from ..core.context import ProjectContext

    try:
        _, registry = _load_contract_registry(args)
        contract = registry[args.contract_id]
    except (ec.ExperimentContractError, KeyError) as exc:
        print(f"experiment-contract run: {exc}", file=sys.stderr)
        return 1
    context = ProjectContext.resolve(work_root=None)
    try:
        cfg = campaign_config.load(Path(args.config) if args.config else paths.RECIPES)
        lanes = expand_contract(
            contract,
            cfg=cfg,
            source_name=args.source,
            build_name=args.build,
            platform_name=args.platform,
        )
    except (campaign_config.ConfigError, CampaignPlannerError) as exc:
        print(f"experiment-contract run: {exc}", file=sys.stderr)
        return 1
    store = ArtifactStore(context.work_root / "artifacts-store")
    results = run_campaign(lanes, cfg=cfg, context=context, store=store)
    failed = 0
    for lane_id, result in results.items():
        if isinstance(result, Exception):
            failed += 1
            print(f"  [FAIL] {lane_id}: {type(result).__name__}: {result}")
        else:
            print(f"  [ OK ] {lane_id}: binary={result.binary_ref.artifact_id}")
    return 1 if failed else 0


def cmd_experiment_report(args: Namespace) -> int:
    """Render EC10's report from a stored evidence JSON file (produced by
    a caller that ran EC06/EC07/EC08/EC09 against real measurements --
    this command does not itself run anything, matching how `report`
    stays a pure rendering step over already-collected evidence)."""
    from ..experiment import contract as ec

    try:
        _, registry = _load_contract_registry(args)
        contract = registry[args.contract_id]
    except (ec.ExperimentContractError, KeyError) as exc:
        print(f"experiment-contract report: {exc}", file=sys.stderr)
        return 1
    try:
        evidence = json.loads(Path(args.evidence_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"experiment-contract report: cannot read {args.evidence_file}: {exc}",
            file=sys.stderr,
        )
        return 1
    correctness_gate = evidence.get("correctness_gate", {})
    aggregated_effects = evidence.get("aggregated_effects", {})
    promotion_gate = evidence.get("promotion_gate") or ec.evaluate_promotion_gate(
        contract,
        correctness_gate=correctness_gate,
        aggregated_effects=aggregated_effects,
        generalisation_result=evidence.get("generalisation_result"),
    )
    print(
        ec.render_report(
            contract,
            correctness_gate=correctness_gate,
            aggregated_effects=aggregated_effects,
            promotion_gate=promotion_gate,
            generalisation_result=evidence.get("generalisation_result"),
        )
    )
    return 0 if promotion_gate.get("passed") else 1
