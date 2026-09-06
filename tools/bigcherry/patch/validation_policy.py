"""RD-patch validation-package static policy (VA02, docs/reference/testing/
PATCH_VALIDATION.md). Separate from ``catalog.py``'s legacy catalog/module
cross-check and from the HI83 dynamic evidence-freshness verifier
(``validation_evidence_statuses`` -- see VA08): this module owns only the
STATIC question of whether a patch's validation *package* (README.md,
validation.toml, an Experiment Contract binding, and a resolvable
ValidationPlan) is present and well-formed for the tracked-status it
claims. It never inspects evidence content or freshness, and never
executes patch code (custom-validator specs are checked structurally, not
imported -- see ``validate_custom_callable_spec``).

The one-time structural-grandfather exemption defined here protects only
the SHAPE of a legacy patch's package (or lack of one) -- see
``load_grandfather_baseline``/``check_validation_packages``. It never
authorizes starting a NEW validation execution; that hard requirement is
enforced by ``require_execution_package``, called from
``validation_campaign.run()`` regardless of grandfather status.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import registry as patch_registry
from . import validation as patch_validation
from ..core import paths
from ..source import sources as source_registry

VALIDATION_PACKAGE_POLICY_VERSION = "rd-validation-package-v1"

# Tracked-status values (source/sources.py::TRACKED_STATUSES) that require
# a full validation package before they may be reported as current.
PACKAGE_STATUSES = frozenset({"ported-benched", "ported-validated", "deferred-hardware"})

_RD_PLAN_ITEM = re.compile(r"^RD\d+$")


def is_rd_validation_patch(descriptor: patch_registry.PatchDescriptor) -> bool:
    """Whether this patch's plan-item binding puts it in RD/experimental
    validation-package scope, per GPT round-6 review (req_6e54ebe0b3764350):
    plan_ids and legacy plan_item are the planning-namespace signals -- NOT
    descriptor.kind ('enhancement' vs 'framework'), which describes
    implementation class, not scope, and would silently miss or
    misclassify future patches. Handles compound bindings like
    'RD05/RD06/RD07' (one packaged patch, several distinct hypotheses)."""
    return any(
        _RD_PLAN_ITEM.fullmatch(token.strip())
        for binding in (*descriptor.plan_ids, descriptor.plan_item or "")
        for token in binding.split("/")
    )


class PolicyError(ValueError):
    """A structural policy violation that is never grandfatherable (a
    malformed patch.toml/validation.toml, an unknown contract id, a path
    escape, a missing required producer, ...)."""


def is_framework_configuration_patch(descriptor: patch_registry.PatchDescriptor) -> bool:
    """Return whether *descriptor* is a package-only framework configuration.

    This classification is deliberately independent of lifecycle state.  A
    bound Experiment Contract or RD plan binding makes the patch experimental
    instead, even when its implementation kind is ``framework``.
    """
    return (
        descriptor.representation == patch_registry.REPRESENTATION_PACKAGED
        and descriptor.kind == "framework"
        and descriptor.origin == "local"
        and not descriptor.external_source
        and not descriptor.experiment_contracts
        and not is_rd_validation_patch(descriptor)
    )


@dataclass(frozen=True)
class PackagePolicyStatus:
    patch_id: str
    # "current" | "grandfathered" | "not-required" | "invalid"
    status: str
    problems: tuple[str, ...] = ()


@dataclass(frozen=True)
class PackagePolicyReport:
    statuses: tuple[PackagePolicyStatus, ...] = ()
    problems: tuple[str, ...] = ()
    grandfathered: tuple[str, ...] = ()


def tracked_statuses_for_patch(
    patch_id: str, *, external_sources_path: str | Path | None = None
) -> tuple[str, ...]:
    """Every distinct tracked-status ever recorded for this exact patch id
    across config/external-sources.toml's [[sources.tracked]] entries.
    Exact-patch binding (``entry["patch"] == patch_id``), not plan-item
    aggregation -- a plan item can span multiple patches/hypotheses."""
    registry = source_registry.load_registry(external_sources_path)
    statuses: set[str] = set()
    for src in registry.get("sources", ()):
        for entry in src.get("tracked") or []:
            if entry.get("patch") == patch_id:
                status = entry.get("status")
                if status:
                    statuses.add(status)
    return tuple(sorted(statuses))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _grandfather_identity(
    descriptor: patch_registry.PatchDescriptor,
    root: Path,
    *,
    external_sources_path: str | Path | None = None,
) -> dict:
    toml_path = root / descriptor.metadata_path
    return {
        "implementation_digest": descriptor.implementation_digest,
        "patch_toml_digest": _sha256_bytes(toml_path.read_bytes()),
        "tracked_statuses": sorted(
            set(
                tracked_statuses_for_patch(
                    descriptor.patch_id, external_sources_path=external_sources_path
                )
            )
        ),
    }


_EMPTY_BASELINE = {"schema_version": 1, "policy_version": VALIDATION_PACKAGE_POLICY_VERSION, "patches": {}}


def load_grandfather_baseline(path: str | Path | None = None) -> dict:
    """Load the one-time reviewed baseline. Malformed, missing, or
    shape-invalid -> empty (fail closed: no entry means no exemption,
    never an error that's silently treated as a pass)."""
    baseline_path = Path(path) if path is not None else paths.VALIDATION_PACKAGE_GRANDFATHER
    if not baseline_path.is_file():
        return dict(_EMPTY_BASELINE)
    try:
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(_EMPTY_BASELINE)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        return dict(_EMPTY_BASELINE)
    if not isinstance(data.get("policy_version"), str):
        return dict(_EMPTY_BASELINE)
    patches = data.get("patches")
    if not isinstance(patches, dict) or not all(isinstance(v, dict) for v in patches.values()):
        return dict(_EMPTY_BASELINE)
    return data


def _is_grandfathered(
    descriptor: patch_registry.PatchDescriptor,
    root: Path,
    baseline: dict,
    *,
    external_sources_path: str | Path | None = None,
) -> bool:
    if baseline.get("policy_version") != VALIDATION_PACKAGE_POLICY_VERSION:
        return False
    entry = baseline.get("patches", {}).get(descriptor.patch_id)
    if entry is None:
        return False
    try:
        current = _grandfather_identity(
            descriptor, root, external_sources_path=external_sources_path
        )
    except OSError:
        return False
    return (
        entry.get("implementation_digest") == current["implementation_digest"]
        and entry.get("patch_toml_digest") == current["patch_toml_digest"]
        and sorted(entry.get("tracked_statuses", ())) == current["tracked_statuses"]
    )


def validate_custom_callable_spec(spec: str, *, package_root: str | Path) -> None:
    """Static (non-executing) equivalent of
    ``validation.resolve_custom_callable``: path containment, file
    existence, and a locked ``check(ctx)`` signature via AST inspection --
    never imports/executes the module. Raises PolicyError on any
    violation."""
    import ast
    import re

    match = re.match(r"^(?P<module>.+\.py):(?P<attr>[A-Za-z_][A-Za-z0-9_]*)$", spec)
    if match is None:
        raise PolicyError(f"custom callable {spec!r} must be 'path/to/checks.py:function_name'")
    root = Path(package_root).resolve()
    candidate = (root / match.group("module")).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise PolicyError(f"custom callable {spec!r} escapes the package root {root}") from None
    if not candidate.is_file():
        raise PolicyError(f"custom callable {spec!r}: no file at {candidate}")

    tree = ast.parse(candidate.read_text(encoding="utf-8"), filename=str(candidate))
    attr = match.group("attr")
    func_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == attr:
            func_node = node
            break
    if func_node is None:
        raise PolicyError(f"custom callable {spec!r}: no sync function {attr!r} found")
    args = func_node.args
    if (
        len(args.posonlyargs) + len(args.args) != 1
        or args.vararg is not None
        or args.kwarg is not None
        or args.kwonlyargs
        or (args.posonlyargs + args.args)[0].arg != "ctx"
    ):
        raise PolicyError(
            f"custom callable {spec!r}: signature must be check(ctx) with exactly "
            "one positional argument, no *args/**kwargs/extra parameters"
        )


def check_validation_packages(
    *,
    root: Path | None = None,
    registry_path: Path | None = None,
    external_sources_path: str | Path | None = None,
    baseline_path: str | Path | None = None,
) -> PackagePolicyReport:
    """The real VA02 static check: for every packaged patch whose tracked
    statuses intersect PACKAGE_STATUSES, require README.md +
    validation.toml + a resolved Experiment Contract + a buildable
    ValidationPlan (build_plan_for_patch already enforces contract
    resolution, compatibility, and required-capability producer
    coverage). Grandfathered legacy patches are reported non-fatally;
    everything else that's missing or malformed is a real problem."""
    resolved_root = root or paths.PATCHES
    reg = patch_registry.load_registry(registry_path or resolved_root)
    baseline = load_grandfather_baseline(baseline_path)

    statuses: list[PackagePolicyStatus] = []
    problems: list[str] = []
    grandfathered: list[str] = []

    for descriptor in reg.descriptors:
        if descriptor.representation != patch_registry.REPRESENTATION_PACKAGED:
            continue
        tracked = set(
            tracked_statuses_for_patch(
                descriptor.patch_id, external_sources_path=external_sources_path
            )
        )
        # GPT round-5 (req_86cfd3a0bff04716) directed: a descriptor.state=
        # "validated" patch must be covered even if its tracked-status
        # metadata is absent or doesn't itself include a PACKAGE_STATUSES
        # value. First scoped this to kind="enhancement" after finding it
        # unscoped swept in every core framework patch; GPT round 6
        # (req_6e54ebe0b3764350) corrected the discriminator itself:
        # descriptor.kind describes implementation class, not planning
        # scope, and would silently miss/misclassify future patches.
        # is_rd_validation_patch() uses the authoritative RD plan-item
        # binding instead (handles compound bindings like RD05/RD06/RD07).
        is_rd_patch = is_rd_validation_patch(descriptor)
        framework_adapter = (
            is_framework_configuration_patch(descriptor) and descriptor.validation_path is not None
            and not (tracked & PACKAGE_STATUSES)
        )
        requires_package = bool(tracked & PACKAGE_STATUSES) or (
            is_rd_patch and descriptor.state == "validated"
        ) or framework_adapter
        requires_architectures = ("ported-validated" in tracked) or (
            is_rd_patch and descriptor.state == "validated"
        )
        if not requires_package:
            statuses.append(PackagePolicyStatus(descriptor.patch_id, "not-required"))
            continue

        package_root = resolved_root / (descriptor.package_root or descriptor.patch_id)
        readme_exists = (package_root / "README.md").is_file()
        # VA17 policy slice: plural -- a patch bound to 2+ contracts (e.g.
        # 1203's RD05/RD06/RD07) has a contract just as much as one bound to
        # exactly 1; the old singular check would report "no
        # experiment-contract bound" for a real, multi-contract patch.
        has_contract = bool(descriptor.experiment_contracts)
        has_adapter = descriptor.validation_path is not None

        patch_problems: list[str] = []
        if not readme_exists:
            patch_problems.append(f"{descriptor.patch_id}: missing README.md")
        if not has_contract and not framework_adapter:
            patch_problems.append(f"{descriptor.patch_id}: no experiment-contract bound in patch.toml")
        if not has_adapter:
            patch_problems.append(f"{descriptor.patch_id}: no validation.toml adapter")

        plan = None
        if (has_contract or framework_adapter) and has_adapter:
            try:
                plan = patch_validation.build_plan_for_patch(descriptor, root=resolved_root)
            except patch_validation.ValidationError as exc:
                patch_problems.append(f"{descriptor.patch_id}: {exc}")

        if plan is not None:
            # Custom-validator specs are checked structurally (AST, never
            # imported/executed) -- parse_validation_toml() leaves
            # validator-specific config opaque, so this was previously
            # dead code: a malformed/missing/escaping custom callable
            # would lint clean as long as producer coverage succeeded.
            for spec in plan.checks:
                if spec.validator == "custom":
                    callable_spec = spec.config.get("callable")
                    if not isinstance(callable_spec, str) or not callable_spec:
                        patch_problems.append(
                            f"{descriptor.patch_id}: check {spec.check_id!r} has validator=custom "
                            "but no 'callable' string in its config"
                        )
                        continue
                    try:
                        validate_custom_callable_spec(callable_spec, package_root=package_root)
                    except (PolicyError, SyntaxError, OSError) as exc:
                        patch_problems.append(f"{descriptor.patch_id}: check {spec.check_id!r}: {exc}")
                elif spec.validator not in patch_validation.BUILTIN_VALIDATORS:
                    patch_problems.append(
                        f"{descriptor.patch_id}: check {spec.check_id!r} uses unknown validator {spec.validator!r}"
                    )
                # GPT round 6 (req_6e54ebe0b3764350): build_validation_plan()
                # only calls _produces() for universal/contract-required
                # capabilities -- a SUPPLEMENTARY check (one the adapter adds
                # beyond what's required) with a capability its validator
                # can't actually produce (e.g. capability="performance",
                # validator="apply") would otherwise silently enter the plan
                # and later emit a PASS mislabeled as that capability.
                if not patch_validation.validator_produces(spec.capability, spec.validator):
                    patch_problems.append(
                        f"{descriptor.patch_id}: check {spec.check_id!r} declares "
                        f"capability={spec.capability!r} but validator={spec.validator!r} "
                        "cannot produce it"
                    )

        if plan is not None and requires_architectures:
            if not descriptor.validation_architectures:
                patch_problems.append(
                    f"{descriptor.patch_id}: ported-validated requires non-empty validation-architectures"
                )
            elif plan.contracts:
                # VA17 policy slice: the required set is the exact UNION of
                # every bound contract's own required architectures (1203:
                # gfx1030/gfx1100/gfx1201, since RD05/RD06 require gfx1201
                # and RD07 requires all three) -- not just one contract's
                # view. A contract declaring no architectures at all
                # contributes nothing to the union (unconstrained, not "no
                # architectures allowed").
                required: set[str] = set()
                for binding in plan.contracts:
                    required.update(binding.architectures)
                if required:
                    declared = set(descriptor.validation_architectures)
                    if declared != required:
                        patch_problems.append(
                            f"{descriptor.patch_id}: validation-architectures {sorted(declared)} "
                            f"does not match contract-required architectures {sorted(required)}"
                        )

        shape_missing = not readme_exists or (not has_contract and not framework_adapter) or not has_adapter
        # Only the SHAPE-absence deficiencies above are grandfatherable.
        # A patch that HAS validation files but they're malformed (bad
        # contract id, missing producer, path escape, etc.) is never
        # exempt -- those problems already landed in patch_problems from
        # the ValidationError branch and stay fatal regardless.
        malformed_present = plan is None and (has_contract or framework_adapter) and has_adapter

        if shape_missing and not malformed_present and not framework_adapter and _is_grandfathered(
            descriptor, resolved_root, baseline, external_sources_path=external_sources_path
        ):
            grandfathered.append(descriptor.patch_id)
            statuses.append(
                PackagePolicyStatus(descriptor.patch_id, "grandfathered", tuple(patch_problems))
            )
            continue

        if patch_problems:
            problems.extend(patch_problems)
            statuses.append(PackagePolicyStatus(descriptor.patch_id, "invalid", tuple(patch_problems)))
        else:
            statuses.append(PackagePolicyStatus(descriptor.patch_id, "current"))

    return PackagePolicyReport(
        statuses=tuple(statuses), problems=tuple(problems), grandfathered=tuple(grandfathered)
    )


def require_execution_package(
    descriptor: patch_registry.PatchDescriptor, *, root: Path | None = None
) -> patch_validation.ValidationPlan:
    """Execution-side anti-grandfather guard (VA02): starting a NEW
    validation run for a patch requires a real README + validation.toml +
    resolved Experiment Contract for experimental work, regardless of any
    lint-side structural-grandfather exemption. Local non-RD framework work
    may execute its adapter without a scientific contract; this does not
    grant eligibility or relax the persisted evidence verifier.
    Never consults grandfather status --
    that status only ever makes lint non-fatal, never authorizes a run."""
    resolved_root = root or paths.PATCHES
    package_root = resolved_root / (descriptor.package_root or descriptor.patch_id)
    if not (package_root / "README.md").is_file():
        raise PolicyError(
            f"{descriptor.patch_id}: cannot start a new validation run -- missing README.md "
            "(structural grandfathering never authorizes execution)"
        )
    # VA17 policy slice: plural -- require_execution_package() must accept
    # a real multi-contract patch (e.g. 1203's RD05/RD06/RD07), not just
    # 0/1-contract patches.
    local_framework = is_framework_configuration_patch(descriptor)
    if not descriptor.experiment_contracts and not local_framework:
        raise PolicyError(
            f"{descriptor.patch_id}: cannot start a new validation run -- no experiment-contract bound"
        )
    if descriptor.validation_path is None:
        raise PolicyError(
            f"{descriptor.patch_id}: cannot start a new validation run -- no validation.toml adapter"
        )
    plan = patch_validation.build_plan_for_patch(descriptor, root=resolved_root)
    if plan is None:
        raise PolicyError(f"{descriptor.patch_id}: cannot start a new validation run -- no resolvable ValidationPlan")
    return plan
