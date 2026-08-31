"""Patch validation framework: plans, requirement aggregation, verdicts
(patch-system PA03 / RS07-RS08; runbook sections 17-21, 31-33, 60-61).

    patch.toml            what is the patch?
    Experiment Contract   what must be demonstrated?
    validation.toml       which implementation produces each piece of
                          evidence?

``validation.toml`` is an execution ADAPTER, not a hypothesis language:
it may ADD requirements' producers but may never REMOVE requirements
imposed by the framework (universal) or the linked Experiment Contract
(section 18). If a required capability has no producer, the plan build
fails with :class:`ConfigurationError` -- never a silent skip.

This module computes PLANS and VERDICTS. It does not execute hardware
itself: validators receive a :class:`ValidationContext` whose helpers
(``run_binary``, ``register_artifact``, ...) carry the execution.

Locked v1 semantics (section 19): a required check is satisfied only by
``pass``; ``blocked``/``error`` are never success; ``not_applicable``
NEVER satisfies a required check (a validator returning it for one is a
contract violation -- demoted to ``error``); it is only meaningful for
non-required/advisory checks.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import tomllib
from dataclasses import dataclass, field
from types import MappingProxyType
from pathlib import Path
from typing import Any, Callable

from ..experiment import contract as experiment_contract
from . import registry as patch_registry

# B4: the validation framework semantic version. The constant is PINNED in
# patch_registry (the lower layer of the DAG -- the registry must not import
# this module) and re-exported here so both observable contracts hold:
# importable from patch_validation AND included in validation_digest.
# Bump on any semantic change to requirement aggregation, validator
# semantics, or result interpretation.
VALIDATION_FRAMEWORK_VERSION = patch_registry.VALIDATION_FRAMEWORK_VERSION

# ------------------------------------------------------------------- statuses

PASS = "pass"
FAIL = "fail"
BLOCKED = "blocked"
ERROR = "error"
NOT_APPLICABLE = "not_applicable"

VALIDATION_STATUSES: tuple[str, ...] = (PASS, FAIL, BLOCKED, ERROR, NOT_APPLICABLE)

# ------------------------------------------------------------ validator set
# Closed, documented (section 21). Deliberately NO general catch-all
# 'fallback' validator: a future concrete need adds a specifically defined
# validator with one specifically named capability, never one that
# satisfies arbitrary missing capabilities.
BUILTIN_VALIDATORS: tuple[str, ...] = (
    "apply",
    "build",
    "backend-ops",
    "trace-marker",
    "compile-option",
    "runtime-smoke",
    "autotune-campaign",
    "benchmark",
    "architecture",
    "custom",
)

# Which built-in validators are capable of producing which capabilities.
# 'custom' is intentionally excluded from every set here: a custom check
# satisfies ONLY the capability it itself declares in [[check]] (section
# 31), and the framework verifies the returned result matches that
# declaration. A built-in is 'capable' of a capability when its generic
# mechanics can produce that class of evidence at all.
_CAPABILITY_PRODUCERS: dict[str, frozenset[str]] = {
    "apply": frozenset({"apply"}),
    "build": frozenset({"build"}),
    "correctness": frozenset({"backend-ops", "autotune-campaign"}),
    "activation": frozenset({"trace-marker"}),
    "smoke": frozenset({"runtime-smoke"}),
    "performance": frozenset({"benchmark", "autotune-campaign"}),
    "controls": frozenset({"benchmark", "autotune-campaign"}),
    "configuration": frozenset({"compile-option"}),
    "architecture": frozenset({"architecture"}),
}


def _capability_is_custom(validator: str) -> bool:
    return validator == "custom"


def validator_produces(capability: str, validator: str) -> bool:
    """Public equivalent of ``_produces()``'s capability/validator check,
    for callers (VA02's static lint) that need to reject a supplementary
    check whose declared validator cannot actually produce its declared
    capability -- e.g. ``capability = "performance", validator = "apply"``
    would otherwise silently enter a plan and later emit a PASS mislabeled
    as a performance result. ``custom`` always returns True here: a custom
    check produces exactly whatever capability it declares (section 31),
    and it is not this function's job to second-guess that."""
    if _capability_is_custom(validator):
        return True
    return validator in _CAPABILITY_PRODUCERS.get(capability, frozenset())


class ValidationError(ValueError):
    """Base class for patch-validation failures."""


class ConfigurationError(ValidationError):
    """The validation plan cannot be built: unknown/duplicate check ids,
    a custom callable escaping the package or violating the locked API, or
    a required capability with no producer (section 18 fail-closed)."""


# ------------------------------------------------------------- result types


@dataclass(frozen=True)
class ArtifactRef:
    """A reference to an evidence artifact (section 20). Paths are
    repository-relative or runtime-storage-relative strings plus a content
    digest, so evidence stays compact/tracked while large payloads
    (build trees, binaries, raw logs) remain untracked under the ignored
    artifacts root (section 40)."""

    name: str
    path: str
    sha256: str


@dataclass(frozen=True)
class ValidationResult:
    """Immutable structured result of one check (section 20).

    Custom code can return one -- but it can never claim
    ``eligible_for_validated_state``: only :func:`compute_verdict`
    computes that, centrally.
    """

    check_id: str
    capability: str
    status: str
    summary: str
    details: tuple[str, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in VALIDATION_STATUSES:
            raise ConfigurationError(
                f"validation result for {self.check_id!r} has unknown status "
                f"{self.status!r} (expected one of {', '.join(VALIDATION_STATUSES)})"
            )


# ------------------------------------------------------------------ CheckSpec


@dataclass(frozen=True)
class CheckSpec:
    """One ``[[check]]`` entry of a validation.toml adapter (section 17).

    ``config`` holds the validator-specific keys (``ops``, ``marker-regex``,
    ``callable``, ``negative-control`` table, ...). The generic parser
    accepts them opaquely; each built-in validator validates its own
    config schema (RS09).
    """

    check_id: str
    capability: str
    validator: str
    required: bool
    config: dict[str, Any] = field(default_factory=dict)


_CHECK_REQUIRED_KEYS = ("id", "capability", "validator")


# Experiment Contract field names (section 16): acceptance thresholds,
# hypothesis content, scope, etc. NEVER belong in validation.toml (the
# adapter says WHICH implementation produces evidence, never WHAT must be
# demonstrated). Rejected inside [[check]] config as well as top-level.
_CONTRACT_AUTHORITY_KEYS: frozenset[str] = frozenset({
    "acceptance", "hypothesis", "rationale", "expected-effect", "expected_effect",
    "target-kernel-gain-pct", "end-to-end-gain-pct", "max-control-regression-pct",
    "target_kernel_gain_pct", "end_to_end_gain_pct", "max_control_regression_pct",
    "controls", "boundaries", "positive", "workloads", "models",
    "source", "source_id", "source-id", "source_evidence", "source-evidence",
    "commits", "atomic_part", "atomic-part", "scope", "backend", "architectures",
    "weight_types", "weight-types", "family", "kind", "target", "prerequisites",
    "contract", "correctness", "performance", "activation", "bit-identical",
    "bit_identical", "required-checks", "required_checks", "metrics",
    "target-metric", "target_metric", "title", "rationale", "expected_effect",
    "expected-effect", "acceptance", "hypothesis", "boundary", "source-evidence",
    "metric", "value_pct", "value-pct", "hardware", "workload", "weight-types",
})


def _contract_authority_keys_in(value: object, prefix: str = "") -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            name = str(key)
            location = f"{prefix}.{name}" if prefix else name
            if name in _CONTRACT_AUTHORITY_KEYS:
                found.append(location)
            found.extend(_contract_authority_keys_in(nested, location))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.extend(_contract_authority_keys_in(nested, f"{prefix}[{index}]"))
    return tuple(found)


def parse_validation_toml(
    path: str | Path, *, patch_id: str = "?"
) -> tuple[CheckSpec, ...]:
    """Parse one ``validation.toml`` adapter into CheckSpecs.

    Raises :class:`ConfigurationError` on: schema != 1, unknown
    top-level keys, a missing required [[check]] key, a non-boolean
    ``required``, a duplicate check id, or any Experiment-Contract
    authority field (section 16 strict ownership).
    """
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigurationError(f"{patch_id}: no validation.toml at {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"{patch_id}: {path.name}: invalid TOML: {exc}") from None

    unknown_top = sorted(set(raw) - {"schema", "check"})
    if unknown_top:
        raise ConfigurationError(
            f"{patch_id}: {path.name}: unknown top-level field(s): "
            f"{', '.join(unknown_top)} (expected 'schema' and/or [[check]])"
        )
    if raw.get("schema") != 1:
        raise ConfigurationError(
            f"{patch_id}: {path.name}: unsupported schema {raw.get('schema')!r} (expected 1)"
        )

    specs: list[CheckSpec] = []
    seen: set[str] = set()
    for index, table in enumerate(raw.get("check") or []):
        where = f"{patch_id}: {path.name} [[check]] #{index}"
        if not isinstance(table, dict):
            raise ConfigurationError(f"{where}: entry must be a table")
        missing = [key for key in _CHECK_REQUIRED_KEYS if key not in table]
        if missing:
            raise ConfigurationError(f"{where}: missing required key(s): {', '.join(missing)}")
        check_id = table["id"]
        if not isinstance(check_id, str) or not check_id:
            raise ConfigurationError(f"{where}: 'id' must be a non-empty string")
        if check_id in seen:
            raise ConfigurationError(f"{where}: duplicate check id {check_id!r}")
        seen.add(check_id)
        capability = table["capability"]
        if not isinstance(capability, str) or not capability:
            raise ConfigurationError(f"{where}: 'capability' must be a non-empty string")
        validator = table["validator"]
        if not isinstance(validator, str) or not validator:
            raise ConfigurationError(f"{where}: 'validator' must be a non-empty string")
        required = table.get("required", True)
        if not isinstance(required, bool):
            raise ConfigurationError(f"{where}: 'required' must be a boolean")
        config = {
            key: value
            for key, value in table.items()
            if key not in ("id", "capability", "validator", "required")
        }
        shadowed = _contract_authority_keys_in(config)
        if shadowed:
            raise ConfigurationError(
                f"{where}: key(s) {', '.join(sorted(shadowed))} belong to the Experiment "
                "Contract, not the validation adapter (section 16 strict ownership)"
            )
        specs.append(
            CheckSpec(
                check_id=check_id,
                capability=capability,
                validator=validator,
                required=required,
                config=config,
            )
        )
    return tuple(specs)


# ------------------------------------------------------- custom callable API
# Locked custom-callable API (v1, section 31 + review B2):
#
#     def check(ctx: ValidationContext) -> ValidationResult
#
# Exactly one positional argument, no additional parameters, no *args, no
# **kwargs, synchronous only.

_CUSTOM_CALLABLE_RE = re.compile(r"^(?P<module>.+\.py):(?P<attr>[A-Za-z_][A-Za-z0-9_]*)$")


def resolve_custom_callable(spec: str, *, package_root: str | Path) -> Callable:
    """Load a ``validation/checks.py:fn`` callable with fail-closed rules
    (section 31): path inside the package root, file exists, callable
    exists, locked API shape. Any violation is a
    :class:`ConfigurationError` -- never a partial load."""
    match = _CUSTOM_CALLABLE_RE.match(spec)
    if match is None:
        raise ConfigurationError(
            f"custom callable {spec!r} must be 'path/to/checks.py:function_name'"
        )
    root = Path(package_root).resolve()
    candidate = (root / match.group("module")).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ConfigurationError(
            f"custom callable {spec!r} escapes the package root {root}"
        ) from None
    if not candidate.is_file():
        raise ConfigurationError(f"custom callable {spec!r}: no file at {candidate}")

    import importlib.util

    module_name = f"_bigcherry_validation_{candidate.stem}_{abs(hash(str(candidate))) % 10_000_000}"
    module_spec = importlib.util.spec_from_file_location(module_name, candidate)
    if module_spec is None or module_spec.loader is None:
        raise ConfigurationError(f"custom callable {spec!r}: cannot load {candidate}")
    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
    except Exception as exc:
        raise ConfigurationError(
            f"custom callable {spec!r}: module load failed: {exc}"
        ) from exc

    attr = match.group("attr")
    if not hasattr(module, attr):
        raise ConfigurationError(f"custom callable {spec!r}: no attribute {attr!r}")
    func = getattr(module, attr)
    if not callable(func):
        raise ConfigurationError(f"custom callable {spec!r}: {attr!r} is not callable")
    if inspect.iscoroutinefunction(func):
        raise ConfigurationError(
            f"custom callable {spec!r}: async callables are not supported in v1"
        )
    params = list(inspect.signature(func).parameters.values())
    if (
        len(params) != 1
        or params[0].kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
        or params[0].name != "ctx"
    ):
        raise ConfigurationError(
            f"custom callable {spec!r}: signature must be check(ctx) with exactly "
            "one positional argument, no *args/**kwargs/extra parameters"
        )
    return func


# ------------------------------------------------------------- contract bind


@dataclass(frozen=True)
class ContractBinding:
    """The plan-side projection of a linked Experiment Contract (RS08).

    Carries the contract's identity + the requirements it imposes. It is
    deliberately a PROJECTION: the authoritative fields stay in
    experiment_contract.ExperimentContract (section 16), and nothing here
    duplicates acceptance thresholds or other contract content.
    """

    contract_id: str
    contract_hash: str
    expected_effect: str
    backend: str
    architectures: tuple[str, ...]
    correctness_checks: tuple[str, ...]
    has_controls: bool
    has_boundaries: bool
    required_capabilities: tuple[str, ...]


def bind_contract(contract: experiment_contract.ExperimentContract) -> ContractBinding:
    """Derive a ContractBinding (and the capabilities it requires) from a
    parsed Experiment Contract (sections 16, 18).

    Requirement derivation:
      - expected_effect includes performance -> 'performance' AND
        'activation' (a runtime performance claim requires causal
        attribution/activation, section 18);
      - non-empty correctness.required_checks -> 'correctness';
      - controls or boundary present -> 'controls'.
    """
    capabilities: list[str] = []
    performance = contract.hypothesis.expected_effect in ("performance", "both")
    if performance:
        capabilities.append("performance")
        capabilities.append("activation")
    if contract.correctness.required_checks:
        capabilities.append("correctness")
    if contract.controls.models or contract.controls.workloads or contract.boundary.dimensions:
        capabilities.append("controls")
    # Acceptance thresholds are performance claims even when the hypothesis'
    # expected_effect is otherwise correctness-only (RS08).
    if (
        contract.acceptance.target_kernel_gain_pct is not None
        or contract.acceptance.end_to_end_gain_pct is not None
    ) and "performance" not in capabilities:
        capabilities.append("performance")
        if "activation" not in capabilities:
            capabilities.append("activation")
    return ContractBinding(
        contract_id=contract.id,
        contract_hash=contract.contract_hash,
        expected_effect=contract.hypothesis.expected_effect,
        backend=contract.scope.backend,
        architectures=tuple(contract.scope.architectures),
        correctness_checks=tuple(contract.correctness.required_checks),
        has_controls=bool(contract.controls.models or contract.controls.workloads),
        has_boundaries=bool(contract.boundary.dimensions),
        required_capabilities=tuple(capabilities),
    )


def check_contract_compatibility(
    binding: ContractBinding,
    *,
    patch_backend: str | None,
    patch_architectures: tuple[str, ...] | None = None,
) -> None:
    """RS08 compatibility verification: the contract's scope must be
    compatible with the patch metadata.

    - backend contradiction (both declared and different) is a
      ConfigurationError; a patch that declares no backend is vacuously
      compatible (v1 flat patches carry no backend metadata);
    - hardware/architecture contradiction: when the patch declares
      validation architectures, the contract's declared architectures must
      not exclude them. Undeclared on either side is vacuously compatible.
    """
    if patch_backend and binding.backend and binding.backend != patch_backend:
        raise ConfigurationError(
            f"contract {binding.contract_id!r} backend {binding.backend!r} contradicts "
            f"patch backend {patch_backend!r}"
        )
    if patch_architectures and binding.architectures:
        declared = set(binding.architectures)
        missing = sorted(set(patch_architectures) - declared)
        if missing:
            raise ConfigurationError(
                f"contract {binding.contract_id!r} architectures {sorted(declared)} "
                f"contradict patch architectures {sorted(set(patch_architectures))} "
                f"(missing: {', '.join(missing)})"
            )


# ------------------------------------------------- descriptor -> plan (RS08)


def load_contract_for_descriptor(
    descriptor: patch_registry.PatchDescriptor,
    *,
    contracts_path: str | Path | None = None,
) -> experiment_contract.ExperimentContract | None:
    """Resolve the descriptor's linked Experiment Contract through the
    existing registry (RS08). A reference to a non-existent contract is a
    ConfigurationError (the registry already fails discovery the same
    way; this keeps the orchestrator self-contained for single-patch use).
    """
    if descriptor.experiment_contract is None:
        return None
    from ..core import paths

    path = Path(contracts_path) if contracts_path is not None else paths.EXPERIMENT_CONTRACTS
    registry = experiment_contract.load_contracts(path)
    try:
        return registry[descriptor.experiment_contract]
    except KeyError:
        raise ConfigurationError(
            f"{descriptor.patch_id}: experiment contract "
            f"{descriptor.experiment_contract!r} not found in {path.name}"
        ) from None


def build_plan_for_patch(
    descriptor: patch_registry.PatchDescriptor,
    *,
    root: str | Path | None = None,
    contracts_path: str | Path | None = None,
) -> ValidationPlan | None:
    """Build the aggregated validation plan for one descriptor (RS08):
    universal requirements + linked Experiment Contract requirements +
    the patch's validation.toml adapter.

    A plan exists only when the patch is UNDER VALIDATION: it links a
    contract or ships a validation.toml. A legacy flat patch with neither
    has no plan (None) -- the plan-based campaign (RS10) simply does not
    cover it; that is not a plan-build failure.

    When a plan exists, the universal requirements apply and every
    contract-required capability must have a producer (ConfigurationError
    otherwise). The adapter may add requirements' producers, never remove
    contract/framework requirements (section 18).
    """
    from ..core import paths

    contract = load_contract_for_descriptor(descriptor, contracts_path=contracts_path)
    binding = bind_contract(contract) if contract is not None else None
    if binding is not None:
        check_contract_compatibility(
            binding,
            patch_backend=descriptor.backend,
            patch_architectures=tuple(descriptor.validation_architectures),
        )

    if descriptor.validation_path is None:
        if contract is None:
            return None
        # A contract demands evidence but the adapter is missing: no
        # producer can exist -> fail closed, not skip.
        raise ConfigurationError(
            f"{descriptor.patch_id}: experiment contract "
            f"{descriptor.experiment_contract!r} requires validation evidence but "
            "no validation.toml adapter exists"
        )

    resolved_root = Path(root) if root is not None else paths.PATCHES
    checks = parse_validation_toml(
        resolved_root / descriptor.validation_path, patch_id=descriptor.patch_id
    )
    return build_validation_plan(descriptor.patch_id, checks, binding=binding)


# -------------------------------------------------------------- requirement
# Universal BigCherry requirements (section 18, source 1): every patch,
# regardless of kind or contract, must at minimum prove it APPLIES and that
# its tree BUILDS. A validated patch that cannot be applied and built is
# definitionally not validated.
UNIVERSAL_REQUIREMENTS: tuple[str, ...] = ("apply", "build")


@dataclass(frozen=True)
class ValidationPlan:
    """The aggregated validation plan for one patch (section 18):
    universal requirements + linked contract requirements + the adapter's
    checks. ``required_capabilities`` is the demand side; ``checks`` is the
    supply side. The adapter may add, never remove."""

    patch_id: str
    checks: tuple[CheckSpec, ...]
    universal_capabilities: tuple[str, ...]
    contract: ContractBinding | None
    required_capabilities: tuple[str, ...]
    framework_version: str = VALIDATION_FRAMEWORK_VERSION

    def checks_for(self, capability: str) -> tuple[CheckSpec, ...]:
        return tuple(c for c in self.checks if c.capability == capability)

    def spec_for(self, check_id: str) -> CheckSpec:
        for check in self.checks:
            if check.check_id == check_id:
                return check
        raise ConfigurationError(f"unknown check id {check_id!r}")

    def required_checks(self) -> tuple[CheckSpec, ...]:
        return tuple(c for c in self.checks if c.required)


def _produces(capability: str, spec: CheckSpec) -> bool:
    # A check is a producer only for the capability it declares. The validator
    # vocabulary describes the evidence mechanic; it must not silently make a
    # performance check satisfy a controls requirement (RS08).
    if spec.capability != capability:
        return False
    if _capability_is_custom(spec.validator):
        # A custom check produces exactly its declared capability; the
        # framework verifies the returned result matches (section 31).
        return True
    return spec.validator in _CAPABILITY_PRODUCERS.get(capability, frozenset())


def build_validation_plan(
    patch_id: str,
    checks: tuple[CheckSpec, ...] | list[CheckSpec],
    *,
    binding: ContractBinding | None = None,
    universal: tuple[str, ...] = UNIVERSAL_REQUIREMENTS,
    framework_version: str = VALIDATION_FRAMEWORK_VERSION,
) -> ValidationPlan:
    """Aggregate the three requirement sources (section 18).

    Fails with :class:`ConfigurationError` when any required capability
    (universal or contract) has no capable producer among the checks.
    """
    required = tuple(universal)
    if binding is not None:
        for capability in binding.required_capabilities:
            if capability not in required:
                required = required + (capability,)

    missing = [
        capability for capability in required
        if not any(spec.required and _produces(capability, spec) for spec in checks)
    ]
    if missing:
        raise ConfigurationError(
            f"{patch_id}: required capabilities with no producer: "
            f"{', '.join(missing)} "
            "(section 18: configuration error, not skip)"
        )

    return ValidationPlan(
        patch_id=patch_id,
        checks=tuple(checks),
        universal_capabilities=tuple(universal),
        contract=binding,
        required_capabilities=required,
        framework_version=framework_version,
    )


# --------------------------------------------------------------- validation
# Validator entry point: (CheckSpec, ValidationContext) -> ValidationResult.
Validator = Callable[
    [CheckSpec, "ValidationContext"], ValidationResult
]

# The built-in registry is populated incrementally in RS09 (one
# validator per step, each independently tested). Names listed in
# BUILTIN_VALIDATORS but absent here are "not yet migrated" -- a check
# referencing one produces a structured ERROR, never a silent skip.
_BUILTIN_REGISTRY: dict[str, Validator] = {}
# Public consumers may inspect the registry but cannot mutate dispatch.
BUILTIN_REGISTRY = MappingProxyType(_BUILTIN_REGISTRY)


def register_builtin(name: str, validator: Validator) -> None:
    """Register a built-in validator exactly once during framework setup.

    The closed registry is not an extension hook: silently replacing an
    existing validator would let package-local code hijack another capability.
    """
    if name not in BUILTIN_VALIDATORS:
        raise ConfigurationError(
            f"unknown built-in validator {name!r} (closed set: "
            f"{', '.join(BUILTIN_VALIDATORS)})"
        )
    if name in _BUILTIN_REGISTRY:
        raise ConfigurationError(f"built-in validator {name!r} is already registered")
    _BUILTIN_REGISTRY[name] = validator


# ------------------------------------------------------------- ValidationContext


@dataclass(frozen=True)
class ValidationContext:
    """Structured context handed to validators / custom checks (section
    32). Custom code receives NO raw global state and NO helper that
    mutates the canonical control/subject source in place: temporary
    instrumentation goes through ``create_source_variant`` (section 33)."""

    descriptor: patch_registry.PatchDescriptor
    base_revision: str
    control_source: Path | None
    subject_source: Path | None
    package_root: Path | None = None  # custom-callable escape boundary
    stock_source: Path | None = None
    control_tree: str | None = None
    subject_tree: str | None = None
    build_identities: dict[str, str] = field(default_factory=dict)
    architecture: str | None = None
    device_identity: str | None = None
    model: str | None = None
    workload: str | None = None
    contract: Any = None  # experiment_contract.ExperimentContract | None
    contract_hash: str | None = None
    run_dir: Path | None = None
    run_binary: Callable[..., Any] | None = None
    register_artifact: Callable[..., ArtifactRef] | None = None
    create_source_variant: Callable[..., Path] | None = None
    apply_evidence: dict[str, Any] = field(default_factory=dict)
    build_evidence: dict[str, Any] = field(default_factory=dict)
    trace_evidence: dict[str, Any] = field(default_factory=dict)
    correctness_evidence: dict[str, Any] = field(default_factory=dict)
    configuration_evidence: dict[str, Any] = field(default_factory=dict)
    smoke_evidence: dict[str, Any] = field(default_factory=dict)
    architecture_evidence: dict[str, Any] = field(default_factory=dict)
    performance_evidence: dict[str, Any] = field(default_factory=dict)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_default_register_artifact(run_dir: Path) -> Callable[..., ArtifactRef]:
    """A register_artifact helper that copies into the run directory's
    artifacts/ subtree and returns a compact ArtifactRef (section 40:
    compact evidence stays tracked, large payloads stay under ignored
    runtime storage)."""
    target_root = run_dir / "artifacts"

    def register(name: str, path: str | Path) -> ArtifactRef:
        path = Path(path)
        digest = _sha256_file(path)
        target_root.mkdir(parents=True, exist_ok=True)
        target = target_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        return ArtifactRef(name=name, path=f"{target.relative_to(run_dir).as_posix()}", sha256=digest)

    return register


def _verified_source_tree(source: Path | None, declared: str | None) -> bool:
    """Recompute a source worktree tree OID instead of trusting caller text."""
    if source is None or not source.is_dir() or not declared:
        return False
    try:
        from .source import git_worktree_tree
        return git_worktree_tree(source) == declared
    except (OSError, RuntimeError, ValueError):
        return False


def _artifact_is_bound(artifact: object, run_dir: Path | None) -> bool:
    if not isinstance(artifact, dict) or run_dir is None:
        return False
    path = artifact.get("path")
    digest = artifact.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        return False
    target = (run_dir / path).resolve()
    try:
        target.relative_to(run_dir.resolve())
    except ValueError:
        return False
    return target.is_file() and _sha256_file(target) == digest


def _bound_artifact_path(artifact: object, run_dir: Path | None) -> Path | None:
    if not isinstance(artifact, dict) or run_dir is None:
        return None
    path = artifact.get("path")
    if not isinstance(path, str):
        return None
    target = (run_dir / path).resolve()
    try:
        target.relative_to(run_dir.resolve())
    except ValueError:
        return None
    return target if target.is_file() else None


def _read_bound_json(artifact: object, run_dir: Path | None) -> dict[str, Any] | None:
    target = _bound_artifact_path(artifact, run_dir)
    if target is None or not _artifact_is_bound(artifact, run_dir):
        return None
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    return value if isinstance(value, dict) else None


def _builtin_apply(spec: CheckSpec, ctx: ValidationContext) -> ValidationResult:
    """Validate evidence-bound apply/tree/idempotency artifacts.

    Materialization itself belongs to the source-isolation/campaign layer;
    this validator recomputes the source tree and binds the claimed evidence
    artifact to the run directory before PASS is possible.
    """
    if ctx.control_source is None or ctx.subject_source is None:
        return ValidationResult(
            check_id=spec.check_id, capability=spec.capability, status=BLOCKED,
            summary="control and subject sources are required for apply evidence",
        )
    if not ctx.control_source.is_dir() or not ctx.subject_source.is_dir():
        return ValidationResult(
            check_id=spec.check_id, capability=spec.capability, status=FAIL,
            summary="control or subject source directory does not exist",
        )
    missing = tuple(
        role for role, source, declared_tree in (
            ("control", ctx.control_source, ctx.control_tree),
            ("subject", ctx.subject_source, ctx.subject_tree),
        )
        if not isinstance(ctx.apply_evidence.get(role), dict)
        or not ctx.apply_evidence[role].get("verified")
        or not ctx.apply_evidence[role].get("idempotent")
        or not _verified_source_tree(source, declared_tree)
        or not _artifact_is_bound(ctx.apply_evidence[role].get("artifact"), ctx.run_dir)
    )
    if missing:
        return ValidationResult(
            check_id=spec.check_id, capability=spec.capability, status=BLOCKED,
            summary=f"verified apply evidence is missing for: {', '.join(missing)}",
        )
    return ValidationResult(
        check_id=spec.check_id, capability=spec.capability, status=PASS,
        summary="verified control/subject application, tree, and idempotency evidence is present",
    )


def _evidence_pass(
    spec: CheckSpec, evidence: dict[str, Any], ctx: ValidationContext, label: str,
) -> ValidationResult:
    payload = _read_bound_json(evidence.get("artifact"), ctx.run_dir)
    if payload is None:
        return ValidationResult(
            check_id=spec.check_id, capability=spec.capability, status=BLOCKED,
            summary=f"verified {label} artifact is required",
        )
    if payload.get("passed") is not True:
        return ValidationResult(
            check_id=spec.check_id, capability=spec.capability, status=FAIL,
            summary=f"{label} evidence failed",
        )
    # VA08 round 3 (req_021c2eb498e04bc0): a PASS here already proves the
    # evidence artifact is bound (verified by _read_bound_json() above) --
    # surface that same artifact in ValidationResult.artifacts so it is
    # actually visible in the serialized record's check_results, not
    # silently dropped. Without this, every real benchmark/autotune-
    # campaign PASS through this shared helper serialized artifacts=()
    # despite genuinely having bound evidence.
    artifact = evidence.get("artifact")
    artifacts = (
        (ArtifactRef(name=label, path=str(artifact["path"]), sha256=str(artifact["sha256"])),)
        if isinstance(artifact, dict) else ()
    )
    return ValidationResult(
        check_id=spec.check_id, capability=spec.capability, status=PASS,
        summary=f"{label} evidence is verified and bound", artifacts=artifacts,
    )


def _builtin_compile_option(spec: CheckSpec, ctx: ValidationContext) -> ValidationResult:
    expected = spec.config.get("options")
    evidence = ctx.configuration_evidence
    payload = _read_bound_json(evidence.get("artifact"), ctx.run_dir) if isinstance(evidence, dict) else None
    if expected is not None and (payload is None or payload.get("options") != expected):
        return _error_result(spec, "compile-option artifact does not match configured options")
    return _evidence_pass(spec, evidence, ctx, "compile-option")


def _builtin_runtime_smoke(spec: CheckSpec, ctx: ValidationContext) -> ValidationResult:
    evidence = ctx.smoke_evidence
    payload = _read_bound_json(evidence.get("artifact"), ctx.run_dir) if isinstance(evidence, dict) else None
    if payload is None or payload.get("exit_code") != 0:
        if payload is None or payload.get("exit_code") is None:
            return ValidationResult(
                check_id=spec.check_id, capability=spec.capability, status=BLOCKED,
                summary="runtime-smoke exit evidence is missing",
            )
        return ValidationResult(
            check_id=spec.check_id, capability=spec.capability, status=FAIL,
            summary="runtime-smoke exited unsuccessfully",
        )
    return _evidence_pass(spec, evidence, ctx, "runtime-smoke")


def _builtin_architecture(spec: CheckSpec, ctx: ValidationContext) -> ValidationResult:
    evidence = ctx.architecture_evidence
    payload = _read_bound_json(evidence.get("artifact"), ctx.run_dir) if isinstance(evidence, dict) else None
    if ctx.architecture and (payload is None or payload.get("architecture") != ctx.architecture):
        return _error_result(spec, "architecture artifact does not match context")
    return _evidence_pass(spec, evidence, ctx, "architecture")


def _builtin_benchmark(spec: CheckSpec, ctx: ValidationContext) -> ValidationResult:
    evidence = ctx.performance_evidence
    payload = _read_bound_json(evidence.get("artifact"), ctx.run_dir) if isinstance(evidence, dict) else None
    if payload is None or not isinstance(payload.get("metrics"), dict) or not payload["metrics"]:
        return _error_result(spec, "benchmark artifact requires non-empty metrics")
    return _evidence_pass(spec, evidence, ctx, "benchmark")


def _builtin_autotune_campaign(spec: CheckSpec, ctx: ValidationContext) -> ValidationResult:
    evidence = ctx.performance_evidence
    payload = _read_bound_json(evidence.get("artifact"), ctx.run_dir) if isinstance(evidence, dict) else None
    if payload is None or not payload.get("campaign_id"):
        return _error_result(spec, "autotune-campaign artifact requires campaign_id")
    return _evidence_pass(spec, evidence, ctx, "autotune-campaign")


def _builtin_backend_ops(spec: CheckSpec, ctx: ValidationContext) -> ValidationResult:
    """Consume evidence-bound backend operation correctness results."""
    configured = spec.config.get("ops")
    if not isinstance(configured, (list, tuple)) or not configured or not all(
        isinstance(item, str) and item for item in configured
    ):
        return _error_result(spec, "backend-ops requires non-empty string ops")
    evidence = ctx.correctness_evidence
    payload = _read_bound_json(evidence.get("artifact"), ctx.run_dir) if isinstance(evidence, dict) else None
    if payload is None:
        return ValidationResult(
            check_id=spec.check_id, capability=spec.capability, status=BLOCKED,
            summary="verified backend-ops artifact is required",
        )
    observed = payload.get("ops")
    if tuple(observed or ()) != tuple(configured):
        return _error_result(spec, "backend-ops artifact operation set does not match config")
    if payload.get("passed") is not True:
        return ValidationResult(
            check_id=spec.check_id, capability=spec.capability, status=FAIL,
            summary="backend operation correctness evidence failed",
        )
    return ValidationResult(
        check_id=spec.check_id, capability=spec.capability, status=PASS,
        summary="configured backend operations passed with bound evidence",
    )


def _builtin_trace_marker(spec: CheckSpec, ctx: ValidationContext) -> ValidationResult:
    """Consume generic positive/negative marker evidence.

    Patch-specific probe commands stay in the campaign/orchestrator. The
    framework receives only verified observations and bound log artifacts.
    """
    marker = spec.config.get("marker-regex")
    if not isinstance(marker, str) or not marker:
        return _error_result(spec, "trace-marker requires non-empty marker-regex")
    positive = ctx.trace_evidence.get("positive")
    negative = ctx.trace_evidence.get("negative")
    if not isinstance(positive, dict) or not isinstance(negative, dict):
        return ValidationResult(
            check_id=spec.check_id, capability=spec.capability, status=BLOCKED,
            summary="positive and negative trace observations are required",
        )
    artifacts = (positive.get("artifact"), negative.get("artifact"))
    if not all(_artifact_is_bound(artifact, ctx.run_dir) for artifact in artifacts):
        return ValidationResult(
            check_id=spec.check_id, capability=spec.capability, status=BLOCKED,
            summary="trace logs are not bound to the validation run",
        )
    positive_path = _bound_artifact_path(positive.get("artifact"), ctx.run_dir)
    negative_path = _bound_artifact_path(negative.get("artifact"), ctx.run_dir)
    if positive_path is None or negative_path is None:
        return ValidationResult(
            check_id=spec.check_id, capability=spec.capability, status=BLOCKED,
            summary="trace log paths are not available",
        )
    try:
        positive_hit = re.search(marker, positive_path.read_text(encoding="utf-8")) is not None
        negative_hit = re.search(marker, negative_path.read_text(encoding="utf-8")) is not None
    except (OSError, UnicodeError, re.error) as exc:
        return _error_result(spec, f"trace log verification failed: {exc}")
    if positive.get("marker_regex") != marker or negative.get("marker_regex") != marker:
        return _error_result(spec, "trace evidence marker-regex does not match check config")
    if positive_hit is not True:
        return ValidationResult(
            check_id=spec.check_id, capability=spec.capability, status=FAIL,
            summary="positive trace probe did not observe the configured marker",
        )
    if negative_hit is not False:
        return ValidationResult(
            check_id=spec.check_id, capability=spec.capability, status=FAIL,
            summary="negative-control trace probe observed the configured marker",
        )
    return ValidationResult(
        check_id=spec.check_id, capability=spec.capability, status=PASS,
        summary="positive marker hit and negative-control non-hit are verified",
        artifacts=tuple(artifact for artifact in artifacts if isinstance(artifact, ArtifactRef)),
    )


def _builtin_build(spec: CheckSpec, ctx: ValidationContext) -> ValidationResult:
    """Validate that the campaign supplied build identities for both sides."""
    required = ("control", "subject")
    missing = tuple(
        role for role, source, declared_tree in (
            ("control", ctx.control_source, ctx.control_tree),
            ("subject", ctx.subject_source, ctx.subject_tree),
        )
        if not isinstance(ctx.build_identities.get(role), str)
        or not ctx.build_identities[role]
        or not isinstance(ctx.build_evidence.get(role), dict)
        or ctx.build_evidence[role].get("build_id") != ctx.build_identities[role]
        or not _verified_source_tree(source, declared_tree)
        or ctx.build_evidence[role].get("source_tree") != declared_tree
        or not ctx.build_evidence[role].get("architecture")
        or not ctx.build_evidence[role].get("options")
        or not _artifact_is_bound(ctx.build_evidence[role].get("compile_commands"), ctx.run_dir)
        or not _artifact_is_bound(ctx.build_evidence[role].get("runtime_bundle"), ctx.run_dir)
    )
    if missing:
        return ValidationResult(
            check_id=spec.check_id, capability=spec.capability, status=BLOCKED,
            summary=f"verified build evidence is missing for: {', '.join(missing)}",
        )
    return ValidationResult(
        check_id=spec.check_id, capability=spec.capability, status=PASS,
        summary="verified control and subject build/source/runtime evidence is present",
    )


register_builtin("apply", _builtin_apply)
register_builtin("build", _builtin_build)
register_builtin("trace-marker", _builtin_trace_marker)
register_builtin("backend-ops", _builtin_backend_ops)
register_builtin("compile-option", _builtin_compile_option)
register_builtin("runtime-smoke", _builtin_runtime_smoke)
register_builtin("architecture", _builtin_architecture)
register_builtin("benchmark", _builtin_benchmark)
register_builtin("autotune-campaign", _builtin_autotune_campaign)


# ------------------------------------------------------------------ verdict


@dataclass(frozen=True)
class Verdict:
    """Central eligibility computation (section 20: only the aggregator
    computes eligibility). ``eligible`` is True iff every required check
    has status ``pass``."""

    eligible: bool
    results: tuple[ValidationResult, ...]
    reasons: tuple[str, ...]
    blocked: bool
    errors: tuple[str, ...]


def _error_result(spec: CheckSpec, summary: str) -> ValidationResult:
    return ValidationResult(
        check_id=spec.check_id,
        capability=spec.capability,
        status=ERROR,
        summary=summary,
    )


def evaluate_check(spec: CheckSpec, ctx: ValidationContext) -> ValidationResult:
    """Dispatch one check through the built-in registry / custom loader.

    An unknown validator name is never silently dropped: the check
    returns a structured ERROR (runbook RS07 negative test), which makes
    the patch not eligible via :func:`compute_verdict`.
    """
    if _capability_is_custom(spec.validator):
        callable_spec = spec.config.get("callable")
        if not isinstance(callable_spec, str) or not callable_spec:
            return _error_result(
                spec, "custom check missing 'callable' (expected 'path:fn')"
            )
        if ctx.package_root is None:
            return _error_result(
                spec, "custom check cannot be loaded: no package root in context"
            )
        func = resolve_custom_callable(callable_spec, package_root=ctx.package_root)
        try:
            result = func(ctx)
        except Exception as exc:
            return _error_result(spec, f"custom check raised: {exc}")
        if not isinstance(result, ValidationResult):
            return _error_result(
                spec, f"custom check returned {type(result).__name__}, not ValidationResult"
            )
        if (result.check_id, result.capability) != (spec.check_id, spec.capability):
            return _error_result(
                spec,
                "custom check returned a result for a different check "
                f"({result.check_id!r}/{result.capability!r}); the framework "
                "does not trust the re-labelling",
            )
        return result
    validator = _BUILTIN_REGISTRY.get(spec.validator)
    if validator is None:
        return _error_result(
            spec,
            f"unknown or un-migrated validator {spec.validator!r} "
            f"(known built-ins: {', '.join(BUILTIN_VALIDATORS)})",
        )
    try:
        return validator(spec, ctx)
    except Exception as exc:
        return _error_result(spec, f"validator {spec.validator!r} raised: {exc}")


def compute_verdict(
    plan: ValidationPlan, results: dict[str, ValidationResult]
) -> Verdict:
    """Aggregate results into the eligibility verdict (section 19).

    - every plan check must have a result; a missing one is a structured
      ERROR (fail-closed);
    - a required check is satisfied only by ``pass``;
    - ``not_applicable`` on a REQUIRED check is a v1 contract violation:
      demoted to ``error`` (it can never satisfy, and pretending it did
      would be worse than an infrastructure error);
    - non-required check errors are reported (``Verdict.errors``) but do
      not by themselves block eligibility.
    """
    all_results: list[ValidationResult] = []
    for spec in plan.checks:
        result = results.get(spec.check_id)
        if result is None:
            result = _error_result(spec, "no result recorded for this check")
        elif (result.check_id, result.capability) != (spec.check_id, spec.capability):
            result = _error_result(
                spec,
                f"recorded result is for {result.check_id!r}/{result.capability!r}, "
                "not this check",
            )
        if result.status == NOT_APPLICABLE and spec.required:
            result = ValidationResult(
                check_id=spec.check_id,
                capability=spec.capability,
                status=ERROR,
                summary=(
                    "not_applicable was returned for a required check, which the "
                    "v1 lock forbids (it never satisfies a required check); "
                    "treating as error"
                ),
                details=result.details,
                artifacts=result.artifacts,
            )
        all_results.append(result)

    by_id = {spec.check_id: next(r for r in all_results if r.check_id == spec.check_id)
             for spec in plan.checks}
    reasons: list[str] = []
    blocked = False
    errors: list[str] = []
    for spec in plan.checks:
        result = by_id[spec.check_id]
        if result.status == ERROR:
            errors.append(spec.check_id)
        if not spec.required:
            continue
        if result.status == PASS:
            continue
        if result.status == BLOCKED:
            blocked = True
            reasons.append(
                f"required check {spec.check_id!r} blocked: {result.summary}"
            )
        elif result.status == ERROR:
            reasons.append(
                f"required check {spec.check_id!r} error: {result.summary}"
            )
        else:  # fail (and any residual non-pass)
            reasons.append(
                f"required check {spec.check_id!r} {result.status}: {result.summary}"
            )

    return Verdict(
        eligible=not reasons,
        results=tuple(all_results),
        reasons=tuple(reasons),
        blocked=blocked,
        errors=tuple(errors),
    )


# ------------------------------------------------- plan canonical identity
# The plan's own canonical serialization is part of the validation
# identity chain (RS11 evidence: the plan is reproducible from
# validation_digest + framework version). Kept here so any layer can
# re-derive it without re-parsing.


def plan_canonical_payload(plan: ValidationPlan) -> dict[str, Any]:
    return {
        "schema": "bigcherry-validation-plan/v1",
        "framework_version": plan.framework_version,
        "patch_id": plan.patch_id,
        "universal": list(plan.universal_capabilities),
        "contract": (
            {
                "id": plan.contract.contract_id,
                "hash": plan.contract.contract_hash,
                "required": list(plan.contract.required_capabilities),
            }
            if plan.contract is not None
            else None
        ),
        "required_capabilities": list(plan.required_capabilities),
        "checks": [
            {
                "id": spec.check_id,
                "capability": spec.capability,
                "validator": spec.validator,
                "required": spec.required,
            }
            for spec in plan.checks
        ],
    }


def plan_digest(plan: ValidationPlan) -> str:
    encoded = json.dumps(
        plan_canonical_payload(plan), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
