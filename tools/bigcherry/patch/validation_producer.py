"""PA36 foundation: the generic validation-producer protocol.

End-state principle (GPT design, req_0fb717b910e145b2, 2026-09-13):
``validation_campaign.py`` knows how to EXECUTE a selected validation
producer, but never knows WHICH patch/RD it is executing. Patch identity,
producer entrypoint, policy, declared inputs, and allowed artifact names
are all owned by ``patches/<id>/validation/``; reusable build/run/evidence
machinery (``build_tree()``, ``capture_completed_build_evidence()``,
``run_paired_lane()``, ``patch_validation_evidence.make_record()``, etc.)
stays shared and lives in ``validation_campaign.py``/``evidence.py``.

This module is infrastructure only (PA36 step 0) -- no patch has migrated
onto it yet. A patch migrates by adding ``patches/<id>/validation/
producer.toml`` + ``producer.py`` and deleting its old ``run_rdXX_*``
function/CLI branch/central artifact names from ``validation_campaign.py``
in the SAME commit (see PA36's atomic-migration checklist; no compatibility
layer, no old-flag aliases).

Dependency direction is load-bearing: this module (and every
``patches/<id>/validation/producer.py``) must NEVER import
``validation_campaign.py`` -- that would recreate the exact coupling this
refactor exists to remove. ``validation_campaign.py`` imports THIS module,
not the other way around.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, cast

import tomllib

from bigcherry.experiment import contract as experiment_contract
from bigcherry.experiment.attestation import ExecutionIdentity

from .validation import (
    ArtifactRef,
    CheckSpec,
    ValidationContext,
    ValidationPlan,
    ValidationResult,
)

# Reuse the project's real build-identity shape (CompletedBuildEvidence.
# campaign_identity()'s return type) rather than inventing a parallel
# concrete type -- it is already exactly Mapping[str, Mapping[str, object]]
# (a plain dict of plain dicts), so this alias just names that shape.
JsonObject = Mapping[str, object]
BuildIdentityMap = Mapping[str, Mapping[str, object]]

_ALLOWED_POLICY_VALUES: dict[str, frozenset[str]] = {
    "trace_probe": frozenset({"run", "skip"}),
    "standard_campaign": frozenset({"run", "skip"}),
    "correctness_evidence_cli": frozenset({"allow", "forbid"}),
    "performance_benchmark_cli": frozenset({"allow", "forbid"}),
}

# Artifact names must be plain, unique basenames -- never a path component
# that could escape the campaign's own artifacts/ directory (the exact
# same discipline evidence.py::_artifact_refs()'s hardcoded allowlist
# already enforces implicitly by being a fixed tuple; this makes it an
# explicit, checked rule for the now-decentralized per-producer names).
_ARTIFACT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.[A-Za-z0-9]+$")


class ValidationProducerError(ValueError):
    """Raised for a malformed producer.toml, an unresolvable producer, or
    a producer result that violates its own declared contract."""


@dataclass(frozen=True)
class FatTargetPlan:
    """The build-once-fat-multiarch rule (STANDARDIZED_PATCH_VALIDATION_
    CRITERIA.md), made a concrete value instead of producer-author
    discipline. ``cmake_value`` is the exact ``AMDGPU_TARGETS`` string
    (semicolon-joined) every control/subject build in a multi-device
    producer run must share -- constructing this from ``targets`` is the
    only sanctioned way to get that string, so it can never accidentally
    drift between two builds of the same producer run."""

    targets: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValidationProducerError("FatTargetPlan requires at least one target")
        if len(set(self.targets)) != len(self.targets):
            raise ValidationProducerError(
                f"FatTargetPlan has duplicate targets: {self.targets!r}"
            )

    @property
    def cmake_value(self) -> str:
        return ";".join(self.targets)


@dataclass(frozen=True)
class ProducerInputSpec:
    """One declared ``[producer.<id>.input.<name>]`` entry from
    producer.toml. ``type`` is presently informational (the producer
    itself is responsible for interpreting the raw string from
    ``--producer-input name=value``); this exists so ``resolve_producer()``
    can fail closed on an undeclared or missing-required input before the
    producer ever runs, rather than the producer discovering a typo'd
    ``--producer-input`` key at an arbitrary point mid-run."""

    name: str
    type: str
    required: bool


@dataclass(frozen=True)
class ProducerSpec:
    """The plan-side projection of one ``[producer.<id>]`` entry --
    resolved, validated, and immutable. Mirrors ``patch_validation.
    ContractBinding``'s role for Experiment Contracts: authoritative
    content stays in producer.toml, this is a checked projection of it."""

    patch_id: str
    producer_id: str
    entrypoint: Path
    callable_name: str
    trace_probe: str
    standard_campaign: str
    correctness_evidence_cli: str
    performance_benchmark_cli: str
    artifact_names: frozenset[str]
    inputs: Mapping[str, ProducerInputSpec] = field(default_factory=dict)


@dataclass(frozen=True)
class ProducerBuildPair:
    """The build-once-fat-multiarch control/subject build pair
    ``ProducerRuntime.build_pair()`` hands back (PA36-F step 1). One
    control build and one subject build, each built exactly once
    regardless of how many devices/architectures the producer will run
    against -- ``CampaignProducerRuntime.build_pair()`` (validation_
    campaign.py) is the one authority enforcing that rule; this is just
    its immutable result shape."""

    base_revision: str
    control_source: Path
    subject_source: Path
    control_composition: tuple[tuple[str, str], ...]
    subject_composition: tuple[tuple[str, str], ...]
    control_bin: Path
    subject_bin: Path
    validation_build_identities: BuildIdentityMap


@dataclass(frozen=True)
class ProducerDeviceContext:
    """One real, verified physical device a producer may run against
    (PA36-F step 1). ``env_overrides``/``env_unset`` are the exact HIP-
    only selector env a producer's subprocess must be launched with --
    never ambient HIP_VISIBLE_DEVICES, never a ROCR_VISIBLE_DEVICES
    double-filter (PNRO17)."""

    architecture: str
    device_index: int
    execution_identity: ExecutionIdentity
    env_overrides: Mapping[str, str]
    env_unset: tuple[str, ...]
    # The verified physical PCI locator from config/environment.toml
    # (PA36 RD13/1206 migration, GPT req_760c0fe82d7b4609 BLOCKER).
    # ``execution_identity`` deliberately omits locators because the
    # llama-bench/llama-perplexity attestation banner carries no PCI
    # locator; but the llama-server attestation channel derives the
    # server architecture ONLY via a locator->arch mapping, so a
    # producer that runs llama-server (RD13) needs this to construct
    # a locator-bearing ExecutionIdentity. Set by device_contexts()
    # (guaranteed non-None there); None only for test/mocked devices.
    locator: str | None = None


@dataclass(frozen=True)
class ProducerPairedBenchmarkOutcome:
    """The public shape of ``validation_campaign.PairedBenchmarkOutcome``,
    exposed to patch-local producers (PA36-F step 1) without exposing the
    campaign module itself. The real implementation stays owned by
    ``validation_campaign.run_paired_llama_benchmark()``; this is only the
    producer-facing result type."""

    runs: Mapping[str, object]
    commands: Mapping[str, Mapping[str, tuple[str, ...]]]
    raw_logs: tuple[JsonObject, ...]


class ProducerRuntime(Protocol):
    """The campaign-owned runtime a producer receives via
    ``ProducerContext.runtime`` (PA36-F step 1). Patch-local producer
    modules must never import ``validation_campaign.py`` directly --
    this Protocol is the sanctioned seam instead; ``validation_campaign.
    CampaignProducerRuntime`` is the one concrete implementation."""

    def build_pair(
        self,
        *,
        targets: tuple[str, ...],
        primary_target: str,
        common_extra_patches: tuple[str, ...] = (),
        # Patches that only form a valid unit together with the focal patch
        # (e.g. a guard the focal needs to be safe) go into the SUBJECT arm
        # only; the record's subject composition names them, so the claim
        # is "focal + companions vs baseline", never "focal alone".
        subject_companion_patches: tuple[str, ...] = (),
        baseline_source: str = "bigcherry",
        control_extra_cmake_args: tuple[str, ...] = (),
        subject_extra_cmake_args: tuple[str, ...] = (),
        require_parity: bool = False,
    ) -> ProducerBuildPair: ...

    # require_parity (GPT review req_7a72896b609a48b5 BLOCKER #3): the
    # legacy RD04 producer asserted real build parity on its own pair
    # after capturing both builds; producers whose measurements depend
    # on build parity (RD04's PPL comparison) pass True and the concrete
    # runtime calls assert_validation_subject_parity() before returning.

    def device_contexts(
        self,
        *,
        device_map: Mapping[str, tuple[int, ...]],
    ) -> tuple[ProducerDeviceContext, ...]: ...

    def write_artifact(
        self,
        *,
        name: str,
        payload: JsonObject,
    ) -> ArtifactRef: ...

    def write_text_artifact(
        self,
        *,
        name: str,
        text: str,
    ) -> ArtifactRef: ...

    def run_paired_llama_benchmark(
        self,
        *,
        control_binary: Path,
        subject_binary: Path,
        model: Path,
        workloads: tuple[str, ...] = ("decode", "prefill"),
        patch_args: tuple[str, ...] = (),
        runtime_args: tuple[str, ...] = (),
        pairs: int = 3,
        log_context: str,
        device: ProducerDeviceContext | None = None,
        env_overrides: Mapping[str, str] | None = None,
        env_unset: tuple[str, ...] = (),
    ) -> ProducerPairedBenchmarkOutcome: ...

    def build_materialized_pair(
        self,
        *,
        control_source: Path,
        subject_source: Path,
        targets: tuple[str, ...],
        primary_target: str,
    ) -> ProducerBuildPair: ...

    def run_trace_probe(
        self,
        *,
        binary: Path,
        model: Path,
        device: ProducerDeviceContext,
        bench_prompt: int,
        bench_gen: int,
        log_context: str,
        disable_fusion: bool = False,
    ) -> str: ...


@dataclass(frozen=True)
class ProducerContext:
    """What a producer receives. Deliberately NOT an argparse.Namespace --
    patch-local producer code must never become coupled to CLI structure,
    so every field here is a plain, already-resolved value the generic
    dispatcher computed."""

    repo_root: Path
    patch_dir: Path
    workdir: Path
    campaign_id: str
    base_revision: str
    hip_path: Path
    fat_targets: FatTargetPlan
    model: Path | None
    corpus: Path | None
    build_env: Mapping[str, str]
    inputs: Mapping[str, str]
    # Control/subject identities the generic campaign path already built,
    # when available -- a producer that materializes and builds its OWN
    # isolated control/subject worktrees (RD12's shape) replaces these in
    # its ProducerResult rather than reusing them.
    validation_build_identities: BuildIdentityMap
    patch_id: str
    device_map: Mapping[str, tuple[int, ...]]
    runtime: ProducerRuntime
    # Standard-campaign scaffold binaries the generic path already built
    # (role -> target -> binary path: {"control": {"llama-bench": ...,
    # "llama-server": ...}, "subject": {...}}), populated only when
    # standard_campaign="run". A producer that consumes scaffold-derived
    # binaries (RD04's benchmark reuses the scaffold parity llama-bench
    # pair instead of building a second pair) reads them here; {} for
    # standard_campaign="skip" (also the default for contexts built
    # outside a standard campaign).
    validation_binaries: Mapping[str, Mapping[str, Path]] = field(default_factory=dict)


@dataclass(frozen=True)
class ProducerCheckResult:
    """One producer-supplied, typed per-check result (PA36-F step 2).
    Replaces the opaque ``named_correctness_results: Mapping[str, object]``
    /``check_results: Mapping[str, JsonObject]`` pair: ``validation_result``
    is the real, already-typed ``ValidationResult`` (its own ``.artifacts``
    is the authoritative bound-artifact set -- there is no second artifact
    field here), ``contract_ids`` states which bound contract(s) this
    check's evidence is scoped to (mirroring ``CheckSpec.contract_ids``/
    ``ValidationContext.contract_ids_for_check()``), and ``disposition``
    -- when present -- is the raw ``{"passed": bool, ...}`` payload that
    becomes one entry of ``make_record()``'s ``contract_verdicts``."""

    check_id: str
    contract_ids: tuple[str, ...]
    validation_result: ValidationResult
    disposition: JsonObject | None = None


@dataclass(frozen=True)
class ProducerResult:
    """What a producer returns. Reconciled against the real current
    ``patch_validation_evidence.make_record()`` call (validation_campaign.py)
    -- every field here maps onto a real make_record() parameter or an
    intermediate value the generic dispatcher needs before calling it.

    No generic ``record_kwargs: dict`` escape hatch -- that would let a
    producer smuggle untyped patch-specific fields through and recreate
    the exact monolith coupling this refactor removes. If a genuinely new
    generic concept is needed, it gets a named field here, not a bag.
    """

    correctness: JsonObject | None
    validation_build_identities: BuildIdentityMap
    activation_evidence: object | None
    performance_evidence: JsonObject | None
    trace_evidence: JsonObject | None
    check_results: tuple[ProducerCheckResult, ...]
    lane_effects: tuple[JsonObject, ...]
    emitted_artifacts: frozenset[str]
    # GPT review req_7a72896b609a48b5 BLOCKER #2: the typed named
    # contract-correctness results a producer measured. The shared
    # dispatcher converts these into the real
    # evaluate_correctness_gate() result and persists it as
    # check_results._contract_correctness_gate -- the producer never
    # computes a gate itself (the contract is the authority on which
    # named checks are required). Promotion is a different semantic type
    # and stays out of this channel entirely.
    contract_correctness_results: tuple[experiment_contract.CorrectnessResult, ...] = ()
    # RD58 (PA36 migration #4, dev-gpt-agent req_82fbbafe52c0472d Q6):
    # the typed producer->dispatcher promotion channel. The producer
    # supplies per-contract lane_effects (real LaneEffect objects, not
    # JsonObject) + the target_metric to aggregate; the dispatcher owns
    # aggregate_contract_effects() + evaluate_promotion_gate() (the
    # producer never computes a gate itself). {} for producers that do
    # not produce promotion results (the RD12/RD04/RD13/RD26 shape).
    promotion_lane_effects: dict[str, tuple[experiment_contract.LaneEffect, ...]] = (
        field(default_factory=dict)
    )
    promotion_target_metric: dict[str, str] = field(default_factory=dict)
    # RD58 (PA36 migration #4, dev-gpt-agent req_ecb4b77a4c4e4bdd
    # BLOCKER): the typed per-contract trigger evidence. The dispatcher
    # computes evaluate_trigger_proof() on this and passes it to
    # evaluate_promotion_gate() -- a contract PASS without trigger proof
    # would be a fail-OPEN (the target code path may never have run).
    # {} for producers that do not produce promotion results.
    promotion_trigger_evidence: dict[
        str, tuple[experiment_contract.TriggerEvidence, ...]
    ] = field(default_factory=dict)
    # RD73 (PA36 migration #5, dev-gpt-agent req_a232ff7fb1f045db):
    # the typed producer->dispatcher resource evidence channel. The
    # producer supplies per-contract ResourceResult objects (e.g. RD73's
    # graph_cache_entries peak); the dispatcher converts these to
    # {metric: result}, calls evaluate_resource_gate(), and passes
    # resource_gate= into evaluate_promotion_gate(). Missing resource
    # evidence for a resource-bound contract must fail closed.
    promotion_resource_results: dict[
        str, tuple[experiment_contract.ResourceResult, ...]
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(set(self.validation_build_identities)) != 2 or set(
            self.validation_build_identities
        ) != {"control", "subject"}:
            raise ValidationProducerError(
                "ProducerResult.validation_build_identities role set must be "
                f"exactly {{'control', 'subject'}}, got "
                f"{set(self.validation_build_identities)!r}"
            )


class ValidationProducer(Protocol):
    def __call__(self, ctx: ProducerContext) -> ProducerResult: ...


@dataclass(frozen=True)
class ProducerSelection:
    spec: ProducerSpec
    producer: ValidationProducer


def _parse_artifact_names(raw: object, *, where: str) -> frozenset[str]:
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValidationProducerError(f"{where}: artifacts must be a list of strings")
    names: list[str] = []
    for name in raw:
        if not name:
            raise ValidationProducerError(f"{where}: artifact name must be non-empty")
        if "/" in name or "\\" in name:
            raise ValidationProducerError(
                f"{where}: artifact name {name!r} must be a basename"
            )
        if ".." in name:
            raise ValidationProducerError(
                f"{where}: artifact name {name!r} must not contain '..'"
            )
        if any(ch in name for ch in "*?[]"):
            raise ValidationProducerError(
                f"{where}: artifact name {name!r} must not contain a glob"
            )
        if not _ARTIFACT_NAME_PATTERN.match(name):
            raise ValidationProducerError(
                f"{where}: artifact name {name!r} is not a valid basename"
            )
        names.append(name)
    if len(set(names)) != len(names):
        raise ValidationProducerError(f"{where}: duplicate artifact name in {names!r}")
    return frozenset(names)


def _parse_policy_value(raw: object, *, field_name: str, where: str) -> str:
    if not isinstance(raw, str) or raw not in _ALLOWED_POLICY_VALUES[field_name]:
        raise ValidationProducerError(
            f"{where}: {field_name} must be one of "
            f"{sorted(_ALLOWED_POLICY_VALUES[field_name])!r}, got {raw!r}"
        )
    return raw


def _parse_inputs(raw: object, *, where: str) -> dict[str, ProducerInputSpec]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValidationProducerError(f"{where}: input must be a table")
    inputs: dict[str, ProducerInputSpec] = {}
    for name, body in raw.items():
        input_where = f"{where}.input.{name}"
        if not isinstance(body, dict):
            raise ValidationProducerError(f"{input_where} must be a table")
        input_type = body.get("type")
        if not isinstance(input_type, str) or not input_type:
            raise ValidationProducerError(
                f"{input_where}: type must be a non-empty string"
            )
        required = body.get("required", False)
        if not isinstance(required, bool):
            raise ValidationProducerError(f"{input_where}: required must be a boolean")
        inputs[name] = ProducerInputSpec(name=name, type=input_type, required=required)
    return inputs


def resolve_producer(*, patch_dir: Path, producer_id: str) -> ProducerSelection:
    """Load ``patch_dir/validation/producer.toml``, resolve the
    ``[producer.<producer_id>]`` entry, and dynamically load its
    ``entrypoint``'s ``callable_name`` -- the shared, single loader every
    patch's producer goes through. No central ``{"rd12": load_rd12, ...}``
    map: that would just be a different registry monolith.

    Fails closed on: missing producer.toml, unknown producer_id, an
    entrypoint that resolves outside ``patch_dir/validation/`` (absolute
    path or ``..`` traversal), a missing callable, invalid/duplicate
    artifact names, an unknown policy enum value, or a malformed
    ``input`` table.
    """
    validation_dir = patch_dir / "validation"
    manifest_path = validation_dir / "producer.toml"
    if not manifest_path.is_file():
        raise ValidationProducerError(f"{patch_dir}: no validation/producer.toml")

    try:
        doc = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValidationProducerError(f"{manifest_path}: {exc}") from exc

    producers = doc.get("producer")
    if not isinstance(producers, dict):
        raise ValidationProducerError(f"{manifest_path}: no [producer.*] entries")
    body = producers.get(producer_id)
    if not isinstance(body, dict):
        raise ValidationProducerError(
            f"{manifest_path}: unknown producer {producer_id!r}; known: {sorted(producers)}"
        )

    where = f"{manifest_path}: producer.{producer_id}"

    entrypoint_raw = body.get("entrypoint")
    if not isinstance(entrypoint_raw, str) or not entrypoint_raw:
        raise ValidationProducerError(f"{where}: entrypoint must be a non-empty string")
    if Path(entrypoint_raw).is_absolute():
        raise ValidationProducerError(f"{where}: entrypoint must be a relative path")
    if ".." in Path(entrypoint_raw).parts:
        raise ValidationProducerError(f"{where}: entrypoint must not contain '..'")
    entrypoint = (validation_dir / entrypoint_raw).resolve()
    resolved_validation_dir = validation_dir.resolve()
    if (
        resolved_validation_dir not in entrypoint.parents
        and entrypoint != resolved_validation_dir
    ):
        raise ValidationProducerError(
            f"{where}: entrypoint {entrypoint} escapes {resolved_validation_dir}"
        )
    if not entrypoint.is_file():
        raise ValidationProducerError(
            f"{where}: entrypoint {entrypoint} does not exist"
        )

    callable_name = body.get("callable")
    if not isinstance(callable_name, str) or not callable_name:
        raise ValidationProducerError(f"{where}: callable must be a non-empty string")

    spec = ProducerSpec(
        patch_id=patch_dir.name,
        producer_id=producer_id,
        entrypoint=entrypoint,
        callable_name=callable_name,
        trace_probe=_parse_policy_value(
            body.get("trace_probe"), field_name="trace_probe", where=where
        ),
        standard_campaign=_parse_policy_value(
            body.get("standard_campaign"),
            field_name="standard_campaign",
            where=where,
        ),
        correctness_evidence_cli=_parse_policy_value(
            body.get("correctness_evidence_cli"),
            field_name="correctness_evidence_cli",
            where=where,
        ),
        performance_benchmark_cli=_parse_policy_value(
            body.get("performance_benchmark_cli"),
            field_name="performance_benchmark_cli",
            where=where,
        ),
        artifact_names=_parse_artifact_names(body.get("artifacts", []), where=where),
        inputs=_parse_inputs(body.get("input"), where=where),
    )

    module_name = f"_bigcherry_producer_{patch_dir.name}_{producer_id}"
    module_spec = importlib.util.spec_from_file_location(module_name, entrypoint)
    if module_spec is None or module_spec.loader is None:
        raise ValidationProducerError(f"{where}: cannot load entrypoint {entrypoint}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)

    producer_callable = getattr(module, callable_name, None)
    if producer_callable is None or not callable(producer_callable):
        raise ValidationProducerError(
            f"{where}: entrypoint {entrypoint} has no callable {callable_name!r}"
        )

    return ProducerSelection(
        spec=spec,
        producer=cast("ValidationProducer", producer_callable),
    )


def validate_producer_inputs(
    spec: ProducerSpec,
    provided: Mapping[str, str],
) -> Mapping[str, str]:
    """Fail closed on an undeclared ``--producer-input`` key or a missing
    required one. Returns ``provided`` unchanged (a pure gate, not a
    transform) so a caller can pass its result straight into
    ``ProducerContext.inputs``."""
    unknown = sorted(set(provided) - set(spec.inputs))
    if unknown:
        raise ValidationProducerError(
            f"{spec.patch_id}/{spec.producer_id}: undeclared producer-input(s) {unknown!r}; "
            f"known: {sorted(spec.inputs)}"
        )
    missing = sorted(
        name
        for name, input_spec in spec.inputs.items()
        if input_spec.required and name not in provided
    )
    if missing:
        raise ValidationProducerError(
            f"{spec.patch_id}/{spec.producer_id}: missing required producer-input(s) {missing!r}"
        )
    return provided


def validate_producer_cli_compatibility(
    spec: ProducerSpec,
    *,
    correctness_evidence_requested: bool,
    performance_benchmark_requested: bool,
) -> None:
    """Replaces the repeated hand-copied exclusion tuples every
    ``--run-rdXX-contract``-style flag used to carry (the ``--correctness-
    evidence``-ambiguity guard and the ``--run-performance-benchmark``
    mutual-exclusion check, both previously re-copied per flag -- see
    RD12's/RD04's landed wiring in validation_campaign.py, commits
    4312a2d5/8a5641e9, for the exact pattern this generalizes)."""
    if correctness_evidence_requested and spec.correctness_evidence_cli == "forbid":
        raise ValidationProducerError(
            f"{spec.patch_id}/{spec.producer_id}: --correctness-evidence is ambiguous "
            "together with this producer -- it already produces its own authoritative "
            "correctness.json"
        )
    if performance_benchmark_requested and spec.performance_benchmark_cli == "forbid":
        raise ValidationProducerError(
            f"{spec.patch_id}/{spec.producer_id}: --run-performance-benchmark is "
            "mutually exclusive with this producer"
        )


def validate_producer_result(
    spec: ProducerSpec,
    result: ProducerResult,
    *,
    plan: ValidationPlan,
    context: ValidationContext,
) -> None:
    """Fail closed on any violation of the producer/typed-result contract
    (PA36-F step 2/3):

    - every emitted artifact is declared in producer.toml (the
      decentralized equivalent of evidence.py's ``_artifact_refs()``
      hardcoded allowlist: an arbitrary file dropped in the workdir still
      cannot become evidence, even post-migration);
    - ``validation_build_identities`` role set is exactly
      ``{control, subject}`` (already enforced by
      ``ProducerResult.__post_init__``, re-asserted here defensively);
    - every ``ProducerCheckResult.check_id`` is unique and exists in
      ``plan``;
    - the wrapped ``ValidationResult.check_id``/``.capability`` match the
      plan's ``CheckSpec`` for that check;
    - ``contract_ids`` matches ``context.contract_ids_for_check(spec)`` --
      a producer can never silently claim a different contract scope than
      the shared context/plan already resolved;
    - every artifact basename referenced by the wrapped
      ``ValidationResult.artifacts`` is declared by ``spec.artifact_names``;
    - ``disposition`` is either ``None`` or a well-formed single-contract
      verdict payload (exactly one contract id, boolean ``passed``, and at
      most one non-``None`` disposition per contract id across all
      results).
    """
    undeclared = sorted(result.emitted_artifacts - spec.artifact_names)
    if undeclared:
        raise ValidationProducerError(
            f"{spec.patch_id}/{spec.producer_id}: producer emitted undeclared "
            f"artifact(s) {undeclared!r}; declared: {sorted(spec.artifact_names)}"
        )
    if len(set(result.validation_build_identities)) != 2 or set(
        result.validation_build_identities
    ) != {"control", "subject"}:
        raise ValidationProducerError(
            f"{spec.patch_id}/{spec.producer_id}: validation_build_identities role "
            f"set must be exactly {{'control', 'subject'}}, got "
            f"{set(result.validation_build_identities)!r}"
        )

    seen_check_ids: set[str] = set()
    disposition_contract_ids: set[str] = set()
    for record in result.check_results:
        where = f"{spec.patch_id}/{spec.producer_id}: check {record.check_id!r}"
        if record.check_id in seen_check_ids:
            raise ValidationProducerError(
                f"{where}: duplicate check_id in check_results"
            )
        seen_check_ids.add(record.check_id)

        try:
            check_spec: CheckSpec = plan.spec_for(record.check_id)
        except Exception as exc:
            raise ValidationProducerError(f"{where}: {exc}") from exc

        vr = record.validation_result
        if vr.check_id != record.check_id:
            raise ValidationProducerError(
                f"{where}: validation_result.check_id {vr.check_id!r} does not match "
                "ProducerCheckResult.check_id"
            )
        if vr.capability != check_spec.capability:
            raise ValidationProducerError(
                f"{where}: validation_result.capability {vr.capability!r} does not match "
                f"plan capability {check_spec.capability!r}"
            )

        expected_contract_ids = context.contract_ids_for_check(check_spec)
        if record.contract_ids != expected_contract_ids:
            raise ValidationProducerError(
                f"{where}: contract_ids {record.contract_ids!r} does not match "
                f"context.contract_ids_for_check() {expected_contract_ids!r}"
            )

        for artifact in vr.artifacts:
            basename = Path(artifact.name).name
            if artifact.name != basename or basename not in spec.artifact_names:
                raise ValidationProducerError(
                    f"{where}: artifact {artifact.name!r} is not a declared basename; "
                    f"declared: {sorted(spec.artifact_names)}"
                )

        if record.disposition is not None:
            if len(record.contract_ids) != 1:
                raise ValidationProducerError(
                    f"{where}: disposition requires exactly one contract_id, got "
                    f"{record.contract_ids!r}"
                )
            if not isinstance(record.disposition.get("passed"), bool):
                raise ValidationProducerError(
                    f"{where}: disposition['passed'] must be a bool"
                )
            (contract_id,) = record.contract_ids
            if contract_id in disposition_contract_ids:
                raise ValidationProducerError(
                    f"{where}: more than one non-None disposition for contract "
                    f"{contract_id!r}"
                )
            disposition_contract_ids.add(contract_id)
