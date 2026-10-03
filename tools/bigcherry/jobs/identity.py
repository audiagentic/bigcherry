"""Scientific-input freezing for durable job series.

This module resolves logical JobSpec paths/patch IDs into content identities.
It intentionally does not decide scheduling placement or hardware capability.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from bigcherry.experiment import contract as experiment_contract
from bigcherry.patch import registry as patch_registry
from bigcherry.patch import validation_policy
from bigcherry.patch.campaign.scaffold import validated_enhancement_patches

from .model import JobSpec, digest


class ScientificIdentityError(RuntimeError):
    pass


class ScientificIdentityResolver(Protocol):
    def resolve(self, job: JobSpec) -> Mapping[str, object]: ...


def file_identity(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ScientificIdentityError(f"scientific input file does not exist: {resolved}")
    state = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            state.update(chunk)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": state.hexdigest(),
    }


def _project_file_identity(root: Path, relative: str) -> dict[str, object]:
    """Content identity stable across detached worktree absolute paths."""
    path = (root / relative).resolve()
    if not path.is_file():
        raise ScientificIdentityError(f"required framework input is missing: {relative}")
    state = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            state.update(chunk)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": state.hexdigest()}


def _patch_identity(
    descriptor: patch_registry.PatchDescriptor,
    contracts: experiment_contract.ContractRegistry,
) -> dict[str, object]:
    bindings: list[dict[str, str]] = []
    for contract_id in descriptor.experiment_contracts:
        try:
            contract = contracts[contract_id]
        except KeyError as exc:
            raise ScientificIdentityError(
                f"patch {descriptor.patch_id} references unknown contract {contract_id}"
            ) from exc
        bindings.append({"id": contract_id, "hash": contract.contract_hash})
    return {
        "patch_id": descriptor.patch_id,
        "implementation_digest": descriptor.implementation_digest,
        "validation_digest": descriptor.validation_digest,
        "experiment_contracts": list(descriptor.experiment_contracts),
        "contract_bindings": bindings,
    }


@dataclass(frozen=True)
class ProjectScientificIdentityResolver:
    project_root: Path

    def resolve(self, job: JobSpec) -> Mapping[str, object]:
        root = self.project_root.resolve()
        patches = root / "patches"
        config = root / "config"
        registry = patch_registry.load_registry(patches)
        contracts = experiment_contract.load_contracts(config / "experiment-contracts.toml")
        by_id = {descriptor.patch_id: descriptor for descriptor in registry.descriptors}
        try:
            focal = by_id[job.patch]
        except KeyError as exc:
            raise ScientificIdentityError(f"unknown focal patch: {job.patch}") from exc
        # Starting new managed work must satisfy the same anti-grandfather
        # validation-package rule as validation_campaign itself.
        validation_policy.require_execution_package(focal, root=patches)

        common: list[dict[str, object]] = []
        for patch_id in job.common_patches:
            try:
                common.append(_patch_identity(by_id[patch_id], contracts))
            except KeyError as exc:
                raise ScientificIdentityError(f"unknown common patch: {patch_id}") from exc

        validated_ids = validated_enhancement_patches(
            patch_id=job.patch,
            common_patches=job.common_patches,
            recipes=config / "recipes.toml",
        )
        validated: list[dict[str, object]] = []
        for patch_id in validated_ids:
            try:
                validated.append(_patch_identity(by_id[patch_id], contracts))
            except KeyError as exc:
                raise ScientificIdentityError(
                    f"validated-enhancements references missing patch: {patch_id}"
                ) from exc

        producer_inputs: list[dict[str, object]] = []
        for key, raw in job.producer_inputs:
            candidate = Path(raw).expanduser()
            item: dict[str, object] = {"key": key, "value": raw}
            if candidate.is_file():
                item["file"] = file_identity(candidate)
            producer_inputs.append(item)

        result: dict[str, object] = {
            "schema": "bigcherry.scientific-identity.v2",
            "focal": _patch_identity(focal, contracts),
            "common": common,
            "validated": validated,
            # These framework registries influence composition/contract meaning
            # but are not part of a patch implementation digest. Freeze their
            # bytes with repository-relative paths so later sessions cannot
            # silently reinterpret the same series after a config-only commit.
            "framework_inputs": {
                "recipes": _project_file_identity(root, "config/recipes.toml"),
                "experiment_contracts": _project_file_identity(
                    root, "config/experiment-contracts.toml"
                ),
            },
            "model": file_identity(Path(job.model)),
            "producer": job.producer,
            "producer_inputs": producer_inputs,
            "producer_corpus": (
                None if job.producer_corpus is None else file_identity(Path(job.producer_corpus))
            ),
            "baseline_source": job.baseline_source,
        }
        return {**result, "identity_hash": digest(result, person=b"bc-science-id")}


class StaticScientificIdentityResolver:
    """Deterministic mock resolver used by service tests."""

    def __init__(self, value: Mapping[str, object] | None = None) -> None:
        self.value = dict(value or {"schema": "mock", "identity_hash": "mock-science"})

    def resolve(self, job: JobSpec) -> Mapping[str, object]:
        return dict(self.value)
