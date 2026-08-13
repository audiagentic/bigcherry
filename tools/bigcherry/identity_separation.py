"""Offline checks for the four autotune identity namespaces (HI19).

Dispatch/signature identity, build and hardware context, candidate identity,
and measured observations are related records, not interchangeable values.
This module validates the boundaries when fields are present.  Older valid
artifacts may omit the newer fields and remain readable; a partially supplied
new namespace fails closed instead of being filled from another namespace.
"""

from __future__ import annotations

import re
from typing import Any


_DIGEST = re.compile(r"[0-9a-fA-F]{32}\Z")
_REVISION = re.compile(r"[0-9a-fA-F]{40}\Z")
_IDENTITY_FIELDS = {"dispatch", "signature", "hardware", "candidate_digest"}
_CONTEXT_FIELDS = {"source_revision", "manifest_hash", "build_descriptor_hash",
                   "variant_set", "hardware_key", "hardware_context"}


class IdentitySeparationError(ValueError):
    """An artifact mixes or substitutes distinct identity namespaces."""


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise IdentitySeparationError(f"{field} must be a 16-byte hexadecimal digest")
    return value.lower()


def _optional_digest(value: Any, field: str) -> str | None:
    return None if value is None else _digest(value, field)


def validate_measurement_identity(
    row: dict[str, Any], *, header: dict[str, Any] | None = None,
    where: str = "measurement",
) -> dict[str, Any]:
    """Validate namespace separation without changing or enriching ``row``.

    ``dispatch`` is the composite lookup identity.  When newer artifacts emit
    component identities remain separate from candidate/observation payloads.
    No missing value is copied from another field.  Legacy rows may contain
    only the signature field for compatibility, but this validator never
    promotes it into a hardware identity.
    """
    if not isinstance(row, dict):
        raise IdentitySeparationError(f"{where} must be an object")

    for field in ("dispatch", "signature", "hardware", "candidate_digest"):
        if field in row:
            try:
                _digest(row[field], f"{where}.{field}")
            except IdentitySeparationError as exc:
                if field == "dispatch":
                    raise IdentitySeparationError(f"{where}: invalid dispatch digest") from exc
                raise

    if "canonical" in row and not isinstance(row["canonical"], dict):
        raise IdentitySeparationError(f"{where}.canonical must be an operation-signature object")
    if "hardware_key" in row and not isinstance(row["hardware_key"], dict):
        raise IdentitySeparationError(f"{where}.hardware_key must be a hardware-context object")

    candidates = row.get("candidates")
    if candidates is not None:
        if not isinstance(candidates, list):
            raise IdentitySeparationError(f"{where}.candidates must be a list")
        names: set[str] = set()
        for index, candidate in enumerate(candidates):
            cwhere = f"{where}.candidates[{index}]"
            if not isinstance(candidate, dict):
                raise IdentitySeparationError(f"{cwhere} must be an object")
            name = candidate.get("name")
            if not isinstance(name, str) or not name or name in names:
                raise IdentitySeparationError(f"{cwhere}.name is not a unique candidate identity")
            names.add(name)
            for field in _IDENTITY_FIELDS | {"canonical", "observations", "hardware_key"}:
                if field in candidate:
                    raise IdentitySeparationError(
                        f"{cwhere}.{field} conflates candidate identity with "
                        "signature, context, or observation data")
            if "candidate_digest" in candidate:
                _digest(candidate["candidate_digest"], f"{cwhere}.candidate_digest")
        winner = row.get("winner")
        if winner is not None and (not isinstance(winner, str) or winner not in names):
            raise IdentitySeparationError(f"{where}.winner is not one of the candidate identities")

    observations = row.get("observations")
    if observations is not None:
        if not isinstance(observations, list):
            raise IdentitySeparationError(f"{where}.observations must be a list")
        for index, observation in enumerate(observations):
            if not isinstance(observation, dict):
                raise IdentitySeparationError(f"{where}.observations[{index}] must be an object")
            if _IDENTITY_FIELDS & observation.keys():
                raise IdentitySeparationError(
                    f"{where}.observations[{index}] must not carry durable identity fields")

    provenance = row.get("provenance")
    if provenance is not None:
        if not isinstance(provenance, dict):
            raise IdentitySeparationError(f"{where}.provenance must be a build-context object")
        for field in ("manifest_hash", "build_descriptor_hash"):
            if field in provenance:
                _digest(provenance[field], f"{where}.provenance.{field}")
        if "source_revision" in provenance and not _REVISION.fullmatch(
                str(provenance["source_revision"])):
            raise IdentitySeparationError(f"{where}.provenance.source_revision is invalid")
        if "hardware" in provenance or "signature" in provenance:
            raise IdentitySeparationError(
                f"{where}.provenance must not carry signature or hardware identity")

    if header is not None:
        if not isinstance(header, dict):
            raise IdentitySeparationError("measurement header must be a build-context object")
        for field in ("manifest_hash", "build_descriptor_hash"):
            if field in header and header[field] is not None:
                # Legacy build descriptors were free-form labels.  Only enforce
                # digest shape where the field is explicitly marked as one.
                if field == "manifest_hash":
                    _digest(header[field], f"header.{field}")
        if "hardware" in header or "signature" in header:
            raise IdentitySeparationError("header must not carry per-operation identity")
    return row
