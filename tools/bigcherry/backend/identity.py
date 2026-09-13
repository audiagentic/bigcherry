"""Canonical backend-neutral resolved software/provider identity (RRVP02)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from ..core.canonical import canonical_json_bytes, domain_blake2b_128


RESOLVED_STACK_IDENTITY_SCHEMA = "bigcherry/resolved-stack-identity/v1"
RESOLVED_STACK_FINGERPRINT_DOMAIN = "bigcherry/backend-stack/v1"

IdentityState = Literal["present", "missing", "unknown"]
IdentityAttributeName = Literal[
    "name",
    "vendor",
    "version",
    "build_id",
    "api_version",
    "abi_version",
    "soname",
    "artifact_sha256",
]

_IDENTITY_STATES = frozenset(("present", "missing", "unknown"))
_IDENTITY_ATTRIBUTE_NAMES = frozenset(
    (
        "name",
        "vendor",
        "version",
        "build_id",
        "api_version",
        "abi_version",
        "soname",
        "artifact_sha256",
    )
)
_COMPONENT_NAME_RE = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*\Z")
_BACKEND_RE = re.compile(r"[a-z][a-z0-9_-]*\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class BackendIdentityError(ValueError):
    pass


def _require_exact_keys(
    raw: Mapping[object, object], *, expected: frozenset[str], context: str
) -> None:
    keys = set(raw)
    non_strings = tuple(key for key in keys if not isinstance(key, str))
    if non_strings:
        raise BackendIdentityError(
            f"{context} field names must be strings, got {non_strings!r}"
        )
    string_keys = cast(set[str], keys)
    missing = expected - string_keys
    unknown = string_keys - expected
    if missing or unknown:
        raise BackendIdentityError(
            f"{context} has invalid fields: missing={sorted(missing)!r}, "
            f"unknown={sorted(unknown)!r}"
        )


def _require_state(value: object, *, context: str) -> IdentityState:
    if not isinstance(value, str) or value not in _IDENTITY_STATES:
        raise BackendIdentityError(
            f"{context} state must be one of {sorted(_IDENTITY_STATES)!r}, got {value!r}"
        )
    return cast(IdentityState, value)


def _require_component_name(value: object) -> str:
    if not isinstance(value, str) or _COMPONENT_NAME_RE.fullmatch(value) is None:
        raise BackendIdentityError(
            "component name must be a lowercase semantic key such as "
            f"'provider.rocblas', got {value!r}"
        )
    return value


def _require_attribute_name(value: object) -> IdentityAttributeName:
    if not isinstance(value, str) or value not in _IDENTITY_ATTRIBUTE_NAMES:
        raise BackendIdentityError(
            f"unknown identity attribute {value!r}; allowed={sorted(_IDENTITY_ATTRIBUTE_NAMES)!r}"
        )
    return cast(IdentityAttributeName, value)


@dataclass(frozen=True)
class IdentityAttribute:
    """One stable software attribute with explicit observation state.

    ``missing`` means the producer positively established absence; ``unknown``
    means it could not establish a value. Neither state may carry a value.
    """

    name: IdentityAttributeName
    state: IdentityState
    value: str | None = None

    def __post_init__(self) -> None:
        _require_attribute_name(self.name)
        _require_state(self.state, context=f"attribute {self.name!r}")
        if self.state == "present":
            if not isinstance(self.value, str) or not self.value.strip():
                raise BackendIdentityError(
                    f"present attribute {self.name!r} requires a non-blank string value"
                )
        elif self.value is not None:
            raise BackendIdentityError(
                f"{self.state} attribute {self.name!r} must have value=None"
            )
        if self.name == "artifact_sha256" and self.state == "present":
            assert self.value is not None
            if _SHA256_RE.fullmatch(self.value) is None:
                raise BackendIdentityError(
                    "artifact_sha256 must be exactly 64 lowercase hexadecimal characters"
                )

    def document(self) -> dict[str, object]:
        return {"state": self.state, "value": self.value}

    @classmethod
    def from_document(cls, name: object, document: object) -> "IdentityAttribute":
        attribute_name = _require_attribute_name(name)
        if not isinstance(document, Mapping):
            raise BackendIdentityError(f"attribute {attribute_name!r} must be an object")
        _require_exact_keys(
            document,
            expected=frozenset(("state", "value")),
            context=f"attribute {attribute_name!r}",
        )
        raw_value = document["value"]
        if raw_value is not None and not isinstance(raw_value, str):
            raise BackendIdentityError(
                f"attribute {attribute_name!r}.value must be a string or null"
            )
        return cls(
            name=attribute_name,
            state=_require_state(document["state"], context=f"attribute {attribute_name!r}"),
            value=raw_value,
        )


@dataclass(frozen=True)
class SoftwareComponentIdentity:
    """Identity for one semantic software/provider component.

    A missing/unknown component is explicit and carries no attributes.
    """

    name: str
    state: IdentityState
    attributes: tuple[IdentityAttribute, ...] = ()

    def __post_init__(self) -> None:
        _require_component_name(self.name)
        _require_state(self.state, context=f"component {self.name!r}")
        attributes = tuple(self.attributes)
        if not all(isinstance(item, IdentityAttribute) for item in attributes):
            raise BackendIdentityError(
                f"component {self.name!r} attributes must be IdentityAttribute values"
            )
        names = tuple(item.name for item in attributes)
        if len(set(names)) != len(names):
            raise BackendIdentityError(f"component {self.name!r} has duplicate attributes")
        attributes = tuple(sorted(attributes, key=lambda item: item.name))
        if self.state != "present" and attributes:
            raise BackendIdentityError(
                f"{self.state} component {self.name!r} cannot carry attributes"
            )
        object.__setattr__(self, "attributes", attributes)

    def document(self) -> dict[str, object]:
        return {
            "state": self.state,
            "attributes": {item.name: item.document() for item in self.attributes},
        }

    @classmethod
    def from_document(cls, name: object, document: object) -> "SoftwareComponentIdentity":
        component_name = _require_component_name(name)
        if not isinstance(document, Mapping):
            raise BackendIdentityError(f"component {component_name!r} must be an object")
        _require_exact_keys(
            document,
            expected=frozenset(("state", "attributes")),
            context=f"component {component_name!r}",
        )
        raw_attributes = document["attributes"]
        if not isinstance(raw_attributes, Mapping):
            raise BackendIdentityError(
                f"component {component_name!r}.attributes must be an object"
            )
        attributes = tuple(
            IdentityAttribute.from_document(attribute_name, attribute_document)
            for attribute_name, attribute_document in raw_attributes.items()
        )
        return cls(
            name=component_name,
            state=_require_state(document["state"], context=f"component {component_name!r}"),
            attributes=attributes,
        )


@dataclass(frozen=True)
class ResolvedStackIdentity:
    """Versioned backend-neutral software/provider identity.

    Only listed components participate in this document. Stage-specific probes
    own which component names they are required to emit; omission is therefore
    not synonymous with an explicit ``missing`` component.
    """

    backend: str
    components: tuple[SoftwareComponentIdentity, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or _BACKEND_RE.fullmatch(self.backend) is None:
            raise BackendIdentityError(
                f"backend must be a lowercase semantic token, got {self.backend!r}"
            )
        components = tuple(self.components)
        if not components:
            raise BackendIdentityError("resolved stack identity requires at least one component")
        if not all(isinstance(item, SoftwareComponentIdentity) for item in components):
            raise BackendIdentityError(
                "components must contain only SoftwareComponentIdentity values"
            )
        names = tuple(item.name for item in components)
        if len(set(names)) != len(names):
            raise BackendIdentityError("resolved stack identity has duplicate components")
        object.__setattr__(
            self, "components", tuple(sorted(components, key=lambda item: item.name))
        )

    def document(self) -> dict[str, object]:
        return {
            "schema": RESOLVED_STACK_IDENTITY_SCHEMA,
            "backend": self.backend,
            "components": {item.name: item.document() for item in self.components},
        }

    @classmethod
    def from_document(cls, document: object) -> "ResolvedStackIdentity":
        if not isinstance(document, Mapping):
            raise BackendIdentityError("resolved stack identity must be an object")
        _require_exact_keys(
            document,
            expected=frozenset(("schema", "backend", "components")),
            context="resolved stack identity",
        )
        if document["schema"] != RESOLVED_STACK_IDENTITY_SCHEMA:
            raise BackendIdentityError(
                f"unsupported resolved stack identity schema: {document['schema']!r}"
            )
        backend = document["backend"]
        if not isinstance(backend, str):
            raise BackendIdentityError("resolved stack identity backend must be a string")
        raw_components = document["components"]
        if not isinstance(raw_components, Mapping):
            raise BackendIdentityError("resolved stack identity components must be an object")
        components = tuple(
            SoftwareComponentIdentity.from_document(component_name, component_document)
            for component_name, component_document in raw_components.items()
        )
        return cls(backend=backend, components=components)

    def canonical_json_bytes(self) -> bytes:
        return canonical_json_bytes(self.document())

    @property
    def fingerprint(self) -> str:
        return domain_blake2b_128(
            RESOLVED_STACK_FINGERPRINT_DOMAIN,
            self.document(),
        )
