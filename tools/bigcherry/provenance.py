"""Common provenance-v2 construction and namespace compatibility checks."""

from __future__ import annotations

from typing import Any


class ProvenanceError(ValueError):
    pass


SCHEMA_VERSION = 2
REQUIRED_NAMESPACES = ("project", "source", "build", "workload", "campaign")


def make(*, project: dict[str, Any], source: dict[str, Any], build: dict[str, Any],
         workload: dict[str, Any], campaign: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "project": dict(project),
        "source": dict(source),
        "build": dict(build),
        "workload": dict(workload),
        "campaign": dict(campaign),
    }


def validate(document: object) -> dict[str, Any]:
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise ProvenanceError("only complete provenance schema v2 is promotable")
    missing = [name for name in REQUIRED_NAMESPACES if not isinstance(document.get(name), dict)]
    if missing:
        raise ProvenanceError("missing provenance namespace(s): " + ", ".join(missing))
    return document


def require_compatible(document: object, **expected: str) -> dict[str, Any]:
    value = validate(document)
    for dotted, wanted in expected.items():
        namespace, _, field = dotted.partition(".")
        if namespace not in REQUIRED_NAMESPACES or not field:
            raise ProvenanceError(f"invalid expected provenance field {dotted!r}")
        actual = value[namespace].get(field)
        if actual != wanted:
            raise ProvenanceError(
                f"provenance mismatch for {dotted}: expected {wanted!r}, got {actual!r}"
            )
    return value
