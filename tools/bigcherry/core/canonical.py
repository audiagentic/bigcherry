"""Small shared canonical JSON and domain-separated digest helpers."""

from __future__ import annotations

import hashlib
import json


def canonical_json_bytes(value: object) -> bytes:
    """Return BigCherry's compact, sorted-key UTF-8 JSON representation."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def domain_blake2b_128(domain: str, value: object) -> str:
    """Return a 128-bit BLAKE2b hex digest separated by ``domain``."""
    return hashlib.blake2b(
        domain.encode("utf-8") + b"\0" + canonical_json_bytes(value),
        digest_size=16,
    ).hexdigest()
