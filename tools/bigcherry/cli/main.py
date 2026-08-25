"""Canonical CLI bootstrap.

Command parsing and handlers remain in the legacy module during TR09. This
module owns the stable package entrypoint and deliberately imports lazily to
avoid making the legacy implementation depend on the presentation layer.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """Run the existing parser/handler implementation through the new entrypoint."""
    from ..__main__ import _legacy_main

    return int(_legacy_main(argv))
