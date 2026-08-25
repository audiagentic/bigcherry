"""Compatibility facade for the canonical release.pin_status module."""

from __future__ import annotations

import importlib

_CANONICAL = importlib.import_module("bigcherry.release.pin_status")
globals().update(
    {
        name: value
        for name, value in vars(_CANONICAL).items()
        if not name.startswith("__")
    }
)
