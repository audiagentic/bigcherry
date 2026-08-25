"""Compatibility facade for the canonical bigcherry.tuning.inventory module.

A true module alias (not a copy of its names) -- so mutating an attribute
through either import path is visible to both, since they are the same
module object.
"""

from __future__ import annotations

import sys
import importlib

sys.modules[__name__] = importlib.import_module("bigcherry.tuning.inventory")
