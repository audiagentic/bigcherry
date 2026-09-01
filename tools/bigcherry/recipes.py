"""Pin management: the single ``pinned = "..."`` ref config/recipes.toml
records.

Patch selection is entirely the v2 ``[source.*]``/``[patch-set.*]``
machinery (``core/config.py``, ``campaign/resolution.py``) -- this module
no longer parses or exposes any patch-selection concept. Its only job is
reading/rewriting the top-level ``pinned`` line.
"""
from __future__ import annotations
import re
import tomllib
from pathlib import Path
from .core import paths

RECIPES_PATH = paths.RECIPES


class RecipeError(ValueError):
    pass


def pinned(path=None) -> str:
    """The top-level ``pinned = "..."`` value."""
    path = Path(path) if path is not None else RECIPES_PATH

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RecipeError(f"no recipe file at {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise RecipeError(f"{path}: {exc}") from None

    value = raw.get("pinned")
    if not isinstance(value, str) or not value:
        raise RecipeError(f"{path}: top-level 'pinned' must be a non-empty string")
    return value


def repin(new_ref: str, path=None) -> str:
    """Rewrites ONLY the `pinned = "..."` line in place via regex
    substitution on the raw text -- NOT a full TOML re-serialise, so comments
    and formatting survive. Returns the old value."""
    if path is None:
        path = RECIPES_PATH
    else:
        path = Path(path)

    content = path.read_text(encoding='utf-8')

    # Find current pinned value
    match = re.search(r'^pinned\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not match:
        raise RecipeError(f"could not find pinned = \"...\" in {path}")

    old_ref = match.group(1)
    new_content = re.sub(
        r'^(pinned\s*=\s*)"[^"]+"',
        rf'\1"{new_ref}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )

    path.write_text(new_content, encoding='utf-8')
    return old_ref
