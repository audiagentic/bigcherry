"""Pin management: the single ``pinned = "..."`` ref an engine's recipes file
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


class RecipeError(ValueError):
    pass


def pinned(path) -> str:
    """The top-level ``pinned = "..."`` value of the recipes file at ``path``."""
    path = Path(path)

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


def repin(new_ref: str, path) -> str:
    """Rewrites ONLY the `pinned = "..."` line in place via regex
    substitution on the raw text -- NOT a full TOML re-serialise, so comments
    and formatting survive. Returns the old value."""
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


def set_pinned_version(version: str, path) -> str:
    """Rewrites ONLY the `pinned-version = "..."` line, as repin does for `pinned`. Returns the old value."""
    path = Path(path)
    content = path.read_text(encoding='utf-8')
    match = re.search(r'^pinned-version\s*=\s*"([^"]*)"', content, re.MULTILINE)
    if not match:
        raise RecipeError(f"could not find pinned-version = \"...\" in {path}")
    path.write_text(
        re.sub(r'^(pinned-version\s*=\s*)"[^"]*"', rf'\g<1>"{version}"', content, count=1, flags=re.MULTILINE),
        encoding='utf-8',
    )
    return match.group(1)


def version_of_tag(tag: str) -> str:
    """The upstream release version a tag names (v1.13.0 -> 1.13.0)."""
    match = re.fullmatch(r"v?(\d+(?:\.\d+)*)", tag)
    if not match:
        raise RecipeError(f"tag {tag!r} does not name a release version (v1.13.0)")
    return match.group(1)
