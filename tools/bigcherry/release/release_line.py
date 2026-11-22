"""A pin bump carries its release line by itself.

The release version's major comes from the pin (llama.cpp: the build number; Radiance: 2.2.0 is 2020), but
release-please keeps counting from the manifest's version unless the package carries `release-as`. So:

  declare   `pin-bump` sets the package's `release-as` (<major>.0.0) in the commit that moves the pin, when the
            new major differs from the manifest's;
  check     CI fails when the pin's major differs from the manifest's and no `release-as` names it, and when
            `release-as` is older than the manifest (a release already went past it);
  clear     once the manifest has reached `release-as` the release exists, and the release-please workflow opens a
            pull request that removes the key (`python -m bigcherry.release.release_line clear`).

Between the release and the clearing pull request the key equals the manifest version: that state is allowed, since
it is the release pull request's own state.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

from ..core import engines
from . import notes as _notes
from .pin_release import PinReleaseError, _release_as_value, _set_release_as_text

_CONFIG = "release-please-config.json"
_MANIFEST = ".release-please-manifest.json"


class ReleaseLineError(RuntimeError):
    pass


def _version(text: str, what: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", text.strip())
    if match is None:
        raise ReleaseLineError(f"{what} {text!r} is not major.minor.patch")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _manifest(root: Path) -> dict[str, str]:
    try:
        return {k: str(v) for k, v in json.loads((root / _MANIFEST).read_text(encoding="utf-8")).items()}
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseLineError(f"cannot read {_MANIFEST}: {exc}") from exc


def _config_text(root: Path) -> str:
    try:
        return (root / _CONFIG).read_text(encoding="utf-8")
    except OSError as exc:
        raise ReleaseLineError(f"cannot read {_CONFIG}: {exc}") from exc


def _pending(root: Path, package: str) -> str | None:
    try:
        return _release_as_value(_config_text(root), package, "release-line")
    except PinReleaseError as exc:
        raise ReleaseLineError(str(exc)) from exc


def _write(root: Path, package: str, version: str | None, expected: str | None = None) -> bool:
    text = _config_text(root)
    try:
        updated, changed = _set_release_as_text(text, package, version, "release-line", expected)
    except PinReleaseError as exc:
        raise ReleaseLineError(str(exc)) from exc
    if changed:
        (root / _CONFIG).write_text(updated, encoding="utf-8", newline="")
    return changed


def declare(root: Path, package: str, new_major: int) -> bool:
    """Make the release line follow a pin whose major is ``new_major``. Returns True when the config changed.

    A major that differs from the manifest's sets `release-as` to <major>.0.0. A pin that moves back to the
    manifest's major drops a key that was set for a major that never released."""
    current = _manifest(root).get(package)
    if current is None:
        raise ReleaseLineError(f"{_MANIFEST} has no package {package!r}")
    pending = _pending(root, package)
    if _version(current, f"{package} manifest version")[0] == new_major:
        if pending is not None and _version(pending, f"{package} release-as")[0] != new_major:
            return _write(root, package, None, expected=pending)
        return False
    return _write(root, package, f"{new_major}.0.0")


def problems(root: Path, package: str, pin_major: int) -> list[str]:
    """What is wrong with the release line of ``package`` for a pin of major ``pin_major`` (empty = fine)."""
    current = _manifest(root).get(package)
    if current is None:
        return [f"{_MANIFEST} has no package {package!r}"]
    cur = _version(current, f"{package} manifest version")
    pending = _pending(root, package)
    found: list[str] = []
    if pending is not None:
        pend = _version(pending, f"{package} release-as")
        if pend < cur:
            found.append(
                f"{package}: release-as {pending} is stale (the manifest is already at {current}); remove it "
                "(the release-please workflow opens that pull request; `release_line clear` does it by hand)")
        elif pend[0] != pin_major:
            found.append(f"{package}: release-as {pending} does not name the pin's major; it must be {pin_major}.0.0")
    elif cur[0] != pin_major:
        found.append(
            f"{package}: the pin's release major is {pin_major} but the manifest is at {current} and "
            f"{_CONFIG} has no release-as, so release-please would continue the old line; `pin-bump` sets "
            f"`release-as` {pin_major}.0.0 when it moves the pin (set it by hand for a pin moved without it)")
    return found


def pin_major(root: Path, engine: str) -> int:
    """The release major of the pin as declared in the working tree's recipes."""
    config = _notes.load_config(root, engine)
    recipes = tomllib.loads((root / config.recipes).read_text(encoding="utf-8"))
    pinned = str(recipes.get("pinned", ""))
    try:
        if config.version_source == "build-tag":
            return _notes.pin_version(root, config, pinned).major
        version = str(recipes.get("pinned-version", ""))
        if not re.fullmatch(r"\d+(\.\d+)*", version):
            raise ReleaseLineError(f"{config.recipes} declares no pinned-version")
        return _notes.pinned_version_major(version).major
    except _notes.ReleaseNotesError as exc:
        raise ReleaseLineError(str(exc)) from exc


def follow_pin(root: Path, engine: str) -> Path | None:
    """`pin-bump`'s declare step: after the recipes moved, declare the release line for the new pin. Returns the
    config path when it changed (the caller commits it with the pin), else None."""
    package = engines.load(root, engine).release_package()
    if package not in json.loads(_config_text(root)).get("packages", {}):
        return None
    return root / _CONFIG if declare(root, package, pin_major(root, engine)) else None


def check_repository(root: Path) -> list[str]:
    """`problems` for every engine that has a release line."""
    found: list[str] = []
    for name in engines.names(root):
        layout = engines.load(root, name)
        package = layout.release_package()
        if package not in json.loads(_config_text(root)).get("packages", {}):
            continue
        found += problems(root, package, pin_major(root, name))
    return found


def clear_released(root: Path) -> list[str]:
    """Remove every `release-as` the manifest has reached. Returns the packages cleared."""
    manifest = _manifest(root)
    cleared: list[str] = []
    for package, current in manifest.items():
        pending = _pending(root, package) if package in json.loads(_config_text(root)).get("packages", {}) else None
        if pending is None:
            continue
        if _version(pending, f"{package} release-as") <= _version(current, f"{package} manifest version"):
            _write(root, package, None, expected=pending)
            cleared.append(package)
    return cleared


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(args[1]) if len(args) > 1 else Path.cwd()
    if not args or args[0] not in ("clear", "check"):
        print("usage: python -m bigcherry.release.release_line clear|check [REPO_ROOT]", file=sys.stderr)
        return 2
    if args[0] == "check":
        found = check_repository(root)
        for line in found:
            print(line, file=sys.stderr)
        return 1 if found else 0
    for package in clear_released(root):
        print(package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
