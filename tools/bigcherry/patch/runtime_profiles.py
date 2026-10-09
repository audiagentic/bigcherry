"""QFP23 runtime profiles: parse and validate the release profile folder (engines/llamacpp/overlay/profile/*.ini).

Model- and scope-specific runtime settings live in these files, not in patch code. They sit in the source overlay (part
of the source identity); the build copies the folder next to the binaries (bin/profile/, part of the runtime bundle),
and 0910_feature_sets loads it at startup in C relative to libggml-base; this module is the authoritative validator used by patch-lint (and the reference the C loader's own
checks mirror): grammar, duplicate profiles/keys, unknown @includes, include cycles, conflicting assignments after
flattening, and flag names that no patch documents in ENV_DOCS.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from bigcherry.core import paths

PROFILE_DIR_NAME = "profile"
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_FLAG = re.compile(r"^[A-Z][A-Z0-9_]{0,126}$")
_VALUE = re.compile(r'^[^\s#"\\]{1,255}$')
_ARCH = re.compile(r"^[a-z0-9_]{1,63}$")


@dataclass
class Profile:
    name: str
    source: str
    line: int
    description: str = ""
    arch: list[str] = field(default_factory=list)
    includes: list[str] = field(default_factory=list)
    assignments: list[tuple[str, str]] = field(default_factory=list)


class ProfileError(ValueError):
    pass


def profile_dir(root: Path | None = None) -> Path:
    return (root or paths.SRC_OVERLAY) / PROFILE_DIR_NAME


def parse_text(text: str, source: str) -> list[Profile]:
    profiles: list[Profile] = []
    cur: Profile | None = None
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        where = f"{source}:{n}"
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            if not _NAME.match(name):
                raise ProfileError(f"{where}: bad profile name {name!r}")
            cur = Profile(name, source, n)
            profiles.append(cur)
            continue
        if cur is None:
            raise ProfileError(f"{where}: entry outside a [profile] section")
        if line.startswith("@"):
            ref = line[1:].strip()
            if not _NAME.match(ref) or ref in cur.includes:
                raise ProfileError(f"{where}: bad or duplicate include {line!r}")
            cur.includes.append(ref)
            continue
        if "=" not in line:
            raise ProfileError(f"{where}: expected NAME = VALUE, @include, description or arch: {line!r}")
        key, value = (part.strip() for part in line.split("=", 1))
        if key == "description":
            if cur.description or not value or '"' in value or "\\" in value:
                raise ProfileError(f"{where}: description must be given once, plain text")
            cur.description = value
        elif key == "arch":
            if not _ARCH.match(value) or value in cur.arch:
                raise ProfileError(f"{where}: bad or duplicate arch {value!r}")
            cur.arch.append(value)
        elif _FLAG.match(key):
            if not _VALUE.match(value):
                raise ProfileError(f"{where}: bad value for {key}: {value!r}")
            if any(k == key for k, _ in cur.assignments):
                raise ProfileError(f"{where}: {key} assigned twice in [{cur.name}]")
            cur.assignments.append((key, value))
        else:
            raise ProfileError(f"{where}: unknown key {key!r}")
    return profiles


def load(location: Path | None = None) -> dict[str, Profile]:
    """Profiles from every *.ini in a folder (default: the overlay's profile/), or from one file."""
    location = location or profile_dir()
    files = sorted(location.glob("*.ini")) if location.is_dir() else [location]
    by_name: dict[str, Profile] = {}
    for path in files:
        for prof in parse_text(path.read_text(encoding="utf-8"), path.name):
            if prof.name in by_name:
                other = by_name[prof.name]
                raise ProfileError(f"{prof.source}:{prof.line}: profile [{prof.name}] already defined at "
                                   f"{other.source}:{other.line}")
            by_name[prof.name] = prof
    return by_name


def flatten(profiles: dict[str, Profile], name: str, _stack: tuple[str, ...] = ()) -> dict[str, str]:
    if name in _stack:
        raise ProfileError(f"profile include cycle: {' -> '.join((*_stack, name))}")
    prof = profiles[name]
    out: dict[str, str] = {}
    items: list[tuple[str, str]] = []
    for ref in prof.includes:
        if ref not in profiles:
            raise ProfileError(f"{prof.source}:{prof.line}: [{name}] includes unknown profile @{ref}")
        items.extend(flatten(profiles, ref, (*_stack, name)).items())
    items.extend(prof.assignments)
    for key, value in items:
        if out.get(key, value) != value:
            raise ProfileError(f"[{name}] sets {key} to both {out[key]!r} and {value!r}")
        out[key] = value
    return out


def check(location: Path | None = None, documented_flags: set[str] | None = None) -> list[str]:
    """Return problem strings (empty = clean)."""
    try:
        profiles = load(location)
        problems: list[str] = []
        for name, prof in profiles.items():
            if not prof.description:
                problems.append(f"{prof.source}:{prof.line}: [{name}] has no description")
            flat = flatten(profiles, name)
            if documented_flags is not None:
                for key in sorted(set(flat) - documented_flags):
                    problems.append(f"{prof.source}: [{name}] sets {key}, which no patch documents in ENV_DOCS")
        return problems
    except ProfileError as exc:
        return [str(exc)]


def documented_flags(patches_root: Path | None = None) -> set[str]:
    """Every runtime flag name documented by some packaged patch's ENV_DOCS (plus BIGCHERRY_FEATURES)."""
    from bigcherry.patch import registry

    root = patches_root or paths.PATCHES
    reg = registry.load_registry(root)
    names = {"BIGCHERRY_FEATURES"}
    for desc in reg.descriptors:
        names.update(doc.name for doc in registry.load_env_docs(desc, root=root))
    return names
