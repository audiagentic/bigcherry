"""Which engine each command acts on (MEN10).

One table, read once at the CLI boundary. Commands never branch on an engine name inside: a command either works for
whichever engine it is given, takes the engine from the lane it builds, covers every engine, or is a tool of one
engine and refuses the others here.

A command moves from ``ONLY`` to ``ENGINE`` when its handler has been migrated to take ``args.engine``; from then on
``--engine`` is required for it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..core import engines, paths

#: Takes ``--engine`` (required) and acts on that engine only.
ENGINE = "engine"
#: Takes the engine from the build lane's source (the recipe that declares it); ``--engine`` is not accepted.
LANE = "lane"
#: Covers every declared engine; ``--engine`` is not accepted.
PLATFORM = "platform"
#: Declares its own engine argument (engine-bench).
SELF = "self"

#: Commands that act on any engine, with their handler migrated to ``args.engine``.
_ENGINE_COMMANDS: frozenset[str] = frozenset(
    {
        "patches",
        "patch-status",
        "patch-explain",
        "patch-graph",
        "patch-verify-evidence",
        "patch-gates",
        "patch-doc",
        "patch-rebase-check",
        "pull",
        "audit",
        "repin",
        "pin-status",
        "pin-bump",
        "pin-release",
        "release-notes",
        "sources",
        "smoke-cpu",
        "accept",
    }
)

_LANE_COMMANDS = frozenset({"build", "campaign-build"})
#: patch-lint checks every engine's patch tree. patch-disposition keeps one repository-wide store keyed by patch ID
#: (IDs carry their engine's prefix, so they never collide) and bound to an exact revision and digest, so it has no
#: engine to choose.
#: remote runs a bigcherry command (carrying its own --engine) on a GPU host, so it has no engine of its own.
_PLATFORM_COMMANDS = frozenset({"patch-lint", "patch-disposition", "remote", "run", "clean"})
_SELF_COMMANDS = frozenset({"engine-bench"})

#: Everything else is today a llama.cpp tool. Each entry is either genuinely llama.cpp-only (tuning, replay,
#: autotune, kernel profiling) or not yet migrated; both refuse another engine rather than silently act on llama.cpp.
_ONLY_LLAMACPP = engines.LLAMACPP


def scope(command: str) -> str:
    if command in _ENGINE_COMMANDS:
        return ENGINE
    if command in _LANE_COMMANDS:
        return LANE
    if command in _PLATFORM_COMMANDS:
        return PLATFORM
    if command in _SELF_COMMANDS:
        return SELF
    return _ONLY_LLAMACPP


def vendor_root(engine: str) -> Path:
    """The engine's vendor checkout (declared in its engine.toml); it belongs to the primary checkout."""
    return engines.load(paths.REPO_ROOT, engine).vendor_root(paths.primary_root())


def checkout_root(args: argparse.Namespace) -> Path:
    """The checkout a command reads: ``--llama-root`` when given, otherwise the chosen engine's vendor checkout."""
    override = getattr(args, "llama_root", None)
    return paths.llama_root(override) if override is not None else vendor_root(args.engine)


def resolve(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate ``--engine`` for ``args.subcommand`` and settle ``args.engine``. Errors exit through ``parser``."""
    requested = args.global_engine
    declared = engines.names(paths.REPO_ROOT)
    if requested is not None and requested not in declared:
        parser.error(f"--engine {requested!r} is not declared (declared: {', '.join(declared)})")
    command = args.subcommand
    kind = scope(command)
    if kind == ENGINE:
        if requested is None:
            parser.error(f"{command} needs --engine ({' | '.join(declared)})")
        args.engine = requested
    elif kind in (PLATFORM, SELF, LANE):
        if requested is not None:
            parser.error(f"{command} does not take the global --engine")
        if kind != SELF:
            args.engine = None
        return
    else:
        if requested is not None and requested != kind:
            parser.error(f"{command} is a {kind} tool; it does not act on --engine {requested}")
        args.engine = kind
