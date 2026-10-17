"""Per-bump release notes (``bigcherry release-notes``).

One release per pin bump of an engine, named ``<tag-prefix><upstream tag>`` (the engine's ``[engine.<name>]`` table in config/release.toml; an engine without one has no release line yet). release-please on GitHub
owns the release PR, the tag and the GitHub release (release-please-config.json, .github/workflows/release-please.yml);
this module produces the release text it publishes. The notes are assembled from three sources, each bounded by the
previous release:

- the change ledger: events newer than the previous release tag's commit that touched the engine's files, grouped by
  change class;
- the patch registry: patches added, removed, whose state changed, or whose implementation changed between the
  previous release tag and the release commit, and which entered or left the production build;
- the bump's release record: its notes field (smoke numbers, regressions found and their disposition).

Everything is read from git objects and tracked files, so the same inputs always produce the same notes.
"""

from __future__ import annotations

import json
import subprocess
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..core import engines


class ReleaseNotesError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReleaseConfig:
    engine: str
    #: what the upstream project is called (llama.cpp), for the notes text
    upstream_name: str
    tag_prefix: str
    notes_dir: str
    ledger: str
    release_records: str
    #: repository-relative recipes file and patch tree of the engine
    recipes: str
    patches: str
    build_patch_sets: tuple[str, ...]
    fallback_prefixes: tuple[str, ...]
    #: repository paths (files or directories) whose changes belong to this engine: engines/<name>/, the engine.toml
    #: layout and the engine's declared extra paths; a ledger event listing a file under any of them is the engine's
    owned_paths: tuple[str, ...]
    ledger_sections: tuple[tuple[str, str], ...]  # (change-class, heading)


def load_config(repo_root: Path, engine: str) -> ReleaseConfig:
    """The release configuration of one engine: the shared part of config/release.toml, that engine's
    [engine.<name>] release line, and its recipes and patch tree from engine.toml."""
    raw = tomllib.loads((repo_root / "config" / "release.toml").read_text(encoding="utf-8"))
    if raw.get("version") != 2:
        raise ReleaseNotesError(f"config/release.toml: unsupported version {raw.get('version')!r}")
    line = raw.get("engine", {}).get(engine)
    if line is None:
        raise ReleaseNotesError(
            f"config/release.toml has no [engine.{engine}] release line: {engine} has no releases configured yet"
        )
    try:
        layout = engines.load(repo_root, engine)
    except engines.EngineError as exc:
        raise ReleaseNotesError(str(exc)) from exc
    return ReleaseConfig(
        engine=engine,
        upstream_name=layout.upstream_name,
        tag_prefix=line["tag-prefix"],
        notes_dir=raw["notes-dir"],
        ledger=raw["ledger"],
        release_records=raw["release-records"],
        recipes=layout.recipes,
        patches=layout.patches,
        build_patch_sets=tuple(line["build-patch-sets"]),
        fallback_prefixes=tuple(line["previous-tag-fallback-prefixes"]),
        owned_paths=engine_paths(layout, tuple(line["paths"])),
        ledger_sections=tuple((s["class"], s["heading"]) for s in raw["ledger-section"]),
    )


def engine_paths(layout: engines.EngineLayout, declared: tuple[str, ...]) -> tuple[str, ...]:
    """Every repository path an engine owns: its engines/<name>/ directory, the locations its engine.toml declares
    (patches, overlay, vendor checkout, recipes, components, external sources) and the extra paths of its release line."""
    owned = [f"{engines.ENGINES_DIR}/{layout.name}", layout.patches, layout.overlay, layout.vendor, layout.recipes,
             *(path for _, path in layout.components), layout.external_sources, *declared]
    return tuple(sorted({path.strip("/") for path in owned if path}))


def event_belongs(event: dict, config: ReleaseConfig) -> bool:
    """A ledger event belongs to an engine's release when any file it lists is under one of the engine's owned paths.
    Events carry no engine field; the files are the record of what a change touched. An event touching no engine's
    files (platform code, tools, docs, planning) belongs to the platform release line instead."""
    owned = config.owned_paths
    for name in event.get("files") or []:
        path = str(name).replace("\\", "/").removeprefix("./")
        if any(path == root or path.startswith(root + "/") for root in owned):
            return True
    return False


def _git(repo_root: Path, *args: str, check: bool = True) -> str:
    done = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, encoding="utf-8")
    if check and done.returncode != 0:
        raise ReleaseNotesError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def _show(repo_root: Path, ref: str, path: str) -> str | None:
    done = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=repo_root, capture_output=True, text=True,
                          encoding="utf-8")
    return done.stdout if done.returncode == 0 else None


def previous_release_tag(repo_root: Path, config: ReleaseConfig, release_tag: str, ref: str) -> str | None:
    """The newest release tag that is an ancestor of ``ref`` and is not this release, across the configured
    canonical prefix and every fallback family."""
    patterns = [f"{prefix}*" for prefix in (config.tag_prefix, *config.fallback_prefixes)]
    tags = _git(repo_root, "tag", "--merged", ref, "--sort=-creatordate", "--list", *patterns).split()
    return next((tag for tag in tags if tag != release_tag), None)


@dataclass
class PatchChanges:
    added: list[tuple[str, str]] = field(default_factory=list)            # (patch id, state)
    removed: list[str] = field(default_factory=list)
    state_changed: list[tuple[str, str, str]] = field(default_factory=list)  # (patch id, old, new)
    changed: list[str] = field(default_factory=list)                      # implementation changed, same state
    entered_build: list[str] = field(default_factory=list)
    left_build: list[str] = field(default_factory=list)


def _patch_ids(repo_root: Path, config: ReleaseConfig, ref: str) -> set[str]:
    listing = _git(repo_root, "ls-tree", "-d", "--name-only", f"{ref}:{config.patches}").split()
    return {name for name in listing if not name.startswith("_")}


def _patch_file(config: ReleaseConfig, patch_id: str, name: str) -> str:
    """Repository-relative path of one file of a patch package."""
    return f"{config.patches}/{patch_id}/{name}"


def _patch_state(repo_root: Path, config: ReleaseConfig, ref: str, patch_id: str) -> str:
    text = _show(repo_root, ref, _patch_file(config, patch_id, "patch.toml"))
    return tomllib.loads(text).get("state", "") if text else ""


def _build_members(repo_root: Path, config: ReleaseConfig, ref: str) -> set[str]:
    text = _show(repo_root, ref, config.recipes)
    if text is None:
        return set()
    sets = tomllib.loads(text).get("patch-set", {})
    return {patch for name in config.build_patch_sets for patch in sets.get(name, {}).get("patches", [])}


def patch_changes(repo_root: Path, config: ReleaseConfig, old_ref: str, new_ref: str) -> PatchChanges:
    old_ids, new_ids = _patch_ids(repo_root, config, old_ref), _patch_ids(repo_root, config, new_ref)
    changes = PatchChanges()
    changes.added = sorted((pid, _patch_state(repo_root, config, new_ref, pid)) for pid in new_ids - old_ids)
    changes.removed = sorted(old_ids - new_ids)
    prefix = config.patches + "/"
    touched = {
        line[len(prefix):].split("/")[0]
        for line in _git(repo_root, "diff", "--name-only", old_ref, new_ref, "--", config.patches).splitlines()
        if line.startswith(prefix) and line[len(prefix):].count("/") >= 1 and line.endswith(("patch.py", "patch.toml"))
    }
    for pid in sorted(touched & old_ids & new_ids):
        old_state = _patch_state(repo_root, config, old_ref, pid)
        new_state = _patch_state(repo_root, config, new_ref, pid)
        if old_state != new_state:
            changes.state_changed.append((pid, old_state, new_state))
        elif _show(repo_root, old_ref, _patch_file(config, pid, "patch.py")) != _show(
            repo_root, new_ref, _patch_file(config, pid, "patch.py")
        ):
            changes.changed.append(pid)
    old_build, new_build = _build_members(repo_root, config, old_ref), _build_members(repo_root, config, new_ref)
    changes.entered_build = sorted(new_build - old_build)
    changes.left_build = sorted(old_build - new_build)
    return changes


def ledger_events(repo_root: Path, config: ReleaseConfig, ref: str, after: datetime | None) -> list[dict]:
    text = _show(repo_root, ref, config.ledger) or ""
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        if event_belongs(event, config) and (after is None or _event_time(event) > after):
            events.append(event)
    return sorted(events, key=_event_time)


def _event_time(event: dict) -> datetime:
    """UTC time of a ledger event: its timestamp-utc, or for older events the time encoded in the event id
    (chg_YYYYMMDD_HHMMSS_...)."""
    stamp = event.get("timestamp-utc")
    if stamp:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    _, day, clock = event["event-id"].split("_")[:3]
    return datetime.strptime(day + clock, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def _patch_title(repo_root: Path, config: ReleaseConfig, ref: str, patch_id: str) -> str:
    """First sentence of the patch module docstring, as the one-line description."""
    text = _show(repo_root, ref, _patch_file(config, patch_id, "patch.py")) or ""
    if not text.startswith('"""'):
        return ""
    first = " ".join(text[3:text.find('"""', 3)].strip().split("\n\n")[0].split())
    return first


def release_version(llama_tag: str, version: str | None = None) -> str:
    """The release-please version of a release: <llama build>.<minor>.<patch>. The first release of a pin is
    <build>.0.0; later releases on the same pin raise minor or patch."""
    build = llama_tag[1:] if llama_tag[:1] == "b" and llama_tag[1:].isdigit() else None
    if build is None:
        raise ReleaseNotesError(f"{llama_tag!r} is not a numbered upstream build (b<number>): releases are versioned "
                                f"<build>.<minor>.<patch>, and a pin that is not a numbered build has no version scheme yet "
                                "(run-multi-engine MEN13 open decision)")
    if version is None:
        return f"{build}.0.0"
    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts) or parts[0] != build:
        raise ReleaseNotesError(f"version {version!r} must be {build}.<minor>.<patch> for llama.cpp {llama_tag}")
    return version


def render(repo_root: Path, config: ReleaseConfig, llama_tag: str, ref: str = "HEAD", version: str | None = None) -> str:
    release_tag = config.tag_prefix + release_version(llama_tag, version)
    record_text = _show(repo_root, ref, f"{config.release_records}/{llama_tag}.json")
    if record_text is None:
        raise ReleaseNotesError(f"no release record {config.release_records}/{llama_tag}.json at {ref}")
    record = json.loads(record_text)
    previous = previous_release_tag(repo_root, config, release_tag, ref)
    after = datetime.fromisoformat(_git(repo_root, "log", "-1", "--format=%cI", previous).strip()) if previous else None

    lines = [f"# {release_tag}", ""]
    lines.append(f"BigCherry on {config.upstream_name} `{llama_tag}` (`{record['revision'][:12]}`), "
                 f"BigCherry commit `{_git(repo_root, 'rev-parse', '--short=12', ref).strip()}`.")
    lines.append(f"Previous release: `{previous}`." if previous else "First tagged release.")
    lines.append("")

    notes = (record.get("notes") or "").strip()
    if notes:
        lines += ["## Bump record", "", notes, ""]

    if previous:
        changes = patch_changes(repo_root, config, previous, ref)
        lines += ["## Patches", ""]
        blocks = (
            ("New patches", [f"`{pid}` ({state}) - {_patch_title(repo_root, config, ref, pid)}" for pid, state in changes.added]),
            ("State changes", [f"`{pid}`: {old or 'none'} -> {new}" for pid, old, new in changes.state_changed]),
            ("Entered the production build", [f"`{pid}`" for pid in changes.entered_build]),
            ("Left the production build", [f"`{pid}`" for pid in changes.left_build]),
            ("Implementation changed", [f"`{pid}`" for pid in changes.changed]),
            ("Removed", [f"`{pid}`" for pid in changes.removed]),
        )
        any_block = False
        for heading, items in blocks:
            if items:
                any_block = True
                lines += [f"### {heading}", ""] + [f"- {item}" for item in items] + [""]
        if not any_block:
            lines += ["No patch changes.", ""]

    events = ledger_events(repo_root, config, ref, after)
    by_class: dict[str, list[dict]] = {}
    for event in events:
        by_class.setdefault(event["change-class"], []).append(event)
    for change_class, heading in config.ledger_sections:
        items = by_class.get(change_class, [])
        if items:
            lines += [f"## {heading}", ""]
            for event in items:
                plans = ", ".join(event.get("plan-item-ids") or [])
                lines.append(f"- {event['user-summary-candidate'].strip()}" + (f" ({plans})" if plans else ""))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_notes(repo_root: Path, config: ReleaseConfig, llama_tag: str, ref: str = "HEAD", version: str | None = None) -> Path:
    out = repo_root / config.notes_dir / f"{config.tag_prefix}{release_version(llama_tag, version)}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(repo_root, config, llama_tag, ref, version), encoding="utf-8", newline="\n")
    return out

