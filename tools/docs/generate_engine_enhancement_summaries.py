#!/usr/bin/env python3
"""Build the per-engine patch and component catalogs from owning package docs.

Source of truth: patch.toml lifecycle states, component.toml declarations, and
README.md (or Radiance SUMMARY.md) Summary/Outcome/Detail sections.

Usage:
    python tools/docs/generate_engine_enhancement_summaries.py
    python tools/docs/generate_engine_enhancement_summaries.py --check
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BEGIN = "<!-- BEGIN GENERATED ENHANCEMENTS -->"
END = "<!-- END GENERATED ENHANCEMENTS -->"
HEADINGS = ("Summary", "Outcome", "Detail")


@dataclass(frozen=True)
class Entry:
    name: str
    state: str
    summary: str
    doc: Path


def section(text: str, heading: str) -> str:
    match = re.search(
        rf"^##\s+{re.escape(heading)}\s*$\n(.*?)(?=^##\s+|\Z)",
        text,
        re.M | re.S,
    )
    if not match:
        return ""
    for paragraph in re.split(r"\n\s*\n", match.group(1).strip()):
        lines = [line.strip() for line in paragraph.splitlines()]
        if lines and not lines[0].startswith(("<!--", "##", "- ", "|")):
            return re.sub(r"\s+", " ", " ".join(lines)).strip()
    return ""


def read_package(name: str, metadata: Path, candidates: tuple[str, ...], issues: list[str], state: str) -> Entry | None:
    doc = next((metadata.parent / n for n in candidates if (metadata.parent / n).is_file()), None)
    if doc is None:
        issues.append(f"missing package doc: {metadata.parent.relative_to(ROOT)}")
        return None
    text = doc.read_text(encoding="utf-8")
    missing = [heading for heading in HEADINGS if not re.search(rf"^##\s+{heading}\s*$", text, re.M)]
    if missing:
        issues.append(f"{doc.relative_to(ROOT)} missing headings: {', '.join(missing)}")
        return None
    summary = section(text, "Summary")
    outcome = section(text, "Outcome")
    if not summary or not outcome:
        issues.append(f"{doc.relative_to(ROOT)} has an empty Summary or Outcome")
        return None
    return Entry(name, state, summary, doc)


def patches(engine: str, issues: list[str]) -> list[Entry]:
    root = ROOT / "engines" / engine / "patches"
    entries = []
    for metadata in sorted(root.glob("*/patch.toml")):
        if metadata.parent.name.startswith("_"):
            continue
        attrs = tomllib.loads(metadata.read_text(encoding="utf-8"))
        name = metadata.parent.name
        if attrs.get("id") != name:
            issues.append(f"{metadata.relative_to(ROOT)} package id does not match its directory")
        candidates = ("SUMMARY.md", "README.md") if engine == "radiance" else ("README.md", "SUMMARY.md")
        entry = read_package(name, metadata, candidates, issues, attrs.get("state", "unknown"))
        if entry:
            entries.append(entry)
    return entries


def components(engine: str, issues: list[str]) -> list[Entry]:
    root = ROOT / "engines" / engine / "components"
    entries = []
    for metadata in sorted(root.glob("*/component.toml")):
        attrs = tomllib.loads(metadata.read_text(encoding="utf-8"))
        name = metadata.parent.name
        if attrs.get("name") != name:
            issues.append(f"{metadata.relative_to(ROOT)} component name does not match directory")
        entry = read_package(name, metadata, ("README.md", "SUMMARY.md"), issues, "See settings")
        if entry:
            entries.append(entry)
    return entries


def rel_link(doc: Path, base: Path) -> str:
    target = os.path.relpath(doc, base.parent).replace(os.sep, "/")
    return f"[{doc.parent.name}]({target})"


def rows(entries: list[Entry], base: Path) -> list[str]:
    lines = ["| Package | State | Summary |", "| --- | --- | --- |"]
    for entry in entries:
        summary = entry.summary.replace("|", r"\|")
        state = entry.state.replace("|", r"\|")
        display_state = "*See component settings*" if state == "See settings" else f"`{state}`"
        lines.append(f"| {rel_link(entry.doc, base)} | {display_state} | {summary} |")
    return lines


def contents(target: Path, patch_entries: list[Entry], component_entries: list[Entry]) -> str:
    lines = [
        BEGIN, "",
        "_Generated from package documentation and metadata by "
        "`tools/docs/generate_engine_enhancement_summaries.py`. "
        "Do not edit this section by hand._", "",
        f"## Engine patches ({len(patch_entries)})", "",
        *rows(patch_entries, target), "",
    ]
    if component_entries:
        lines += [f"## Components ({len(component_entries)})", "", *rows(component_entries, target), ""]
    lines += [END]
    return "\n".join(lines)


def apply(target: Path, patch_entries: list[Entry], component_entries: list[Entry], check: bool, issues: list[str]) -> None:
    if not target.exists():
        issues.append(f"missing catalog: {target.relative_to(ROOT)}")
        return
    current = target.read_text(encoding="utf-8")
    if current.count(BEGIN) != 1 or current.count(END) != 1:
        issues.append(f"missing or duplicate catalog markers: {target.relative_to(ROOT)}")
        return
    expected = re.sub(
        re.escape(BEGIN) + r".*?" + re.escape(END),
        lambda _: contents(target, patch_entries, component_entries),
        current,
        count=1,
        flags=re.S,
    )
    if expected == current:
        print(f"unchanged {target.relative_to(ROOT)}")
    elif check:
        issues.append(f"stale catalog: {target.relative_to(ROOT)} (regenerate without --check)")
    else:
        target.write_text(expected, encoding="utf-8")
        print(f"updated {target.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify docs and generated indexes without writing")
    args = parser.parse_args()
    issues: list[str] = []

    llama = ROOT / "docs/reference/engines/llamacpp/PATCHES.md"
    apply(llama, patches("llamacpp", issues), [], args.check, issues)

    radiance = ROOT / "docs/reference/engines/radiance/ENHANCEMENTS.md"
    apply(radiance, patches("radiance", issues), components("radiance", issues), args.check, issues)

    for issue in issues:
        print(f"ERROR: {issue}", file=sys.stderr)
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
