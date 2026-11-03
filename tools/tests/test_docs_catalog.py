"""Keep engine documentation aligned with the current repository inventory."""
from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "tools/docs/generate_engine_enhancement_summaries.py"


def test_generated_catalog_is_current() -> None:
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_every_package_has_an_indexed_doc() -> None:
    catalogs = {
        "llamacpp": ROOT / "docs/reference/engines/llamacpp/PATCHES.md",
        "radiance": ROOT / "docs/reference/engines/radiance/ENHANCEMENTS.md",
    }
    headings = ("Summary", "Outcome", "Detail")
    for engine, index in catalogs.items():
        text = index.read_text(encoding="utf-8")
        for metadata in sorted((ROOT / "engines" / engine / "patches").glob("*/patch.toml")):
            if metadata.parent.name.startswith("_"):
                continue
            doc = next(
                (metadata.parent / name for name in ("README.md", "SUMMARY.md")
                 if (metadata.parent / name).exists()),
                None,
            )
            assert doc is not None, metadata
            body = doc.read_text(encoding="utf-8")
            assert all(re.search(rf"^## {heading}$", body, re.M) for heading in headings), doc
            assert f"[{metadata.parent.name}]" in text, metadata

        for metadata in (ROOT / "engines" / engine / "components").glob("*/component.toml"):
            doc = metadata.parent / "README.md"
            assert doc.exists(), doc
            assert f"[{metadata.parent.name}]" in text
    assert "[_template]" not in catalogs["llamacpp"].read_text(encoding="utf-8")


def test_root_readme_pins_and_inventory_match_sources() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    radiance = tomllib.loads((ROOT / "engines/radiance/recipes.toml").read_text(encoding="utf-8"))
    llama = tomllib.loads((ROOT / "config/recipes.toml").read_text(encoding="utf-8"))

    assert f"**v{radiance['pinned-version']}**" in readme
    assert f"**v{radiance['pinned-version']}**" in (
        ROOT / "docs/reference/engines/radiance/README.md").read_text(encoding="utf-8")
    assert f"**{llama['pinned']}**" in readme

    for engine in ("radiance", "llamacpp"):
        patches = [p for p in (ROOT / "engines" / engine / "patches").glob("*/patch.toml")
                   if not p.parent.name.startswith("_")]
        count = len(patches)
        validated = sum(tomllib.loads(p.read_text(encoding="utf-8"))["state"] == "validated"
                        for p in patches)
        assert f"**{count} patches**" in readme
        if engine == "radiance":
            assert f"Radiance has {validated} validated patch" in readme
        else:
            assert f"llama.cpp has {validated}" in readme

    components = list((ROOT / "engines/radiance/components").glob("*/component.toml"))
    assert f"{len(components)} components" in readme


def test_current_entry_page_links() -> None:
    pages = [
        "README.md",
        "engines/radiance/README.md",
        "docs/reference/README.md",
        "docs/reference/START_HERE.md",
        "docs/reference/engines/radiance/README.md",
        "docs/reference/engines/radiance/RUNBOOK.md",
        "docs/reference/engines/radiance/PARITY.md",
        "docs/reference/engines/radiance/LAYOUTS.md",
        "docs/reference/engines/radiance/models/qwen3.8-flash-next.md",
        "docs/reference/engines/radiance/models/qwen3.8-27b.md",
        "docs/reference/engines/radiance/models/qwen3.6-35b-a3b.md",
    ]
    for path in pages:
        page = ROOT / path
        for match in re.finditer(r"(?<!!)\[[^]]+\]\(([^)]+)\)", page.read_text(encoding="utf-8")):
            destination = match.group(1).split("#", 1)[0]
            if not destination or destination.startswith(("http://", "https://", "mailto:")):
                continue
            assert (page.parent / unquote(destination)).exists(), f"{path}: {destination}"
