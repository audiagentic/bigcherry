"""Keep patch and component indexes complete and derived from their packages."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

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
        "llamacpp": (ROOT / "docs/reference/engines/llamacpp/PATCHES.md"),
        "radiance": (ROOT / "docs/reference/engines/radiance/ENHANCEMENTS.md"),
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
