"""Regression coverage for BU01 radiance path selection."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("changed_paths", ROOT / ".github/scripts/changed_paths.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def selected(paths):
    patterns = [
        "engines/radiance/**",
        "!engines/radiance/**/*.md",
        "!engines/radiance/**/*.mdx",
        "!engines/radiance/**/docs/**",
        "tools/bigcherry/build/**",
    ]
    compiled = [(mod.glob_to_regex(p.lstrip("!")), not p.startswith("!")) for p in patterns]
    return any(mod.path_selected(path, compiled) for path in paths)


def test_radiance_markdown_and_docs_skip():
    assert not selected(["engines/radiance/README.md"])
    assert not selected(["engines/radiance/components/libr3/README.md"])
    assert not selected(["engines/radiance/components/docs/example.cpp"])
    assert not selected(["engines/radiance/docs/planning/BU01.md"])


def test_buildable_changes_still_run():
    for name in (
        "engines/radiance/components/libr3/CMakeLists.txt",
        "engines/radiance/components/libr3/src/kernel.hip",
        "engines/radiance/components/libr3/component.toml",
        "engines/radiance/patches/fix.patch",
        "engines/radiance/tests/component_test.py",
        "tools/bigcherry/build/recipe.py",
    ):
        assert selected([name])
    assert selected(["engines/radiance/README.md", "engines/radiance/patches/fix.patch"])


def test_last_match_wins_and_other_jobs_unchanged():
    match = mod.glob_to_regex
    rules = [(match("x/**"), True), (match("x/**/*.md"), False), (match("x/important.md"), True)]
    assert mod.path_selected("x/important.md", rules)
    assert not mod.path_selected("x/other.md", rules)
